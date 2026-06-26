#include "PINNPredictor.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <cassert>
#include <cstdlib>

// Fixed vertex indices (FixedConstraint in the SOFA scene)
const int PINNPredictor::FIXED[3] = {3, 39, 64};

// ── Constructor ───────────────────────────────────────────────────────────────
PINNPredictor::PINNPredictor()
    : device_(torch::kCPU)  // always CPU — avoids CUDA allocator/glibc tcache conflicts
{
    std::memset(buf_ddx_, 0, sizeof(buf_ddx_));
    std::memset(buf_ddy_, 0, sizeof(buf_ddy_));
    std::memset(buf_ddz_, 0, sizeof(buf_ddz_));
    std::memset(buf_sax_, 0, sizeof(buf_sax_));
    std::memset(buf_say_, 0, sizeof(buf_say_));
    std::memset(buf_saz_, 0, sizeof(buf_saz_));
    std::memset(buf_rxx_, 0, sizeof(buf_rxx_));
    std::memset(buf_ryy_, 0, sizeof(buf_ryy_));
    std::memset(buf_rzz_, 0, sizeof(buf_rzz_));
    std::memset(tool_hist_, 0, sizeof(tool_hist_));
    std::memset(rt_hist_,   0, sizeof(rt_hist_));
    std::memset(last_sax_, 0, sizeof(last_sax_));
    std::memset(last_say_, 0, sizeof(last_say_));
    std::memset(last_saz_, 0, sizeof(last_saz_));
    std::memset(last_rxx_, 0, sizeof(last_rxx_));
    std::memset(last_ryy_, 0, sizeof(last_ryy_));
    std::memset(last_rzz_, 0, sizeof(last_rzz_));
    std::memset(pred_ddx_, 0, sizeof(pred_ddx_));
    std::memset(pred_ddy_, 0, sizeof(pred_ddy_));
    std::memset(pred_ddz_, 0, sizeof(pred_ddz_));
    std::memset(vert_pos_, 0, sizeof(vert_pos_));

    // Build active vertex list (all except FIXED)
    int k = 0;
    for (int i = 0; i < N_V; ++i)
    {
        bool fixed = false;
        for (int j = 0; j < 3; ++j)
            if (FIXED[j] == i) { fixed = true; break; }
        if (!fixed)
            active_verts_[k++] = i;
    }
    assert(k == N_ACTIVE);
}

// ── init ──────────────────────────────────────────────────────────────────────
bool PINNPredictor::init(const std::string& model_pt,
                          const std::string& norm_csv,
                          const std::string& vert_csv)
{
    std::cout << "[PINNPredictor] Loading model from " << model_pt << std::endl;
    try {
        model_ = torch::jit::load(model_pt, device_);
        model_.eval();
    } catch (const c10::Error& e) {
        std::cerr << "[PINNPredictor] Failed to load model: " << e.what() << std::endl;
        return false;
    }

    if (!loadNormCSV(norm_csv))   return false;
    if (!loadVerticesCSV(vert_csv)) return false;

    // Warm-up: one forward pass to trigger JIT compilation
    torch::InferenceMode guard;
    auto dummy = torch::zeros({1, N_IN}, torch::kFloat32);
    model_.forward({dummy});

    initialized_ = true;
    std::cout << "[PINNPredictor] Ready. Device: "
              << (device_.type() == torch::kCUDA ? "CUDA" : "CPU") << std::endl;
    return true;
}

// ── loadNormCSV ───────────────────────────────────────────────────────────────
bool PINNPredictor::loadNormCSV(const std::string& path)
{
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "[PINNPredictor] Cannot open norm CSV: " << path << std::endl;
        return false;
    }
    auto parseLine = [](const std::string& line, std::vector<float>& out) {
        std::stringstream ss(line);
        std::string cell;
        std::getline(ss, cell, ','); // skip key
        while (std::getline(ss, cell, ','))
            if (!cell.empty()) out.push_back(std::stof(cell));
    };

    std::string line;
    while (std::getline(f, line)) {
        if      (line.rfind("x_mean", 0) == 0) parseLine(line, X_mean_);
        else if (line.rfind("x_std",  0) == 0) parseLine(line, X_std_);
        else if (line.rfind("y_mean", 0) == 0) parseLine(line, Y_mean_);
        else if (line.rfind("y_std",  0) == 0) parseLine(line, Y_std_);
    }
    if ((int)X_mean_.size() != N_IN || (int)X_std_.size() != N_IN ||
        (int)Y_mean_.size() != N_OUT || (int)Y_std_.size() != N_OUT) {
        std::cerr << "[PINNPredictor] Norm CSV shape mismatch: "
                  << "X_mean=" << X_mean_.size() << " (want " << N_IN << "), "
                  << "Y_mean=" << Y_mean_.size() << " (want " << N_OUT << ")" << std::endl;
        return false;
    }
    std::cout << "[PINNPredictor] Normalization stats loaded." << std::endl;
    return true;
}

// ── loadVerticesCSV ───────────────────────────────────────────────────────────
bool PINNPredictor::loadVerticesCSV(const std::string& path)
{
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "[PINNPredictor] Cannot open vertices CSV: " << path << std::endl;
        return false;
    }
    std::string line;
    std::getline(f, line);
    int n = std::stoi(line);
    if (n != N_V) {
        std::cerr << "[PINNPredictor] Vertices count mismatch: " << n << " vs " << N_V << std::endl;
        return false;
    }
    for (int i = 0; i < N_V; ++i) {
        std::getline(f, line);
        std::stringstream ss(line);
        std::string v;
        std::getline(ss, v, ','); vert_pos_[i][0] = std::stof(v);
        std::getline(ss, v, ','); vert_pos_[i][1] = std::stof(v);
        std::getline(ss, v, ','); vert_pos_[i][2] = std::stof(v);
    }
    std::cout << "[PINNPredictor] Loaded " << N_V << " vertex positions." << std::endl;
    return true;
}

// ── updateFEM ─────────────────────────────────────────────────────────────────
void PINNPredictor::updateFEM(const float* ddx, const float* ddy, const float* ddz,
                               const float* sax, const float* say, const float* saz,
                               const float* rxx, const float* ryy, const float* rzz)
{
    std::memcpy(fem_ddx_, ddx, N_V * sizeof(float));
    std::memcpy(fem_ddy_, ddy, N_V * sizeof(float));
    std::memcpy(fem_ddz_, ddz, N_V * sizeof(float));
    std::memcpy(fem_sax_, sax, N_V * sizeof(float));
    std::memcpy(fem_say_, say, N_V * sizeof(float));
    std::memcpy(fem_saz_, saz, N_V * sizeof(float));
    std::memcpy(fem_rxx_, rxx, N_V * sizeof(float));
    std::memcpy(fem_ryy_, ryy, N_V * sizeof(float));
    std::memcpy(fem_rzz_, rzz, N_V * sizeof(float));

    // Update last known real stress/strain for future fallback
    std::memcpy(last_sax_, sax, N_V * sizeof(float));
    std::memcpy(last_say_, say, N_V * sizeof(float));
    std::memcpy(last_saz_, saz, N_V * sizeof(float));
    std::memcpy(last_rxx_, rxx, N_V * sizeof(float));
    std::memcpy(last_ryy_, ryy, N_V * sizeof(float));
    std::memcpy(last_rzz_, rzz, N_V * sizeof(float));

    fem_pending_ = true;
}

// ── predictForce ─────────────────────────────────────────────────────────────
std::array<float, 3> PINNPredictor::predictForce(
    float tx,  float ty,  float tz,
    float tvx, float tvy, float tvz,
    float prev_fx, float prev_fy, float prev_fz,
    float min_dist, double real_time)
{
    if (!initialized_) return {0.f, 0.f, 0.f};

    // Guard: NaN/inf tool position causes std::partial_sort undefined behaviour
    if (!std::isfinite(tx) || !std::isfinite(ty) || !std::isfinite(tz))
        return {0.f, 0.f, 0.f};

    // 1. Decide what goes into lag1 of the buffer
    const float *new_ddx, *new_ddy, *new_ddz;
    const float *new_sax, *new_say, *new_saz;
    const float *new_rxx, *new_ryy, *new_rzz;

    if (fem_pending_) {
        // Real FEM result arrived — use it
        new_ddx = fem_ddx_; new_ddy = fem_ddy_; new_ddz = fem_ddz_;
        new_sax = fem_sax_; new_say = fem_say_; new_saz = fem_saz_;
        new_rxx = fem_rxx_; new_ryy = fem_ryy_; new_rzz = fem_rzz_;
        fem_pending_ = false;
    } else {
        // FEM still computing — use PINN's own predicted deform + last known stress/strain
        new_ddx = pred_ddx_; new_ddy = pred_ddy_; new_ddz = pred_ddz_;
        new_sax = last_sax_; new_say = last_say_; new_saz = last_saz_;
        new_rxx = last_rxx_; new_ryy = last_ryy_; new_rzz = last_rzz_;
    }

    // 2. Shift buffers (push new data to lag1, drop lag5)
    shiftBuffers(new_ddx, new_ddy, new_ddz,
                 new_sax, new_say, new_saz,
                 new_rxx, new_ryy, new_rzz,
                 tx, ty, tz, tvx, tvy, tvz,
                 prev_fx, prev_fy, prev_fz, real_time);

    // 3. Find 20 nearest neighbours to current tool position
    int nb[N_NB];
    findNeighbours(tx, ty, tz, nb);

    // 4. Contact features
    float delta_min_dist = first_step_ ? 0.f : (min_dist - prev_min_dist_);
    float lag1_contact   = (prev_force_mag_ > 0.01f) ? 1.f : 0.f;
    prev_min_dist_       = min_dist;
    float cur_f_mag = std::sqrt(prev_fx*prev_fx + prev_fy*prev_fy + prev_fz*prev_fz);
    prev_force_mag_  = cur_f_mag;
    first_step_      = false;

    // 5. Build 960-dim feature vector
    float X[N_IN];
    buildFeatureVector(X, tx, ty, tz, tvx, tvy, tvz,
                       prev_fx, prev_fy, prev_fz,
                       min_dist, delta_min_dist, lag1_contact, real_time, nb);

    // 6. Normalize
    float X_norm[N_IN];
    for (int i = 0; i < N_IN; ++i)
        X_norm[i] = (X[i] - X_mean_[i]) / (X_std_[i] + 1e-8f);

    // ── DEBUG: one-shot raw feature dump for direct comparison against Python ───
    {
        static bool dumped = false;
        static int call_count = 0;
        ++call_count;
        if (!dumped && call_count == 50)
        {
            dumped = true;
            std::ofstream dbg("/tmp/cpp_feature_dump.csv");
            dbg << "idx,raw,mean,std,norm\n";
            for (int i = 0; i < N_IN; ++i)
                dbg << i << "," << X[i] << "," << X_mean_[i] << "," << X_std_[i] << "," << X_norm[i] << "\n";
            dbg << "tx," << tx << "\nty," << ty << "\ntz," << tz << "\n";
            dbg << "tvx," << tvx << "\ntvy," << tvy << "\ntvz," << tvz << "\n";
            dbg << "prev_fx," << prev_fx << "\nprev_fy," << prev_fy << "\nprev_fz," << prev_fz << "\n";
            dbg << "min_dist," << min_dist << "\nreal_time," << real_time << "\n";
            std::cerr << "[PINN-DEBUG] Dumped first feature vector to /tmp/cpp_feature_dump.csv" << std::endl;
        }
    }

    // 7. Run model
    static int s_pred = 0;
    ++s_pred;
    const bool log = (s_pred <= 3 || s_pred % 500 == 0 || s_pred >= 3490);

    // ── DIAG-5: scan X_norm for NaN/inf and report the distribution ───────────
    {
        float x_min = X_norm[0], x_max = X_norm[0];
        int nan_idx = -1;
        for (int i = 0; i < N_IN; ++i) {
            if (!std::isfinite(X_norm[i]) && nan_idx < 0) nan_idx = i;
            x_min = std::min(x_min, X_norm[i]);
            x_max = std::max(x_max, X_norm[i]);
        }
        if (nan_idx >= 0)
            std::cerr << "[PINN-GUARD] predictForce #" << s_pred
                      << " NaN/inf in X_norm at idx=" << nan_idx
                      << " X=" << X[nan_idx] << " mean=" << X_mean_[nan_idx]
                      << " std=" << X_std_[nan_idx]
                      << " — returning {0,0,0}" << std::endl;
        if (log)
            std::cerr << "[PINN] predictForce #" << s_pred
                      << " X_norm range=[" << x_min << "," << x_max << "]"
                      << (std::abs(x_min) > 10.f || x_max > 10.f
                          ? "  << EXTREME — out-of-dist, expect NaN output" : "")
                      << std::endl;
        if (nan_idx >= 0) return {0.f, 0.f, 0.f};
    }

    if (log) std::cerr << "[PINN] predictForce #" << s_pred << " BEFORE model.forward" << std::endl;

    // Set PINN_SKIP_INFERENCE=1 to return zeros without running the model (crash diagnostic)
    if (std::getenv("PINN_SKIP_INFERENCE"))
    {
        if (log) std::cerr << "[PINN] predictForce #" << s_pred << " SKIPPED (PINN_SKIP_INFERENCE set)" << std::endl;
        return {0.f, 0.f, 0.f};
    }

    torch::InferenceMode guard;
    auto input = torch::from_blob(X_norm, {1, N_IN}, torch::kFloat32).clone();
    auto output = model_.forward({input}).toTensor().contiguous();
    const float* raw = output.data_ptr<float>();

    if (log) std::cerr << "[PINN] predictForce #" << s_pred << " AFTER model.forward" << std::endl;

    // 8. Denormalize all outputs
    float Y_pred[N_OUT];
    for (int i = 0; i < N_OUT; ++i)
        Y_pred[i] = raw[i] * Y_std_[i] + Y_mean_[i];

    // 9. Decode force — invert log1p compression
    std::array<float, 3> force;
    for (int i = 0; i < N_FORCE; ++i)
        force[i] = expm1_signed(Y_pred[i]);

    // 10. Store PINN's predicted deform for next tick (if FEM not ready)
    decodeDeform(Y_pred + N_FORCE, pred_ddx_, pred_ddy_, pred_ddz_);

    return force;
}

// ── findNeighbours ────────────────────────────────────────────────────────────
void PINNPredictor::findNeighbours(float tx, float ty, float tz, int* nb_out) const
{
    // Brute-force over 181 vertices — fast enough (181 ops per haptic tick)
    float dists[N_V];
    for (int i = 0; i < N_V; ++i) {
        float dx = vert_pos_[i][0] - tx;
        float dy = vert_pos_[i][1] - ty;
        float dz = vert_pos_[i][2] - tz;
        dists[i] = dx*dx + dy*dy + dz*dz;
    }
    // Partial sort: find N_NB smallest
    int idx[N_V];
    for (int i = 0; i < N_V; ++i) idx[i] = i;
    std::partial_sort(idx, idx + N_NB, idx + N_V,
                      [&](int a, int b){ return dists[a] < dists[b]; });
    for (int i = 0; i < N_NB; ++i) nb_out[i] = idx[i];
}

// ── buildFeatureVector ────────────────────────────────────────────────────────
// Must exactly match the ordering in train_pinn_contactweight.py (n_lags=8, the
// sweep-confirmed sweet spot):
//   dt_cum(N_LAGS) + dt_pred(1) + pos_vel(6) + contact(3) + tool_hist(9*N_LAGS)
//   + nb_deform/stress/strain(60*N_LAGS each) + accel(3) = 190*N_LAGS+13 = 1533
void PINNPredictor::buildFeatureVector(
    float* X,
    float tx, float ty, float tz,
    float tvx, float tvy, float tvz,
    float prev_fx, float prev_fy, float prev_fz,
    float min_dist, float delta_min_dist, float lag1_contact,
    double real_time, const int* nb) const
{
    int idx = 0;

    // --- dt_cum (5): rt[lag] - rt[lag5_base] for lag 1..5 ---
    // rt_hist_[0]=lag1 (most recent), rt_hist_[4]=lag5 (oldest = base)
    double base_time = rt_hist_[N_LAGS - 1];
    for (int lag = 0; lag < N_LAGS; ++lag) {
        double dt = rt_hist_[lag] - base_time;
        X[idx++] = (float)std::min(std::max(dt, 0.0), 0.15);
    }

    // --- dt_pred (1): current_time - lag5_base ---
    float dt_pred = (float)std::min(std::max(real_time - base_time, 0.0), 0.15);
    X[idx++] = dt_pred;

    // --- current_pos_vel (6) ---
    X[idx++] = tx;  X[idx++] = ty;  X[idx++] = tz;
    X[idx++] = tvx; X[idx++] = tvy; X[idx++] = tvz;

    // --- contact_features (3) ---
    X[idx++] = min_dist;
    X[idx++] = delta_min_dist;
    X[idx++] = lag1_contact;

    // --- tool_hist (45): lag1..5 × [x,y,z,vx,vy,vz,log_fx,log_fy,log_fz] ---
    for (int lag = 0; lag < N_LAGS; ++lag) {
        for (int k = 0; k < 9; ++k)
            X[idx++] = tool_hist_[lag][k];
    }

    // --- nb_deform (300): lag1..5 × (ddx[nb×20] + ddy[nb×20] + ddz[nb×20]) ---
    for (int lag = 0; lag < N_LAGS; ++lag) {
        for (int n = 0; n < N_NB; ++n) X[idx++] = buf_ddx_[lag][nb[n]];
        for (int n = 0; n < N_NB; ++n) X[idx++] = buf_ddy_[lag][nb[n]];
        for (int n = 0; n < N_NB; ++n) X[idx++] = buf_ddz_[lag][nb[n]];
    }

    // --- nb_stress (300): lag1..5 × (sax[nb] + say[nb] + saz[nb]) ---
    for (int lag = 0; lag < N_LAGS; ++lag) {
        for (int n = 0; n < N_NB; ++n) X[idx++] = buf_sax_[lag][nb[n]];
        for (int n = 0; n < N_NB; ++n) X[idx++] = buf_say_[lag][nb[n]];
        for (int n = 0; n < N_NB; ++n) X[idx++] = buf_saz_[lag][nb[n]];
    }

    // --- nb_realstrain (300): lag1..5 × (rxx[nb] + ryy[nb] + rzz[nb]) ---
    for (int lag = 0; lag < N_LAGS; ++lag) {
        for (int n = 0; n < N_NB; ++n) X[idx++] = buf_rxx_[lag][nb[n]];
        for (int n = 0; n < N_NB; ++n) X[idx++] = buf_ryy_[lag][nb[n]];
        for (int n = 0; n < N_NB; ++n) X[idx++] = buf_rzz_[lag][nb[n]];
    }

    // --- acceleration (3): wide-baseline (lag1 vs lag3) velocity difference,
    // matching training exactly: accel = (v[lag1] - v[lag3]) / (rt[lag1] - rt[lag3]),
    // zero if that interval is degenerate (not enough real history yet). Adjacent-step
    // differencing was deliberately avoided in training (amplifies sensor noise); this
    // wider baseline must be replicated bit-for-bit or the feature means something
    // different here than what the model learned.
    {
        const float dt_accel = (float)(rt_hist_[0] - rt_hist_[2]);
        float ax = 0.f, ay = 0.f, az = 0.f;
        if (dt_accel > 1e-4f) {
            ax = (tool_hist_[0][3] - tool_hist_[2][3]) / dt_accel;
            ay = (tool_hist_[0][4] - tool_hist_[2][4]) / dt_accel;
            az = (tool_hist_[0][5] - tool_hist_[2][5]) / dt_accel;
        }
        ax = std::min(std::max(ax, -500.f), 500.f);
        ay = std::min(std::max(ay, -500.f), 500.f);
        az = std::min(std::max(az, -500.f), 500.f);
        X[idx++] = ax; X[idx++] = ay; X[idx++] = az;
    }

    assert(idx == N_IN);  // must be exactly 1533 (190*N_LAGS+13, N_LAGS=8)
}

// ── shiftBuffers ──────────────────────────────────────────────────────────────
// Push new data into lag1 (index 0); lag5 (index 4) falls off.
void PINNPredictor::shiftBuffers(
    const float* new_ddx, const float* new_ddy, const float* new_ddz,
    const float* new_sax, const float* new_say, const float* new_saz,
    const float* new_rxx, const float* new_ryy, const float* new_rzz,
    float tx, float ty, float tz,
    float tvx, float tvy, float tvz,
    float fx, float fy, float fz,
    double rt)
{
    // Shift lag4→lag5, lag3→lag4, ..., lag1→lag2
    for (int lag = N_LAGS - 1; lag > 0; --lag) {
        std::memcpy(buf_ddx_[lag], buf_ddx_[lag-1], N_V * sizeof(float));
        std::memcpy(buf_ddy_[lag], buf_ddy_[lag-1], N_V * sizeof(float));
        std::memcpy(buf_ddz_[lag], buf_ddz_[lag-1], N_V * sizeof(float));
        std::memcpy(buf_sax_[lag], buf_sax_[lag-1], N_V * sizeof(float));
        std::memcpy(buf_say_[lag], buf_say_[lag-1], N_V * sizeof(float));
        std::memcpy(buf_saz_[lag], buf_saz_[lag-1], N_V * sizeof(float));
        std::memcpy(buf_rxx_[lag], buf_rxx_[lag-1], N_V * sizeof(float));
        std::memcpy(buf_ryy_[lag], buf_ryy_[lag-1], N_V * sizeof(float));
        std::memcpy(buf_rzz_[lag], buf_rzz_[lag-1], N_V * sizeof(float));
        std::memcpy(tool_hist_[lag], tool_hist_[lag-1], 9 * sizeof(float));
        rt_hist_[lag] = rt_hist_[lag-1];
    }

    // New data becomes lag1 (index 0)
    std::memcpy(buf_ddx_[0], new_ddx, N_V * sizeof(float));
    std::memcpy(buf_ddy_[0], new_ddy, N_V * sizeof(float));
    std::memcpy(buf_ddz_[0], new_ddz, N_V * sizeof(float));
    std::memcpy(buf_sax_[0], new_sax, N_V * sizeof(float));
    std::memcpy(buf_say_[0], new_say, N_V * sizeof(float));
    std::memcpy(buf_saz_[0], new_saz, N_V * sizeof(float));
    std::memcpy(buf_rxx_[0], new_rxx, N_V * sizeof(float));
    std::memcpy(buf_ryy_[0], new_ryy, N_V * sizeof(float));
    std::memcpy(buf_rzz_[0], new_rzz, N_V * sizeof(float));

    tool_hist_[0][0] = tx;  tool_hist_[0][1] = ty;  tool_hist_[0][2] = tz;
    tool_hist_[0][3] = tvx; tool_hist_[0][4] = tvy; tool_hist_[0][5] = tvz;
    // Guard against NaN forces: a NaN here would poison tool_hist and cascade forever
    tool_hist_[0][6] = std::isfinite(fx) ? log1p_signed(fx) : 0.f;
    tool_hist_[0][7] = std::isfinite(fy) ? log1p_signed(fy) : 0.f;
    tool_hist_[0][8] = std::isfinite(fz) ? log1p_signed(fz) : 0.f;
    rt_hist_[0] = rt;
}

// ── decodeDeform ──────────────────────────────────────────────────────────────
// Maps 534-dim output (178 active verts × 3, interleaved ddx/ddy/ddz per vertex)
// back to per-vertex arrays of length 181. Fixed vertices stay at 0.
void PINNPredictor::decodeDeform(const float* pred_534,
                                  float* out_ddx, float* out_ddy, float* out_ddz) const
{
    std::memset(out_ddx, 0, N_V * sizeof(float));
    std::memset(out_ddy, 0, N_V * sizeof(float));
    std::memset(out_ddz, 0, N_V * sizeof(float));
    for (int k = 0; k < N_ACTIVE; ++k) {
        int vi = active_verts_[k];
        out_ddx[vi] = pred_534[k * 3 + 0];
        out_ddy[vi] = pred_534[k * 3 + 1];
        out_ddz[vi] = pred_534[k * 3 + 2];
    }
}
