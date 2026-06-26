#pragma once
/**
 * PINNPredictor — Real-time PINN force prediction for haptic feedback.
 *
 * Maintains a 5-lag rolling buffer of tissue state (deformation, stress, strain).
 * Buffer is updated with real FEM results when available; falls back to
 * PINN's own predicted deformation when FEM hasn't finished yet.
 *
 * Usage (from LCPForceFeedback.inl):
 *   1. Call init() once in LCPForceFeedback::init()
 *   2. Call updateFEM() every SOFA FEM step (in handleEvent/AnimateEndEvent)
 *   3. Call predictForce() every haptic tick (in doComputeForce)
 */

#include <array>
#include <string>
#include <vector>
#include <cmath>
#include <torch/script.h>
#include <torch/torch.h>

class PINNPredictor
{
public:
    static constexpr int N_V        = 181;   // total liver vertices
    static constexpr int N_LAGS     = 8;     // history window — found via sweep to be the
                                              // sweet spot (5 underfits sustained contact,
                                              // 12 overfits and hurts worst-case force error)
    static constexpr int N_NB       = 20;    // nearest neighbours
    static constexpr int N_IN       = 1533;  // model input dim: 190*N_LAGS + 13
                                              // (dt_cum(N_LAGS) + dt_pred(1) + pos_vel(6) +
                                              // contact(3) + tool_hist(9*N_LAGS) +
                                              // nb_deform/stress/strain(60*N_LAGS each) + accel(3))
    static constexpr int N_OUT      = 537;   // model output dim (3 force + 534 deform)
    static constexpr int N_FORCE    = 3;
    static constexpr int N_DEFORM   = 534;   // 178 active verts × 3
    static constexpr int N_ACTIVE   = 178;   // N_V - 3 fixed
    static const     int FIXED[3];           // {3, 39, 64} — defined in .cpp

    PINNPredictor();
    ~PINNPredictor() = default;

    /**
     * Load TorchScript model, normalization CSV, and vertex positions CSV.
     * Must be called once before any predictions.
     */
    bool init(const std::string& model_pt_path,
              const std::string& norm_csv_path,
              const std::string& vertices_csv_path);

    /**
     * Called from SOFA FEM thread after each simulation step completes.
     * ddx/ddy/ddz: delta deformation per vertex (curPos - restPos) - (prevPos - restPos)
     * sax/say/saz: contact-proxy EMA per vertex (freePos - curPos), smoothed
     * rxx/ryy/rzz: real Hooke's-law strain diagonal, per vertex
     * All arrays are of length N_V=181.
     */
    void updateFEM(const float* ddx, const float* ddy, const float* ddz,
                   const float* sax, const float* say, const float* saz,
                   const float* rxx, const float* ryy, const float* rzz);

    /**
     * Called from haptic thread every tick.
     * Returns predicted [fx, fy, fz] in Newtons.
     *
     * prev_fx/fy/fz: force from the PREVIOUS haptic tick (log1p-compressed input)
     * min_dist:      distance from tool tip to nearest liver vertex
     * real_time:     wall-clock elapsed seconds since simulation started
     */
    std::array<float, 3> predictForce(
        float tx,  float ty,  float tz,
        float tvx, float tvy, float tvz,
        float prev_fx, float prev_fy, float prev_fz,
        float min_dist, double real_time);

    bool isInitialized() const { return initialized_; }

private:
    // ── Model ────────────────────────────────────────────────────────────────
    torch::jit::script::Module model_;
    torch::Device device_;
    bool initialized_ = false;

    // ── Normalization (loaded from CSV) ──────────────────────────────────────
    std::vector<float> X_mean_, X_std_;   // length N_IN
    std::vector<float> Y_mean_, Y_std_;   // length N_OUT

    // ── Vertex rest positions (N_V × 3) for neighbour lookup ─────────────────
    float vert_pos_[N_V][3];
    int   active_verts_[N_ACTIVE];        // indices NOT in FIXED[]

    // ── Rolling buffers: index 0 = lag1 (most recent), 4 = lag5 (oldest/base) ─
    float buf_ddx_[N_LAGS][N_V];
    float buf_ddy_[N_LAGS][N_V];
    float buf_ddz_[N_LAGS][N_V];
    float buf_sax_[N_LAGS][N_V];
    float buf_say_[N_LAGS][N_V];
    float buf_saz_[N_LAGS][N_V];
    float buf_rxx_[N_LAGS][N_V];
    float buf_ryy_[N_LAGS][N_V];
    float buf_rzz_[N_LAGS][N_V];

    // tool_hist_[lag][9] = {x,y,z,vx,vy,vz,log_fx,log_fy,log_fz}
    float  tool_hist_[N_LAGS][9];
    double rt_hist_[N_LAGS];

    // ── Last known real FEM stress/strain (held when FEM not ready) ───────────
    float last_sax_[N_V], last_say_[N_V], last_saz_[N_V];
    float last_rxx_[N_V], last_ryy_[N_V], last_rzz_[N_V];

    // ── PINN's last predicted deform (used in buffer when FEM not ready) ──────
    float pred_ddx_[N_V], pred_ddy_[N_V], pred_ddz_[N_V];

    // ── Pending FEM data (set by updateFEM, consumed by predictForce) ─────────
    float fem_ddx_[N_V], fem_ddy_[N_V], fem_ddz_[N_V];
    float fem_sax_[N_V], fem_say_[N_V], fem_saz_[N_V];
    float fem_rxx_[N_V], fem_ryy_[N_V], fem_rzz_[N_V];
    bool  fem_pending_ = false;

    // ── State for contact features ────────────────────────────────────────────
    float prev_min_dist_  = 0.0f;
    float prev_force_mag_ = 0.0f;
    bool  first_step_     = true;

    // ── Helpers ───────────────────────────────────────────────────────────────
    bool loadNormCSV(const std::string& path);
    bool loadVerticesCSV(const std::string& path);

    void findNeighbours(float tx, float ty, float tz, int* nb_out) const;

    void buildFeatureVector(float* X_out,
                            float tx,  float ty,  float tz,
                            float tvx, float tvy, float tvz,
                            float prev_fx, float prev_fy, float prev_fz,
                            float min_dist, float delta_min_dist,
                            float lag1_contact, double real_time,
                            const int* nb) const;

    void shiftBuffers(const float* new_ddx, const float* new_ddy, const float* new_ddz,
                      const float* new_sax, const float* new_say, const float* new_saz,
                      const float* new_rxx, const float* new_ryy, const float* new_rzz,
                      float tx, float ty, float tz,
                      float tvx, float tvy, float tvz,
                      float fx, float fy, float fz,
                      double rt);

    void decodeDeform(const float* pred_deform_534,
                      float* out_ddx, float* out_ddy, float* out_ddz) const;

    static float log1p_signed(float x)
    {
        return (x >= 0.f ? 1.f : -1.f) * std::log1p(std::abs(x));
    }
    static float expm1_signed(float x)
    {
        return (x >= 0.f ? 1.f : -1.f) * std::expm1(std::abs(x));
    }
};
