#include "DataCollector.h"
#include <sofa/core/ObjectFactory.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <sofa/simulation/Node.h>
#include <sofa/helper/system/FileSystem.h>
#include <sstream>
#include <cstdio>
#include <cstdlib>

namespace pinn
{

static std::string sofa_root_path(const char* rel) {
    const char* env = std::getenv("SOFA_ROOT");
    return std::string(env ? env : ".") + "/" + rel;
}

DataCollector::DataCollector()
    : d_outputFile(initData(&d_outputFile,
                            sofa_root_path("pinn_project/data/training_data.csv"),
                            "outputFile", "Path to output CSV file"))
    , d_collectEvery(initData(&d_collectEvery, 5,
                              "collectEvery", "Collect data every N sim steps"))
    , d_toolPath(initData(&d_toolPath, std::string(""),
                          "toolPath", "SOFA path to tool MechanicalObject"))
    , d_dumpAxbEvery(initData(&d_dumpAxbEvery, 0,
                              "dumpAxbEvery", "Dump solver RHS/solution every N steps (0=disabled)"))
    , d_axbDir(initData(&d_axbDir, sofa_root_path("pinn_project/data/ax_b"),
                        "axbDir", "Output directory for Ax=b dumps"))
{
    this->f_listening.setValue(true);
}

DataCollector::~DataCollector()
{
    {
        std::lock_guard<std::mutex> lk(m_queueMutex);
        m_stopWriter.store(true);
    }
    m_queueCV.notify_all();
    if (m_writerThread.joinable())
        m_writerThread.join();

    if (m_file.is_open())
    {
        m_file.flush();
        m_file.close();
        msg_info() << "Data collection complete. File closed.";
    }
}

void DataCollector::init()
{
    m_liverDofs = dynamic_cast<sofa::core::behavior::MechanicalState<sofa::defaulttype::Vec3Types>*>(
        this->getContext()->getMechanicalState());
    if (!m_liverDofs) { msg_error() << "DataCollector: No MechanicalObject (Vec3) found."; return; }

    const auto& restPos = m_liverDofs->read(sofa::core::ConstVecCoordId::restPosition())->getValue();
    m_nVertices = (int)restPos.size();
    m_prevDeform.assign(m_nVertices, sofa::type::Vec3d(0.0, 0.0, 0.0));

    this->getContext()->get(m_femFF);
    this->getContext()->get(m_topo);
    if (!m_femFF) msg_error() << "DataCollector: TetrahedronFEMForceField not found.";
    if (!m_topo)  msg_error() << "DataCollector: BaseMeshTopology not found.";

    msg_info() << "DataCollector initialized. Liver vertices: " << m_nVertices
               << "  collectEvery=" << d_collectEvery.getValue();

    m_file.open(d_outputFile.getValue(), std::ios::app);
    if (!m_file.is_open()) { msg_error() << "Cannot open file: " << d_outputFile.getValue(); return; }
    m_file.rdbuf()->pubsetbuf(nullptr, 1 << 20); // 1 MB write buffer

    m_file.seekp(0, std::ios::end);
    m_headerWritten = (m_file.tellp() != 0);
    msg_info() << "Writing to: " << d_outputFile.getValue();

    if (m_headerWritten)
    {
        std::ifstream checkFile(d_outputFile.getValue());
        std::string firstLine;
        std::getline(checkFile, firstLine);
        if (firstLine.find("dt_since_last") == std::string::npos ||
            firstLine.find("real_time") == std::string::npos)
        {
            msg_error() << "DataCollector: OLD header detected — delete file and rerun.";
        }
    }

    m_vertStress.assign(m_nVertices, sofa::type::Vec3d(0,0,0));
    m_vertStrain.assign(m_nVertices, sofa::type::Vec3d(0,0,0));
    m_contactProxy.assign(m_nVertices, sofa::type::Vec3d(0,0,0));
    m_vertCount.assign(m_nVertices, 0);

    // Pre-allocate the frame's flat value buffer (7 groups × nVerts × 3 doubles).
    // SOFA thread fills this in-place; copy to queue is 7×181×3×8 = ~30KB (< 128KB
    // mmap threshold → uses arena, no system call).
    m_frame.vals.resize(7 * m_nVertices * 3, 0.0);

    m_startRealTime = std::chrono::high_resolution_clock::now();

    if (d_dumpAxbEvery.getValue() > 0)
        sofa::helper::system::FileSystem::findOrCreateAValidPath(d_axbDir.getValue());

    auto* liverNode = dynamic_cast<sofa::simulation::Node*>(this->getContext());
    if (liverNode)
    {
        auto* surfNode = liverNode->getChild("Surf");
        if (surfNode)
            m_contactDofs = dynamic_cast<sofa::core::behavior::MechanicalState<sofa::defaulttype::Vec3Types>*>(
                surfNode->getObject("surfDofs"));
    }
    m_nSurfaceVerts = m_contactDofs ? (int)m_contactDofs->getSize() : 0;

    m_stopWriter.store(false);
    m_writerThread = std::thread(&DataCollector::writerLoop, this);
}

void DataCollector::bwdInit()
{
    this->getContext()->get(m_lcpFF, sofa::core::objectmodel::BaseContext::SearchRoot);
    if (!m_lcpFF)
        msg_warning() << "DataCollector: LCPForceFeedback NOT FOUND — tool force/position will be zero.";

    m_solver = this->getContext()->get<sofa::core::behavior::LinearSolver>(
        sofa::core::objectmodel::BaseContext::SearchRoot);
}

// ── Background writer thread ──────────────────────────────────────────────────
// Pops raw FrameData structs (~30KB each) and converts to CSV text here,
// keeping all string allocation and disk I/O off the SOFA thread.
void DataCollector::writerLoop()
{
    // Pre-allocate the row string ONCE here — reused with clear() each frame.
    // 250KB stays allocated on this thread; SOFA thread never touches it.
    std::string row;
    row.reserve(250000);
    char buf[32];

    auto append_d = [&](double v) {
        auto [ptr, ec] = std::to_chars(buf, buf+sizeof(buf), v, std::chars_format::general, 7);
        row.append(buf, ptr - buf);
    };

    while (true)
    {
        std::unique_lock<std::mutex> lk(m_queueMutex);
        m_queueCV.wait(lk, [this]{ return !m_frameQueue.empty() || m_stopWriter.load(); });

        while (!m_frameQueue.empty())
        {
            FrameData frame = std::move(m_frameQueue.front());
            m_frameQueue.pop();
            lk.unlock();

            // Build CSV row from raw frame data
            row.clear();
            const int nv = (int)(frame.vals.size() / (7 * 3));

            row += std::to_string(frame.step);   row += ',';
            append_d(frame.real_time);            row += ',';
            append_d(frame.sim_time);             row += ',';
            append_d(frame.dt);                   row += ',';
            row += std::to_string(frame.session_id); row += ',';
            append_d(frame.toolPos[0]);   row+=','; append_d(frame.toolPos[1]);   row+=','; append_d(frame.toolPos[2]);
            row+=','; append_d(frame.toolVel[0]);   row+=','; append_d(frame.toolVel[1]);   row+=','; append_d(frame.toolVel[2]);
            row+=','; append_d(frame.toolForce[0]); row+=','; append_d(frame.toolForce[1]); row+=','; append_d(frame.toolForce[2]);

            // groups: 0=contactProxy 1=accStress 2=vels 3=prevDeform 4=deform 5=vertStress 6=vertStrain
            for (int g = 0; g < 7; g++) {
                const int base = g * nv * 3;
                for (int i = 0; i < nv; i++) {
                    row+=','; append_d(frame.vals[base + i*3 + 0]);
                    row+=','; append_d(frame.vals[base + i*3 + 1]);
                    row+=','; append_d(frame.vals[base + i*3 + 2]);
                }
            }
            row += '\n';

            m_file.write(row.data(), (std::streamsize)row.size());

            lk.lock();
        }

        if (m_stopWriter.load() && m_frameQueue.empty()) break;
    }
    m_file.flush();
}

void DataCollector::writeHeader()
{
    std::ostringstream h;
    h << "step,real_time,sim_time,dt_since_last,session_id";
    h << ",tool_x,tool_y,tool_z";
    h << ",tool_vx,tool_vy,tool_vz";
    h << ",tool_fx,tool_fy,tool_fz";
    for (int i = 0; i < m_nVertices; i++) h << ",fvx" << i << ",fvy" << i << ",fvz" << i;
    for (int i = 0; i < m_nVertices; i++) h << ",sax" << i << ",say" << i << ",saz" << i;
    for (int i = 0; i < m_nVertices; i++) h << ",vvx" << i << ",vvy" << i << ",vvz" << i;
    for (int i = 0; i < m_nVertices; i++) h << ",pdx" << i << ",pdy" << i << ",pdz" << i;
    for (int i = 0; i < m_nVertices; i++) h << ",dx"  << i << ",dy"  << i << ",dz"  << i;
    for (int i = 0; i < m_nVertices; i++) h << ",rsxx" << i << ",rsyy" << i << ",rszz" << i;
    for (int i = 0; i < m_nVertices; i++) h << ",rexx" << i << ",reyy" << i << ",rezz" << i;
    h << "\n";
    m_file << h.str();
    m_file.flush();
}

void DataCollector::handleEvent(sofa::core::objectmodel::Event* e)
{
    if (!sofa::simulation::AnimateEndEvent::checkEventType(e)) return;

    m_step++;
    if (m_step % d_collectEvery.getValue() != 0) return;

    writeSample();
}

// ── Read per-tet stress/strain cached by FEM during its addForce() ────────────
// FEM already computed JtD (strain) and K*JtD (stress) for every tet this step.
// We just average them over the 4 vertices of each tet — no matrix math needed.
void DataCollector::readFEMCache()
{
    if (!m_femFF || !m_topo) return;

    const auto& tetrahedra = m_topo->getTetrahedra();
    const int nTets = (int)m_femFF->getNumTetra();
    if (nTets == 0) return;

    for (int t = 0; t < nTets; ++t)
    {
        const auto& tet    = tetrahedra[t];
        const auto& strain = m_femFF->getLastStrain((unsigned int)t);
        const auto& stress = m_femFF->getLastStress((unsigned int)t);

        for (int i = 0; i < 4; ++i)
        {
            const auto v = tet[i];
            m_vertStress[v][0] += stress[0]; m_vertStress[v][1] += stress[1]; m_vertStress[v][2] += stress[2];
            m_vertStrain[v][0] += strain[0]; m_vertStrain[v][1] += strain[1]; m_vertStrain[v][2] += strain[2];
            m_vertCount[v]++;
        }
    }

    for (int v = 0; v < m_nVertices; ++v)
    {
        if (m_vertCount[v] > 0)
        {
            m_vertStress[v] /= m_vertCount[v];
            m_vertStrain[v] /= m_vertCount[v];
        }
    }
}

void DataCollector::writeSample()
{
    if (!m_liverDofs || !m_file.is_open()) return;

    if (!m_headerWritten)
    {
        m_nSurfaceVerts = m_contactDofs ? (int)m_contactDofs->getSize() : m_nVertices;
        writeHeader();
        m_headerWritten = true;
    }

    // Find instrument DOFs once
    if (!m_instrDofs)
    {
        auto* root = dynamic_cast<sofa::simulation::Node*>(this->getContext()->getRootContext());
        if (root)
        {
            if (!d_toolPath.getValue().empty())
            {
                auto* obj = root->getObject(d_toolPath.getValue());
                m_instrDofs = dynamic_cast<sofa::core::behavior::MechanicalState<sofa::defaulttype::Rigid3Types>*>(obj);
            }
            if (!m_instrDofs)
            {
                auto* instrNode = root->getChild("Instrument");
                if (instrNode)
                {
                    m_instrDofs = dynamic_cast<sofa::core::behavior::MechanicalState<sofa::defaulttype::Rigid3Types>*>(
                        instrNode->getObject("instrumentState"));
                }
            }
            if (m_instrDofs)
            {
                msg_info() << "DataCollector: Instrument found.";
            }
            else
            {
                msg_warning() << "DataCollector: Instrument MechanicalState not found.";
            }
        }
    }

    // ── Collect scalars ───────────────────────────────────────────────────────
    sofa::type::Vec3d toolPos(0,0,0), toolVel(0,0,0), toolForce(0,0,0);
    double sim_time  = this->getContext()->getTime();
    double real_time = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - m_startRealTime).count();

    if (m_instrDofs)
    {
        const auto& pos = m_instrDofs->read(sofa::core::ConstVecCoordId::position())->getValue();
        if (!pos.empty()) toolPos = { pos[0].getCenter()[0], pos[0].getCenter()[1], pos[0].getCenter()[2] };

        const auto& vel = m_instrDofs->read(sofa::core::ConstVecDerivId::velocity())->getValue();
        if (!vel.empty()) toolVel = { vel[0].getLinear()[0], vel[0].getLinear()[1], vel[0].getLinear()[2] };

        if (m_lcpFF)
        {
            // getRealForce() reads m_realForceCache — set by the haptic thread's
            // doComputeForce() every cycle, always from real LCP (never PINN output).
            toolForce = m_lcpFF->getRealForce();

            // Use haptic timestamps only when they look valid
            auto hapticData = m_lcpFF->getForce();
            if (hapticData.sim_time > 0)
            {
                sim_time  = hapticData.sim_time;
                real_time = std::chrono::duration<double>(hapticData.real_time - m_startRealTime).count();
            }
        }
    }

    const double dt_since_last = sim_time - m_lastSimTime;
    m_lastSimTime = sim_time;

    const auto& curPos     = m_liverDofs->read(sofa::core::ConstVecCoordId::position())->getValue();
    const auto& freePos    = m_liverDofs->read(sofa::core::ConstVecCoordId::freePosition())->getValue();
    const auto& restPos    = m_liverDofs->read(sofa::core::ConstVecCoordId::restPosition())->getValue();
    const auto& vertexVels = m_liverDofs->read(sofa::core::ConstVecDerivId::velocity())->getValue();

    // Contact proxy must be computed before session detection
    for (int i = 0; i < m_nVertices && i < (int)freePos.size(); i++)
    {
        const sofa::type::Vec3d proxy = freePos[i] - curPos[i];
        m_contactProxy[i] = proxy;
    }

    // Use contactProxy magnitude for contact detection — toolForce from haptic
    // thread is unreliable (LCP constraint problem may not be populated yet).
    double maxProxy = 0.0;
    for (int i = 0; i < m_nVertices; i++)
        maxProxy = std::max(maxProxy, m_contactProxy[i].norm());

    if (maxProxy < 0.005) { m_isRecording = false; }
    else if (!m_isRecording) { m_isRecording = true; m_sessionId++; }

    // Zero other buffers (contactProxy already filled above)
    const sofa::type::Vec3d zero3(0,0,0);
    std::fill(m_vertStress.begin(),   m_vertStress.end(),   zero3);
    std::fill(m_vertStrain.begin(),   m_vertStrain.end(),   zero3);
    std::fill(m_vertCount.begin(),    m_vertCount.end(),    0);

    if (!m_accStressInit) { m_accStress.assign(m_nVertices, zero3); m_accStressInit = true; }
    for (int i = 0; i < m_nVertices; i++)
    {
        m_accStress[i][0] = m_alpha*m_accStress[i][0] + (1-m_alpha)*m_contactProxy[i][0];
        m_accStress[i][1] = m_alpha*m_accStress[i][1] + (1-m_alpha)*m_contactProxy[i][1];
        m_accStress[i][2] = m_alpha*m_accStress[i][2] + (1-m_alpha)*m_contactProxy[i][2];
    }

    readFEMCache();

    // ── Pack raw data into pre-allocated frame struct (~30KB, no string work) ────
    m_frame.step       = m_step;
    m_frame.real_time  = real_time;
    m_frame.sim_time   = sim_time;
    m_frame.dt         = dt_since_last;
    m_frame.session_id = m_isRecording ? m_sessionId : 0;
    m_frame.toolPos[0] = toolPos[0];  m_frame.toolPos[1] = toolPos[1];  m_frame.toolPos[2] = toolPos[2];
    m_frame.toolVel[0] = toolVel[0];  m_frame.toolVel[1] = toolVel[1];  m_frame.toolVel[2] = toolVel[2];
    m_frame.toolForce[0] = toolForce[0]; m_frame.toolForce[1] = toolForce[1]; m_frame.toolForce[2] = toolForce[2];

    // Fill flat vals: groups 0-6, each nVertices × 3
    {
        double* p = m_frame.vals.data();
        // group 0: contactProxy
        for (int i = 0; i < m_nVertices; i++) { *p++=m_contactProxy[i][0]; *p++=m_contactProxy[i][1]; *p++=m_contactProxy[i][2]; }
        // group 1: accStress (EMA)
        for (int i = 0; i < m_nVertices; i++) { *p++=m_accStress[i][0];    *p++=m_accStress[i][1];    *p++=m_accStress[i][2]; }
        // group 2: velocities
        for (int i = 0; i < m_nVertices; i++) { *p++=vertexVels[i][0];     *p++=vertexVels[i][1];     *p++=vertexVels[i][2]; }
        // group 3: prevDeform (displacement from previous step)
        for (int i = 0; i < m_nVertices; i++) { *p++=m_prevDeform[i][0];   *p++=m_prevDeform[i][1];   *p++=m_prevDeform[i][2]; }
        // group 4: deform (current displacement = curPos - restPos)
        for (int i = 0; i < m_nVertices; i++) { *p++=curPos[i][0]-restPos[i][0]; *p++=curPos[i][1]-restPos[i][1]; *p++=curPos[i][2]-restPos[i][2]; }
        // group 5: vertStress (from FEM cache)
        for (int i = 0; i < m_nVertices; i++) { *p++=m_vertStress[i][0];   *p++=m_vertStress[i][1];   *p++=m_vertStress[i][2]; }
        // group 6: vertStrain (from FEM cache)
        for (int i = 0; i < m_nVertices; i++) { *p++=m_vertStrain[i][0];   *p++=m_vertStrain[i][1];   *p++=m_vertStrain[i][2]; }
    }

    // ── Push copy of raw frame to writer thread (~30KB arena alloc, no mmap) ──
    {
        std::lock_guard<std::mutex> lk(m_queueMutex);
        m_frameQueue.push(m_frame);
    }
    m_queueCV.notify_one();

    // Update previous deformation buffer
    for (int i = 0; i < m_nVertices; i++)
        m_prevDeform[i] = curPos[i] - restPos[i];

    if (m_solver && d_dumpAxbEvery.getValue() > 0 && m_step % d_dumpAxbEvery.getValue() == 0)
    {
        const int fi = m_step / d_dumpAxbEvery.getValue();
        dumpVector(m_solver->getSystemRHBaseVector(), fi, "b");
        dumpVector(m_solver->getSystemLHBaseVector(), fi, "x");
    }

    m_prevToolPos = toolPos;
    m_firstStep   = false;
}

void DataCollector::dumpVector(sofa::linearalgebra::BaseVector* v, int frameIdx, const char* prefix)
{
    if (!v) return;
    char suffix[32];
    std::snprintf(suffix, sizeof(suffix), "/%s_%05d.txt", prefix, frameIdx);
    std::ofstream f(d_axbDir.getValue() + suffix);
    for (sofa::linearalgebra::BaseVector::Index i = 0; i < v->size(); ++i)
        f << v->element(i) << "\n";
}

} // namespace pinn

extern "C" {
    void initExternalModule() {}
    const char* getModuleName()    { return "PINNDataCollector"; }
    const char* getModuleVersion() { return "1.0"; }
}
