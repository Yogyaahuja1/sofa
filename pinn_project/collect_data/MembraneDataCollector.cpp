#include "MembraneDataCollector.h"
#include <sofa/core/ObjectFactory.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <sofa/simulation/Node.h>
#include <sstream>
#include <charconv>
#include <cstdlib>

namespace pinn
{

static std::string sofa_root_path(const char* rel) {
    const char* env = std::getenv("SOFA_ROOT");
    return std::string(env ? env : ".") + "/" + rel;
}

MembraneDataCollector::MembraneDataCollector()
    : d_outputFile(initData(&d_outputFile,
                            sofa_root_path("pinn_project/data/membrane_training_data.csv"),
                            "outputFile", "Path to membrane output CSV file"))
    , d_collectEvery(initData(&d_collectEvery, 3,
                              "collectEvery", "Collect data every N sim steps"))
{
    this->f_listening.setValue(true);
}

MembraneDataCollector::~MembraneDataCollector()
{
    {
        std::lock_guard<std::mutex> lk(m_queueMutex);
        m_stopWriter.store(true);
    }
    m_queueCV.notify_all();
    if (m_writerThread.joinable())
        m_writerThread.join();
    if (m_file.is_open()) { m_file.flush(); m_file.close(); }
}

void MembraneDataCollector::init()
{
    m_membraneDofs = dynamic_cast<sofa::core::behavior::MechanicalState<sofa::defaulttype::Vec3Types>*>(
        this->getContext()->getMechanicalState());
    if (!m_membraneDofs) { msg_error() << "MembraneDataCollector: No MechanicalObject (Vec3) found."; return; }

    const auto& restPos = m_membraneDofs->read(sofa::core::ConstVecCoordId::restPosition())->getValue();
    m_nVertices = (int)restPos.size();
    m_prevDeform.assign(m_nVertices, sofa::type::Vec3d(0.0, 0.0, 0.0));
    m_contactProxy.assign(m_nVertices, sofa::type::Vec3d(0.0, 0.0, 0.0));
    m_accStress.assign(m_nVertices, sofa::type::Vec3d(0.0, 0.0, 0.0));
    m_vertStress.assign(m_nVertices, sofa::type::Vec3d(0.0, 0.0, 0.0));
    m_vertStrain.assign(m_nVertices, sofa::type::Vec3d(0.0, 0.0, 0.0));
    m_vertCount.assign(m_nVertices, 0);

    this->getContext()->get(m_triangleFemFF);
    this->getContext()->get(m_topo);
    if (!m_triangleFemFF)
        msg_warning() << "MembraneDataCollector: TriangularFEMForceField not found — stress/strain will be zero.";
    if (!m_topo)
        msg_error() << "MembraneDataCollector: BaseMeshTopology not found.";

    msg_info() << "MembraneDataCollector initialized. Vertices: " << m_nVertices
               << "  collectEvery=" << d_collectEvery.getValue();

    // 6 groups: contactProxy | accStress | prevDeform | deform | vertStress | vertStrain
    m_frame.vals.resize(6 * m_nVertices * 3, 0.0);

    // Probe existing CSV to find max session_id — avoids ID collisions when scene is reloaded
    {
        std::ifstream reader(d_outputFile.getValue(), std::ios::binary | std::ios::ate);
        if (reader.is_open()) {
            auto fileSize = (std::streamoff)reader.tellg();
            // Rows are ~150KB each; read 300KB from end to capture at least one full row start
            std::streamoff readFrom = std::max((std::streamoff)0, fileSize - 300000);
            reader.seekg(readFrom);
            std::string chunk((size_t)(fileSize - readFrom), '\0');
            reader.read(&chunk[0], (std::streamsize)chunk.size());
            // Strip trailing newlines to find end of last data row
            size_t end = chunk.size();
            while (end > 0 && (chunk[end-1] == '\n' || chunk[end-1] == '\r')) end--;
            size_t nl = (end > 0) ? chunk.rfind('\n', end - 1) : std::string::npos;
            size_t rowStart = (nl == std::string::npos) ? 0 : nl + 1;
            // Read first 200 chars of last row — enough to reach session_id (field 4)
            std::string rowHead = chunk.substr(rowStart, std::min((size_t)200, end - rowStart));
            std::istringstream ss(rowHead);
            std::string tok;
            for (int f = 0; f <= 4 && std::getline(ss, tok, ','); f++)
                if (f == 4) break;
            if (!tok.empty() && tok != "session_id") {
                try {
                    m_sessionId = std::stoi(tok) + 1;
                    msg_info() << "MembraneDataCollector: resuming, next session_id=" << m_sessionId;
                } catch (...) {}
            }
        }
    }

    m_file.open(d_outputFile.getValue(), std::ios::app);
    if (!m_file.is_open()) { msg_error() << "Cannot open: " << d_outputFile.getValue(); return; }
    m_file.rdbuf()->pubsetbuf(nullptr, 1 << 20);
    m_file.seekp(0, std::ios::end);
    m_headerWritten = (m_file.tellp() != 0);

    m_startRealTime = std::chrono::high_resolution_clock::now();
    m_stopWriter.store(false);
    m_writerThread = std::thread(&MembraneDataCollector::writerLoop, this);
}

void MembraneDataCollector::bwdInit()
{
    this->getContext()->get(m_lcpFF, sofa::core::objectmodel::BaseContext::SearchRoot);
    if (!m_lcpFF)
        msg_warning() << "MembraneDataCollector: LCPForceFeedback not found — tool force will be zero.";
}

void MembraneDataCollector::writerLoop()
{
    std::string row;
    row.reserve(300000);
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
            MembraneFrameData frame = std::move(m_frameQueue.front());
            m_frameQueue.pop();
            lk.unlock();

            row.clear();
            const int nv = (int)(frame.vals.size() / (6 * 3));

            row += std::to_string(frame.step);      row += ',';
            append_d(frame.real_time);               row += ',';
            append_d(frame.sim_time);                row += ',';
            append_d(frame.dt);                      row += ',';
            row += std::to_string(frame.session_id); row += ',';
            append_d(frame.toolPos[0]);   row+=','; append_d(frame.toolPos[1]);   row+=','; append_d(frame.toolPos[2]);
            row+=','; append_d(frame.toolVel[0]);   row+=','; append_d(frame.toolVel[1]);   row+=','; append_d(frame.toolVel[2]);
            row+=','; append_d(frame.toolForce[0]); row+=','; append_d(frame.toolForce[1]); row+=','; append_d(frame.toolForce[2]);

            // groups: 0=contactProxy 1=accStress 2=prevDeform 3=deform 4=vertStress 5=vertStrain
            for (int g = 0; g < 6; g++) {
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

void MembraneDataCollector::writeHeader()
{
    std::ostringstream h;
    h << "step,real_time,sim_time,dt_since_last,session_id";
    h << ",tool_x,tool_y,tool_z";
    h << ",tool_vx,tool_vy,tool_vz";
    h << ",tool_fx,tool_fy,tool_fz";
    for (int i = 0; i < m_nVertices; i++) h << ",fvx" << i << ",fvy" << i << ",fvz" << i;
    for (int i = 0; i < m_nVertices; i++) h << ",sax" << i << ",say" << i << ",saz" << i;
    for (int i = 0; i < m_nVertices; i++) h << ",pdx" << i << ",pdy" << i << ",pdz" << i;
    for (int i = 0; i < m_nVertices; i++) h << ",dx"  << i << ",dy"  << i << ",dz"  << i;
    for (int i = 0; i < m_nVertices; i++) h << ",msxx" << i << ",msyy" << i << ",msxy" << i;
    for (int i = 0; i < m_nVertices; i++) h << ",mexx" << i << ",meyy" << i << ",mexy" << i;
    h << "\n";
    m_file << h.str();
    m_file.flush();
}

void MembraneDataCollector::readFEMCache()
{
    if (!m_triangleFemFF || !m_topo) return;

    std::fill(m_vertStress.begin(), m_vertStress.end(), sofa::type::Vec3d(0,0,0));
    std::fill(m_vertStrain.begin(), m_vertStrain.end(), sofa::type::Vec3d(0,0,0));
    std::fill(m_vertCount.begin(),  m_vertCount.end(),  0);

    const auto& triangles  = m_topo->getTriangles();
    const auto& triInfoVec = m_triangleFemFF->triangleInfo.getValue();
    const int nTris = (int)triangles.size();

    for (int t = 0; t < nTris && t < (int)triInfoVec.size(); ++t)
    {
        const auto& tri    = triangles[t];
        const auto& stress = triInfoVec[t].stress;
        const auto& strain = triInfoVec[t].strain;
        for (int i = 0; i < 3; ++i)
        {
            const auto v = (int)tri[i];
            if (v < m_nVertices)
            {
                m_vertStress[v][0] += stress[0]; m_vertStress[v][1] += stress[1]; m_vertStress[v][2] += stress[2];
                m_vertStrain[v][0] += strain[0]; m_vertStrain[v][1] += strain[1]; m_vertStrain[v][2] += strain[2];
                m_vertCount[v]++;
            }
        }
    }
    for (int v = 0; v < m_nVertices; ++v)
        if (m_vertCount[v] > 0)
        { m_vertStress[v] /= m_vertCount[v]; m_vertStrain[v] /= m_vertCount[v]; }
}

void MembraneDataCollector::handleEvent(sofa::core::objectmodel::Event* e)
{
    if (!sofa::simulation::AnimateEndEvent::checkEventType(e)) return;
    m_step++;
    if (m_step % d_collectEvery.getValue() != 0) return;
    writeSample();
}

void MembraneDataCollector::writeSample()
{
    if (!m_membraneDofs) return;

    auto now = std::chrono::high_resolution_clock::now();
    double real_time = std::chrono::duration<double>(now - m_startRealTime).count();
    double sim_time  = this->getContext()->getTime();
    double dt        = sim_time - m_lastSimTime;

    if (!m_headerWritten) { writeHeader(); m_headerWritten = true; }

    // Find instrument DOFs once (same pattern as DataCollector)
    if (!m_instrDofs)
    {
        auto* root = dynamic_cast<sofa::simulation::Node*>(this->getContext()->getRootContext());
        if (root)
        {
            auto* instrNode = root->getChild("Instrument");
            if (instrNode)
                m_instrDofs = dynamic_cast<sofa::core::behavior::MechanicalState<sofa::defaulttype::Rigid3Types>*>(
                    instrNode->getObject("instrumentState"));
        }
    }

    sofa::type::Vec3d toolPos(0,0,0), toolVel(0,0,0), toolForce(0,0,0);
    if (m_instrDofs)
    {
        const auto& pos = m_instrDofs->read(sofa::core::ConstVecCoordId::position())->getValue();
        if (!pos.empty()) toolPos = { pos[0].getCenter()[0], pos[0].getCenter()[1], pos[0].getCenter()[2] };
        const auto& vel = m_instrDofs->read(sofa::core::ConstVecDerivId::velocity())->getValue();
        if (!vel.empty()) toolVel = { vel[0].getLinear()[0], vel[0].getLinear()[1], vel[0].getLinear()[2] };
        if (m_lcpFF)
        {
            toolForce = m_lcpFF->getRealForce();
            auto hapticData = m_lcpFF->getForce();
            if (hapticData.sim_time > 0)
            {
                sim_time  = hapticData.sim_time;
                real_time = std::chrono::duration<double>(hapticData.real_time - m_startRealTime).count();
            }
        }
    }

    const auto& curPos  = m_membraneDofs->read(sofa::core::ConstVecCoordId::position())->getValue();
    const auto& freePos = m_membraneDofs->read(sofa::core::ConstVecCoordId::freePosition())->getValue();
    const auto& restPos = m_membraneDofs->read(sofa::core::ConstVecCoordId::restPosition())->getValue();

    // contactProxy = freePos - curPos (penetration depth vector, like liver)
    for (int i = 0; i < m_nVertices; ++i)
        m_contactProxy[i] = (i < (int)freePos.size() && i < (int)curPos.size())
            ? sofa::type::Vec3d(freePos[i][0]-curPos[i][0], freePos[i][1]-curPos[i][1], freePos[i][2]-curPos[i][2])
            : sofa::type::Vec3d(0,0,0);

    // Contact detection via proxy norm (same threshold as liver)
    double maxProxy = 0.0;
    for (int i = 0; i < m_nVertices; ++i) maxProxy = std::max(maxProxy, m_contactProxy[i].norm());
    if (maxProxy < 0.005) { m_isRecording = false; }
    else if (!m_isRecording) { m_isRecording = true; m_sessionId++; }

    // accStress EMA (alpha=0.9, same as liver)
    if (!m_accStressInit) { m_accStress.assign(m_nVertices, sofa::type::Vec3d(0,0,0)); m_accStressInit = true; }
    for (int i = 0; i < m_nVertices; ++i)
    {
        m_accStress[i][0] = m_alpha*m_accStress[i][0] + (1-m_alpha)*m_contactProxy[i][0];
        m_accStress[i][1] = m_alpha*m_accStress[i][1] + (1-m_alpha)*m_contactProxy[i][1];
        m_accStress[i][2] = m_alpha*m_accStress[i][2] + (1-m_alpha)*m_contactProxy[i][2];
    }

    std::fill(m_vertStress.begin(), m_vertStress.end(), sofa::type::Vec3d(0,0,0));
    std::fill(m_vertStrain.begin(), m_vertStrain.end(), sofa::type::Vec3d(0,0,0));
    std::fill(m_vertCount.begin(),  m_vertCount.end(),  0);
    readFEMCache();

    // Pointer-walk fill: 6 groups × nVertices × 3
    // group 0: contactProxy | 1: accStress | 2: prevDeform | 3: deform | 4: stress | 5: strain
    {
        double* p = m_frame.vals.data();
        for (int i = 0; i < m_nVertices; ++i) { *p++=m_contactProxy[i][0]; *p++=m_contactProxy[i][1]; *p++=m_contactProxy[i][2]; }
        for (int i = 0; i < m_nVertices; ++i) { *p++=m_accStress[i][0];    *p++=m_accStress[i][1];    *p++=m_accStress[i][2]; }
        for (int i = 0; i < m_nVertices; ++i) { *p++=m_prevDeform[i][0];   *p++=m_prevDeform[i][1];   *p++=m_prevDeform[i][2]; }
        for (int i = 0; i < m_nVertices; ++i)
        {
            const double dx = (i < (int)curPos.size() && i < (int)restPos.size()) ? curPos[i][0]-restPos[i][0] : 0.0;
            const double dy = (i < (int)curPos.size() && i < (int)restPos.size()) ? curPos[i][1]-restPos[i][1] : 0.0;
            const double dz = (i < (int)curPos.size() && i < (int)restPos.size()) ? curPos[i][2]-restPos[i][2] : 0.0;
            *p++=dx; *p++=dy; *p++=dz;
            m_prevDeform[i] = { dx, dy, dz };
        }
        for (int i = 0; i < m_nVertices; ++i) { *p++=m_vertStress[i][0]; *p++=m_vertStress[i][1]; *p++=m_vertStress[i][2]; }
        for (int i = 0; i < m_nVertices; ++i) { *p++=m_vertStrain[i][0]; *p++=m_vertStrain[i][1]; *p++=m_vertStrain[i][2]; }
    }

    m_lastSimTime = sim_time;
    m_frame.step       = m_step;
    m_frame.real_time  = real_time;
    m_frame.sim_time   = sim_time;
    m_frame.dt         = dt;
    m_frame.session_id = m_isRecording ? m_sessionId : 0;
    m_frame.toolPos[0] = toolPos[0];   m_frame.toolPos[1] = toolPos[1];   m_frame.toolPos[2] = toolPos[2];
    m_frame.toolVel[0] = toolVel[0];   m_frame.toolVel[1] = toolVel[1];   m_frame.toolVel[2] = toolVel[2];
    m_frame.toolForce[0] = toolForce[0]; m_frame.toolForce[1] = toolForce[1]; m_frame.toolForce[2] = toolForce[2];

    std::lock_guard<std::mutex> lk(m_queueMutex);
    m_frameQueue.push(m_frame);
    m_queueCV.notify_one();
}

int MembraneDataCollectorClass = sofa::core::RegisterObject("PINN data collector for triangular membrane FEM")
    .add<MembraneDataCollector>();

} // namespace pinn
