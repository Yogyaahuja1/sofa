#include "DataCollector.h"
#include <sofa/core/ObjectFactory.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <sofa/simulation/Node.h>
#include <sofa/helper/system/FileSystem.h>
#include <cstdio>

namespace pinn
{

DataCollector::DataCollector()
    : d_outputFile(initData(&d_outputFile,
                            std::string("/home/yogyaahuja/sofa/pinn_project/data/training_data.csv"),
                            "outputFile", "Path to output CSV file"))
    , d_collectEvery(initData(&d_collectEvery, 5,
                              "collectEvery", "Collect data every N steps"))
    , d_toolPath(initData(&d_toolPath, std::string(""),
                          "toolPath", "SOFA path to tool MechanicalObject"))
    , d_dumpAxbEvery(initData(&d_dumpAxbEvery, 0,
                              "dumpAxbEvery", "Dump solver RHS (b) and solution (x) vectors every N steps (0=disabled)"))
    , d_axbDir(initData(&d_axbDir, std::string("/home/yogyaahuja/sofa/pinn_project/data/ax_b"),
                        "axbDir", "Output directory for b_NNNNN.txt / x_NNNNN.txt dumps"))
{
    msg_error() << "===== NEW PINN PLUGIN LOADED =====";
    this->f_listening.setValue(true);
}

DataCollector::~DataCollector()
{
    if (m_file.is_open())
    {
        m_file.close();
        msg_info() << "Data collection complete. File closed.";
    }
}

void DataCollector::init()
{
    m_liverDofs = dynamic_cast<sofa::core::behavior::MechanicalState<sofa::defaulttype::Vec3Types>*>(
        this->getContext()->getMechanicalState());

    if (!m_liverDofs)
    {
        msg_error() << "DataCollector: No MechanicalObject (Vec3) found.";
        return;
    }
    // In init(), after finding m_liverDofs, add:
    const auto& restPos = m_liverDofs->read(
        sofa::core::ConstVecCoordId::restPosition())->getValue();
    m_nVertices = (int)restPos.size();

    // ── NEW: initialise previous deformation buffer to zeros ──────
    m_prevDeform.assign(m_nVertices, sofa::type::Vec3d(0.0, 0.0, 0.0));
    // ─────────────────────────────────────────────────────────────

    // ── NEW: locate the FEM forcefield + topology for real stress/strain ──
    this->getContext()->get(m_femFF);
    this->getContext()->get(m_topo);
    if (!m_femFF)
        msg_error() << "DataCollector: TetrahedronFEMForceField not found (real stress/strain disabled).";
    if (!m_topo)
        msg_error() << "DataCollector: BaseMeshTopology not found (real stress/strain disabled).";
    // ─────────────────────────────────────────────────────────────

    msg_info() << "DataCollector initialized."
               << " Liver vertices: " << m_nVertices
               << " Collecting every " << d_collectEvery.getValue() << " steps.";

    m_file.open(d_outputFile.getValue(), std::ios::app);
    if (!m_file.is_open())
    {
        msg_error() << "Cannot open file: " << d_outputFile.getValue();
        return;
    }

    m_file.seekp(0, std::ios::end);
    m_headerWritten = (m_file.tellp() != 0);
    msg_info() << "Writing to: " << d_outputFile.getValue();

    if (m_headerWritten)
    {
        // File is non-empty: we are appending to it, so its existing header
        // is kept as-is. If it predates the dt_since_last/real_time columns,
        // new rows will have extra fields vs the header — warn loudly so
        // stale CSVs don't silently get fed to training.
        std::ifstream checkFile(d_outputFile.getValue());
        std::string firstLine;
        std::getline(checkFile, firstLine);
        if (firstLine.find("dt_since_last") == std::string::npos ||
            firstLine.find("real_time") == std::string::npos)
        {
            msg_error() << "DataCollector: " << d_outputFile.getValue()
                         << " has an OLD header (missing 'dt_since_last' and/or 'real_time'). "
                         << "Delete this file and rerun to collect data in the new format, "
                         << "otherwise rows will be misaligned with the header.";
        }
    }

    // Reference point for the real_time column: elapsed wall-clock seconds
    // since recording started (keeps the value small/precise instead of a
    // raw epoch timestamp).
    m_startRealTime = std::chrono::high_resolution_clock::now();

    if (d_dumpAxbEvery.getValue() > 0)
        sofa::helper::system::FileSystem::findOrCreateAValidPath(d_axbDir.getValue());

    auto* liverNode = dynamic_cast<sofa::simulation::Node*>(this->getContext());
    if (liverNode)
    {
        auto* surfNode = liverNode->getChild("Surf");
        if (surfNode)
        {
            m_contactDofs = dynamic_cast<sofa::core::behavior::MechanicalState<sofa::defaulttype::Vec3Types>*>(
                surfNode->getObject("spheres"));
        }
    }

    if (m_contactDofs)
    {
        const auto& surfPos = m_contactDofs->read(
            sofa::core::ConstVecCoordId::position())->getValue();
        m_nSurfaceVerts = static_cast<int>(surfPos.size());
    }
    else
    {
        m_nSurfaceVerts = 0;
    }
}
void DataCollector::bwdInit()
{
    // Search the entire tree from the absolute root automatically
    this->getContext()->get(m_lcpFF, sofa::core::objectmodel::BaseContext::SearchRoot);

    if (m_lcpFF)
    {
        // Using msg_error just so it highlights in RED in the terminal!
        msg_error() << "SUCCESS! DataCollector: LCPForceFeedback found in the scene!";
    }
    else
    {
        msg_error() << "FAILED: DataCollector: LCPForceFeedback STILL NOT FOUND.";
    }

    // ── AX=B SPIKE: locate the solver that assembles A and solves Ax=b ──
    m_solver = this->getContext()->get<sofa::core::behavior::LinearSolver>(
        sofa::core::objectmodel::BaseContext::SearchRoot);

    if (m_solver)
        msg_error() << "AX=B SPIKE: Found LinearSolver '" << m_solver->getName() << "'.";
    else
        msg_error() << "AX=B SPIKE: No LinearSolver found in scene.";
}
void DataCollector::writeHeader()
{
    m_file << "step,real_time,sim_time,dt_since_last,session_id";

    // Tool state — unchanged
    m_file << ",tool_x,tool_y,tool_z";
    m_file << ",tool_vx,tool_vy,tool_vz";
    m_file << ",tool_fx,tool_fy,tool_fz";

    // ── NEW: per-vertex liver forces (from LCP + FEM) ─────────────
    for (int i = 0; i < m_nVertices; i++)
        m_file << ",fvx" << i << ",fvy" << i << ",fvz" << i;
    // ─────────────────────────────────────────────────────────────
    for (int i = 0; i < m_nVertices; i++)
        m_file << ",sax" << i << ",say" << i << ",saz" << i;
    // ── NEW: per-vertex liver velocity (from CG solver result) ────
    for (int i = 0; i < m_nVertices; i++)
        m_file << ",vvx" << i << ",vvy" << i << ",vvz" << i;
    // ─────────────────────────────────────────────────────────────

    // ── NEW: previous deformation (what solver used as x_old) ─────
    for (int i = 0; i < m_nVertices; i++)
        m_file << ",pdx" << i << ",pdy" << i << ",pdz" << i;
    // ─────────────────────────────────────────────────────────────

    // Current deformation — unchanged
    for (int i = 0; i < m_nVertices; i++)
        m_file << ",dx" << i << ",dy" << i << ",dz" << i;

    // ── NEW: real (Hooke's law) per-vertex stress/strain ──────────
    for (int i = 0; i < m_nVertices; i++)
        m_file << ",rsxx" << i << ",rsyy" << i << ",rszz" << i;
    for (int i = 0; i < m_nVertices; i++)
        m_file << ",rexx" << i << ",reyy" << i << ",rezz" << i;
    // ─────────────────────────────────────────────────────────────

    m_file << "\n";
    m_file.flush();
}

void DataCollector::handleEvent(sofa::core::objectmodel::Event* e)
{
    if (sofa::simulation::AnimateEndEvent::checkEventType(e))
    {
        m_step++;
        writeSample();
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

    if (!m_instrDofs)
    {
        auto* root = dynamic_cast<sofa::simulation::Node*>(
            this->getContext()->getRootContext());
        if (root)
        {
            if (!d_toolPath.getValue().empty())
            {
                auto* obj = root->getObject(d_toolPath.getValue());
                m_instrDofs = dynamic_cast<sofa::core::behavior::MechanicalState<sofa::defaulttype::Rigid3Types>*>(obj);
            }

            auto* instrNode = root->getChild("Instrument");
            if (instrNode)
            {
                m_instrDofs = dynamic_cast<sofa::core::behavior::MechanicalState<sofa::defaulttype::Rigid3Types>*>(
                    instrNode->getObject("instrumentState"));
                if (m_instrDofs)
                    msg_info() << "DataCollector: Instrument found.";
    
                
            }
        }
        if (!m_instrDofs)
            msg_warning() << "DataCollector: Instrument MechanicalState not found.";
    }

    // ── Tool state ────────────────────────────────────────────────
    sofa::type::Vec3d toolPos(0.0, 0.0, 0.0);
    sofa::type::Vec3d toolVel(0.0, 0.0, 0.0);
    sofa::type::Vec3d toolForce(0.0, 0.0, 0.0);

    // Default time fallback if haptics module isn't ready
    double sim_time = this->getContext()->getTime();
    double real_time = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - m_startRealTime).count();

    if (m_instrDofs)
    {
        // ... keep your existing pos and vel reading code exactly as it is ...
        const auto& pos = m_instrDofs->read(sofa::core::ConstVecCoordId::position())->getValue();
        if (!pos.empty()) {
            toolPos[0] = pos[0].getCenter()[0]; toolPos[1] = pos[0].getCenter()[1]; toolPos[2] = pos[0].getCenter()[2];
        }
        const auto& vel = m_instrDofs->read(sofa::core::ConstVecDerivId::velocity())->getValue();
        if (!vel.empty()) {
            toolVel[0] = vel[0].getLinear()[0]; toolVel[1] = vel[0].getLinear()[1]; toolVel[2] = vel[0].getLinear()[2];
        }

        // Updated to extract the comprehensive struct packet safely
        if (m_lcpFF)
        {
            auto hapticData = m_lcpFF->getForce();
            toolForce = hapticData.force;
            sim_time  = hapticData.sim_time; // Syncs the explicit calculation time
            real_time = std::chrono::duration<double>(
                hapticData.real_time - m_startRealTime).count();
        }
    }

    // ── Time gap since the previous sample ─────────────────────────
    const double dt_since_last = sim_time - m_lastSimTime;
    m_lastSimTime = sim_time;

    // ── Liver positions ───────────────────────────────────────────
    const auto& curPos = m_liverDofs->read(sofa::core::ConstVecCoordId::position())->getValue();
    const auto& freePos = m_liverDofs->read(sofa::core::ConstVecCoordId::freePosition())->getValue();
    const auto& restPos = m_liverDofs->read(sofa::core::ConstVecCoordId::restPosition())->getValue();

    // ── Liver per-vertex forces and velocities ────────────────────
    const auto& vertexForces = m_liverDofs->read(sofa::core::ConstVecDerivId::force())->getValue();
    const auto& vertexVels = m_liverDofs->read(sofa::core::ConstVecDerivId::velocity())->getValue();

    // ── CONTINUOUS TRACKING GATEWAY (No early returns) ────────────
    if (toolForce.norm() < 0.01) 
    {
        m_isRecording = false; 
    }
    else 
    {
        if (!m_isRecording) 
        {
            m_isRecording = true;
            m_sessionId++; 
        }
    }

    // Proxy force = where tissue wanted to go - where it actually went
    std::vector<sofa::type::Vec3d> contactProxy(m_nVertices, sofa::type::Vec3d(0,0,0));
    for (int i = 0; i < m_nVertices && i < (int)freePos.size(); i++)
    {
        contactProxy[i] = freePos[i] - curPos[i];
    }    

    if (!m_accStressInit)
    {
        m_accStress.assign(m_nVertices, sofa::type::Vec3d(0,0,0));
        m_accStressInit = true;
    }

    for (int i = 0; i < m_nVertices; i++)
    {
        m_accStress[i][0] = m_alpha * m_accStress[i][0] + (1-m_alpha) * contactProxy[i][0];
        m_accStress[i][1] = m_alpha * m_accStress[i][1] + (1-m_alpha) * contactProxy[i][1];
        m_accStress[i][2] = m_alpha * m_accStress[i][2] + (1-m_alpha) * contactProxy[i][2];
    }

    // ── Write row ─────────────────────────────────────────────────
    // Writes step, the high-precision simulation time, and our running session index
    m_file << m_step << "," << real_time << "," << sim_time << "," << dt_since_last
           << "," << (m_isRecording ? m_sessionId : 0)
           << "," << toolPos[0]   << "," << toolPos[1]   << "," << toolPos[2]
           << "," << toolVel[0]   << "," << toolVel[1]   << "," << toolVel[2]
           << "," << toolForce[0] << "," << toolForce[1] << "," << toolForce[2];
    // ── NEW: per-vertex liver forces ──────────────────────────────
    // REPLACE WITH — write contact proxy AND accumulated stress:
for (int i = 0; i < m_nVertices; i++)
{
    m_file << "," << contactProxy[i][0]
           << "," << contactProxy[i][1]
           << "," << contactProxy[i][2];
}
for (int i = 0; i < m_nVertices; i++)
{
    m_file << "," << m_accStress[i][0]
           << "," << m_accStress[i][1]
           << "," << m_accStress[i][2];
}
    // ─────────────────────────────────────────────────────────────

    // ── NEW: per-vertex liver velocities ──────────────────────────
    for (int i = 0; i < m_nVertices; i++)
    {
        m_file << "," << vertexVels[i][0]
               << "," << vertexVels[i][1]
               << "," << vertexVels[i][2];
    }
    // ─────────────────────────────────────────────────────────────

    // ── NEW: previous deformation (x_old from solver perspective) ─
    for (int i = 0; i < m_nVertices; i++)
    {
        m_file << "," << m_prevDeform[i][0]
               << "," << m_prevDeform[i][1]
               << "," << m_prevDeform[i][2];
    }
    // ─────────────────────────────────────────────────────────────

    // ── Current deformation — unchanged ──────────────────────────
    for (int i = 0; i < m_nVertices; i++)
    {
        m_file << "," << (curPos[i][0] - restPos[i][0])
               << "," << (curPos[i][1] - restPos[i][1])
               << "," << (curPos[i][2] - restPos[i][2]);
    }

    // ── NEW: real (Hooke's law) per-vertex stress/strain ──────────
    std::vector<sofa::type::Vec3d> vertStress(m_nVertices, sofa::type::Vec3d(0,0,0));
    std::vector<sofa::type::Vec3d> vertStrain(m_nVertices, sofa::type::Vec3d(0,0,0));
    computeRealStressStrain(vertStress, vertStrain);
    for (int i = 0; i < m_nVertices; i++)
    {
        m_file << "," << vertStress[i][0]
               << "," << vertStress[i][1]
               << "," << vertStress[i][2];
    }
    for (int i = 0; i < m_nVertices; i++)
    {
        m_file << "," << vertStrain[i][0]
               << "," << vertStrain[i][1]
               << "," << vertStrain[i][2];
    }
    // ─────────────────────────────────────────────────────────────

    m_file << "\n";

    if ((m_step / d_collectEvery.getValue()) % 10 == 0)
        m_file.flush();

    // ── NEW: update previous deformation buffer ───────────────────
    for (int i = 0; i < m_nVertices; i++)
        m_prevDeform[i] = curPos[i] - restPos[i];
    // ─────────────────────────────────────────────────────────────

    // ── AX=B SPIKE: log A, b, x shapes for the first few steps ────────────
    if (m_solver && m_step <= 3)
    {
        auto* A = m_solver->getSystemBaseMatrix();
        auto* b = m_solver->getSystemRHBaseVector();
        auto* x = m_solver->getSystemLHBaseVector();
        msg_error() << "AX=B SPIKE step " << m_step << ": A="
                     << (A ? (std::to_string(A->rowSize()) + "x" + std::to_string(A->colSize())) : "null")
                     << " (type=" << (A ? typeid(*A).name() : "n/a") << ")"
                     << ", b size=" << (b ? std::to_string(b->size()) : "null")
                     << " (type=" << (b ? typeid(*b).name() : "n/a") << ")"
                     << ", x size=" << (x ? std::to_string(x->size()) : "null");
    }

    // ── AX=B DUMP: every dumpAxbEvery steps, write b and x to disk ────────
    // Pair with a GlobalSystemMatrixExporter (same exportEveryNbSteps) for A_NNNNN.csv.
    if (m_solver && d_dumpAxbEvery.getValue() > 0 && m_step % d_dumpAxbEvery.getValue() == 0)
    {
        const int frameIdx = m_step / d_dumpAxbEvery.getValue();
        dumpVector(m_solver->getSystemRHBaseVector(), frameIdx, "b");
        dumpVector(m_solver->getSystemLHBaseVector(), frameIdx, "x");
    }

    // ADD temporarily for debugging:
    if (m_step <= 100)
    {
        double maxF = 0.0;
        for (int i = 0; i < m_nVertices; i++)
            maxF = std::max(maxF, vertexForces[i].norm());
        msg_info() << "Step " << m_step << " max vertex force = " << maxF;
    }

    m_prevToolPos = toolPos;
    m_firstStep   = false;
}

// Real (Hooke's law) per-vertex stress/strain, replicating
// TetrahedronFEMForceField's corotational "large" formulation
// (accumulateForceLarge) via its public getters. For each tetrahedron we
// recompute the local displacement D, then strain = J^T * D and
// stress = K * strain (both 6-Voigt: [xx, yy, zz, yz, xz, xy]). We export
// only the 3 diagonal (normal) components, averaged over the tetrahedra
// sharing each vertex.
void DataCollector::computeRealStressStrain(std::vector<sofa::type::Vec3d>& vertStress,
                                             std::vector<sofa::type::Vec3d>& vertStrain)
{
    if (!m_femFF || !m_topo || !m_liverDofs) return;

    const auto& curPos = m_liverDofs->read(sofa::core::ConstVecCoordId::position())->getValue();
    const auto& tetrahedra = m_topo->getTetrahedra();

    std::vector<int> vertCount(m_nVertices, 0);

    for (size_t t = 0; t < tetrahedra.size(); ++t)
    {
        const auto& tet = tetrahedra[t];

        const auto& R_2_0 = m_femFF->getActualTetraRotation((unsigned int)t);
        const auto R_0_2 = R_2_0.transposed();

        sofa::type::Vec3d deforme[4];
        for (int i = 0; i < 4; ++i)
            deforme[i] = R_0_2 * curPos[tet[i]];

        deforme[1][0] -= deforme[0][0];
        deforme[2][0] -= deforme[0][0];
        deforme[2][1] -= deforme[0][1];
        deforme[3]    -= deforme[0];

        const auto& rotatedInit = m_femFF->getRotatedInitialElements((unsigned int)t);

        sofa::type::Vec<12, double> D;
        D[3]  = rotatedInit[1][0] - deforme[1][0];
        D[6]  = rotatedInit[2][0] - deforme[2][0];
        D[7]  = rotatedInit[2][1] - deforme[2][1];
        D[9]  = rotatedInit[3][0] - deforme[3][0];
        D[10] = rotatedInit[3][1] - deforme[3][1];
        D[11] = rotatedInit[3][2] - deforme[3][2];

        const auto& J = m_femFF->getStrainDisplacement((unsigned int)t);
        const auto& K = m_femFF->getMaterialStiffness((unsigned int)t);

        const auto strain = J.multTranspose(D);
        const auto stress = K * strain;

        for (int i = 0; i < 4; ++i)
        {
            const auto v = tet[i];
            vertStress[v][0] += stress[0];
            vertStress[v][1] += stress[1];
            vertStress[v][2] += stress[2];
            vertStrain[v][0] += strain[0];
            vertStrain[v][1] += strain[1];
            vertStrain[v][2] += strain[2];
            vertCount[v]++;
        }
    }

    for (int v = 0; v < m_nVertices; ++v)
    {
        if (vertCount[v] > 0)
        {
            vertStress[v] /= (double)vertCount[v];
            vertStrain[v] /= (double)vertCount[v];
        }
    }
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