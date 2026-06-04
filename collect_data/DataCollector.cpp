#include "DataCollector.h"
#include <sofa/core/ObjectFactory.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <sofa/simulation/Node.h>

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

    const auto& restPos = m_liverDofs->read(
        sofa::core::ConstVecCoordId::restPosition())->getValue();
    m_nVertices = (int)restPos.size();

    // ── NEW: initialise previous deformation buffer to zeros ──────
    m_prevDeform.assign(m_nVertices, sofa::type::Vec3d(0.0, 0.0, 0.0));
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

void DataCollector::writeHeader()
{
    m_file << "step";

    // Tool state — unchanged
    m_file << ",tool_x,tool_y,tool_z";
    m_file << ",tool_vx,tool_vy,tool_vz";
    m_file << ",tool_fx,tool_fy,tool_fz";

    // ── NEW: per-vertex liver forces (from LCP + FEM) ─────────────
    for (int i = 0; i < m_nSurfaceVerts; i++)
        m_file << ",fvx" << i << ",fvy" << i << ",fvz" << i;
    // ─────────────────────────────────────────────────────────────

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

    m_file << "\n";
    m_file.flush();
}

void DataCollector::handleEvent(sofa::core::objectmodel::Event* e)
{
    if (sofa::simulation::AnimateEndEvent::checkEventType(e))
    {
        m_step++;
        if (m_step % d_collectEvery.getValue() == 0)
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

    if (m_instrDofs)
    {
        const auto& pos = m_instrDofs->read(
            sofa::core::ConstVecCoordId::position())->getValue();
        if (!pos.empty())
        {
            toolPos[0] = pos[0].getCenter()[0];
            toolPos[1] = pos[0].getCenter()[1];
            toolPos[2] = pos[0].getCenter()[2];
        }

        const auto& vel = m_instrDofs->read(
            sofa::core::ConstVecDerivId::velocity())->getValue();
        if (!vel.empty())
        {
            toolVel[0] = vel[0].getLinear()[0];
            toolVel[1] = vel[0].getLinear()[1];
            toolVel[2] = vel[0].getLinear()[2];
        }

        const auto& forces = m_instrDofs->read(
            sofa::core::ConstVecDerivId::force())->getValue();
        if (!forces.empty())
        {
            toolForce[0] = forces[0].getLinear()[0];
            toolForce[1] = forces[0].getLinear()[1];
            toolForce[2] = forces[0].getLinear()[2];
        }


    }

    // ── Liver positions ───────────────────────────────────────────
    const auto& curPos = m_liverDofs->read(
        sofa::core::ConstVecCoordId::position())->getValue();
    const auto& restPos = m_liverDofs->read(
        sofa::core::ConstVecCoordId::restPosition())->getValue();

    // DEBUG: check gravity sag
    if (m_step % 100 == 0)
    {
        double avgDisp = 0.0;
        double maxDispDbg = 0.0;

        for (int i = 0; i < m_nVertices; i++)
        {
            double d = (curPos[i] - restPos[i]).norm();
            avgDisp += d;
            maxDispDbg = std::max(maxDispDbg, d);
        }

        avgDisp /= m_nVertices;

        msg_info() << "Step " << m_step
                << " avg deformation = " << avgDisp
                << " max deformation = " << maxDispDbg;
    }

    // ── NEW: liver vertex forces (LCP contact + FEM internal) ─────
    const auto& vertexForces = (m_contactDofs ? m_contactDofs : m_liverDofs)->read(
        sofa::core::ConstVecDerivId::force())->getValue();
    // ─────────────────────────────────────────────────────────────

    // ── NEW: liver vertex velocities (result of CG solver) ────────
    const auto& vertexVels = m_liverDofs->read(
        sofa::core::ConstVecDerivId::velocity())->getValue();
    // ─────────────────────────────────────────────────────────────

    // ── Sanity check ──────────────────────────────────────────────
    double maxDisp = 0.0;
    for (int i = 0; i < m_nVertices; i++)
    {
        double d = std::abs(curPos[i][1] - restPos[i][1]);
        maxDisp = std::max(maxDisp, d);
    }
    if (maxDisp > 50.0)
    {
        // Still update previous deformation buffer even on skipped frame
        for (int i = 0; i < m_nVertices; i++)
            m_prevDeform[i] = curPos[i] - restPos[i];
        return;
    }

    // ── Write row ─────────────────────────────────────────────────
    m_file << m_step
           << "," << toolPos[0]   << "," << toolPos[1]   << "," << toolPos[2]
           << "," << toolVel[0]   << "," << toolVel[1]   << "," << toolVel[2]
           << "," << toolForce[0] << "," << toolForce[1] << "," << toolForce[2];

    // ── NEW: per-vertex liver forces ──────────────────────────────
    int nForceVerts = static_cast<int>(vertexForces.size());
    for (int i = 0; i < nForceVerts; i++)
    {
        m_file << "," << vertexForces[i][0]
               << "," << vertexForces[i][1]
               << "," << vertexForces[i][2];
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

    m_file << "\n";

    if ((m_step / d_collectEvery.getValue()) % 10 == 0)
        m_file.flush();

    // ── NEW: update previous deformation buffer ───────────────────
    for (int i = 0; i < m_nVertices; i++)
        m_prevDeform[i] = curPos[i] - restPos[i];
    // ─────────────────────────────────────────────────────────────

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

} // namespace pinn

extern "C" {
    void initExternalModule() {}
    const char* getModuleName()    { return "PINNDataCollector"; }
    const char* getModuleVersion() { return "1.0"; }
}