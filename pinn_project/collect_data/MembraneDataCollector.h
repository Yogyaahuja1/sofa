#pragma once

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/core/objectmodel/Data.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/type/Vec.h>
#include <sofa/component/haptics/LCPForceFeedback.h>
#include <sofa/component/solidmechanics/fem/elastic/TriangularFEMForceField.h>

#include <chrono>
#include <fstream>
#include <string>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

namespace pinn
{

struct MembraneFrameData {
    int    step      {0};
    double real_time {0};
    double sim_time  {0};
    double dt        {0};
    int    session_id{0};
    double toolPos[3]{};
    double toolVel[3]{};
    double toolForce[3]{};
    // Groups per vertex: prevDeform | deform | vertStress | vertStrain
    // (4 groups × nVerts × 3 doubles)
    std::vector<double> vals;
};

class MembraneDataCollector : public sofa::core::objectmodel::BaseObject
{
public:
    SOFA_CLASS(MembraneDataCollector, sofa::core::objectmodel::BaseObject);

    sofa::core::objectmodel::Data<std::string> d_outputFile;
    sofa::core::objectmodel::Data<int>         d_collectEvery;

    MembraneDataCollector();
    ~MembraneDataCollector() override;

    void init() override;
    void handleEvent(sofa::core::objectmodel::Event* e) override;
    void bwdInit() override;

private:
    sofa::core::behavior::MechanicalState<sofa::defaulttype::Vec3Types>*        m_membraneDofs  {nullptr};
    sofa::component::haptics::LCPForceFeedback<sofa::defaulttype::Rigid3Types>* m_lcpFF         {nullptr};
    sofa::core::behavior::MechanicalState<sofa::defaulttype::Rigid3Types>*      m_instrDofs     {nullptr};
    sofa::component::solidmechanics::fem::elastic::TriangularFEMForceField
        <sofa::defaulttype::Vec3Types>*                                          m_triangleFemFF {nullptr};
    sofa::core::topology::BaseMeshTopology*                                      m_topo          {nullptr};

    bool   m_isRecording   {false};
    int    m_sessionId     {0};
    int    m_step          {0};
    int    m_nVertices     {0};
    bool   m_headerWritten {false};
    double m_lastSimTime   {0.0};

    std::vector<sofa::type::Vec3d> m_prevDeform;
    std::vector<sofa::type::Vec3d> m_contactProxy;
    std::vector<sofa::type::Vec3d> m_accStress;
    bool   m_accStressInit {false};
    double m_alpha         {0.9};
    std::vector<sofa::type::Vec3d> m_vertStress;
    std::vector<sofa::type::Vec3d> m_vertStrain;
    std::vector<int>               m_vertCount;

    MembraneFrameData m_frame;

    sofa::type::Vec3d m_prevToolPos {0.0, 0.0, 0.0};

    std::chrono::high_resolution_clock::time_point m_startRealTime;

    std::ofstream                   m_file;
    std::queue<MembraneFrameData>   m_frameQueue;
    std::mutex                      m_queueMutex;
    std::condition_variable         m_queueCV;
    std::thread                     m_writerThread;
    std::atomic<bool>               m_stopWriter {false};

    void writerLoop();
    void writeHeader();
    void writeSample();
    void readFEMCache();
};

} // namespace pinn
