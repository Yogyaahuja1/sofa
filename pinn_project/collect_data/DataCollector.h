#pragma once

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/core/objectmodel/Data.h>
#include <sofa/core/behavior/LinearSolver.h>
#include <sofa/linearalgebra/BaseVector.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/type/Vec.h>
#include <sofa/type/Mat.h>
#include <sofa/component/haptics/LCPForceFeedback.h>
#include <sofa/component/solidmechanics/fem/elastic/TetrahedronFEMForceField.h>

#include <charconv>
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

// Raw data captured on SOFA thread — no string conversion, no mmap.
// 30KB per frame (7 groups × 181 verts × 3 doubles).
// Writer thread converts this to CSV text asynchronously.
struct FrameData {
    int    step      {0};
    double real_time {0};
    double sim_time  {0};
    double dt        {0};
    int    session_id{0};
    double toolPos[3]{};
    double toolVel[3]{};
    double toolForce[3]{};
    // Flat layout: contactProxy | accStress | vels | prevDeform | deform | vertStress | vertStrain
    // Each group is nVertices × 3 doubles.
    std::vector<double> vals;
};

class DataCollector : public sofa::core::objectmodel::BaseObject
{
public:
    SOFA_CLASS(DataCollector, sofa::core::objectmodel::BaseObject);

    sofa::core::objectmodel::Data<std::string> d_outputFile;
    sofa::core::objectmodel::Data<int>         d_collectEvery;
    sofa::core::objectmodel::Data<std::string> d_toolPath;
    sofa::core::objectmodel::Data<int>         d_dumpAxbEvery;
    sofa::core::objectmodel::Data<std::string> d_axbDir;

    DataCollector();
    ~DataCollector() override;

    void init() override;
    void handleEvent(sofa::core::objectmodel::Event* e) override;
    void bwdInit() override;

    void setLCPForceFeedback(sofa::component::haptics::LCPForceFeedback<sofa::defaulttype::Rigid3Types>* lcp)
    {
        m_lcpFF = lcp;
        if (m_lcpFF) msg_info() << "DataCollector: LCPForceFeedback set externally.";
    }

private:
    sofa::core::behavior::MechanicalState<sofa::defaulttype::Vec3Types>*   m_liverDofs   {nullptr};
    sofa::component::haptics::LCPForceFeedback<sofa::defaulttype::Rigid3Types>* m_lcpFF  {nullptr};
    sofa::core::behavior::MechanicalState<sofa::defaulttype::Vec3Types>*   m_contactDofs {nullptr};
    sofa::core::behavior::MechanicalState<sofa::defaulttype::Rigid3Types>* m_instrDofs   {nullptr};
    sofa::component::solidmechanics::fem::elastic::TetrahedronFEMForceField
        <sofa::defaulttype::Vec3Types>*                                    m_femFF       {nullptr};
    sofa::core::topology::BaseMeshTopology*                                m_topo        {nullptr};
    sofa::core::behavior::LinearSolver*                                    m_solver      {nullptr};

    bool   m_isRecording   {false};
    int    m_sessionId     {0};
    int    m_step          {0};
    int    m_nVertices     {0};
    int    m_nSurfaceVerts {0};
    bool   m_headerWritten {false};
    bool   m_firstStep     {true};
    double m_dt            {0.005};
    double m_lastSimTime   {0.0};

    std::vector<sofa::type::Vec3d> m_prevDeform;
    std::vector<sofa::type::Vec3d> m_accStress;
    bool   m_accStressInit {false};
    double m_alpha         {0.9};

    // Per-call scratch buffers (zeroed each writeSample, no heap alloc)
    std::vector<sofa::type::Vec3d> m_vertStress;
    std::vector<sofa::type::Vec3d> m_vertStrain;
    std::vector<sofa::type::Vec3d> m_contactProxy;
    std::vector<int>               m_vertCount;

    // Pre-allocated frame — filled on SOFA thread, copied into queue
    FrameData m_frame;

    sofa::type::Vec3d m_prevToolPos   {0.0, 0.0, 0.0};
    sofa::type::Vec3d m_prevToolForce {0.0, 0.0, 0.0};
    bool              m_hasForce      {false};

    // Timing diagnostic (handleEvent wrapper)
    double m_writeSampleAvg   {0.0};
    int    m_writeSampleCount {0};

    std::chrono::high_resolution_clock::time_point m_startRealTime;

    // ── Async writer — queue holds raw FrameData (~30KB each, <128KB → arena alloc)
    // String building and disk I/O happen entirely on the writer thread.
    std::ofstream             m_file;
    std::queue<FrameData>     m_frameQueue;
    std::mutex                m_queueMutex;
    std::condition_variable   m_queueCV;
    std::thread               m_writerThread;
    std::atomic<bool>         m_stopWriter {false};

    void writerLoop();

    void writeHeader();
    void writeSample();
    void readFEMCache();
    void dumpVector(sofa::linearalgebra::BaseVector* v, int frameIdx, const char* prefix);
};

} // namespace pinn
