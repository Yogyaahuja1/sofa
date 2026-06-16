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
#include <chrono>
#include <fstream>
#include <string>
#include <vector>
#include <sofa/component/haptics/LCPForceFeedback.h>
#include <sofa/component/solidmechanics/fem/elastic/TetrahedronFEMForceField.h>

namespace pinn
{

class DataCollector : public sofa::core::objectmodel::BaseObject
{
public:
    SOFA_CLASS(DataCollector, sofa::core::objectmodel::BaseObject);

    sofa::core::objectmodel::Data<std::string> d_outputFile;
    sofa::core::objectmodel::Data<int>         d_collectEvery;
    sofa::core::objectmodel::Data<std::string> d_toolPath;

    // Ax=b dump: every N sim steps, write the solver's RHS (b) and solution (x)
    // vectors to <axbDir>/b_NNNNN.txt and x_NNNNN.txt (0 = disabled). Pair this
    // with a GlobalSystemMatrixExporter using the same exportEveryNbSteps so its
    // A_NNNNN.csv files line up with the same frame index.
    sofa::core::objectmodel::Data<int>         d_dumpAxbEvery;
    sofa::core::objectmodel::Data<std::string> d_axbDir;

    DataCollector();
    ~DataCollector() override;

    void init() override;
    void handleEvent(sofa::core::objectmodel::Event* e) override;
    void bwdInit() override;

    // --- ADD THIS NEW METHOD ---
    void setLCPForceFeedback(sofa::component::haptics::LCPForceFeedback<sofa::defaulttype::Rigid3Types>* lcp)
    {
        m_lcpFF = lcp;
        if (m_lcpFF)
            msg_info() << "DataCollector: LCPForceFeedback set externally.";
    }

private:
    // Liver DOFs

    sofa::core::behavior::MechanicalState
        <sofa::defaulttype::Vec3Types>* m_liverDofs {nullptr};

    sofa::component::haptics::LCPForceFeedback<sofa::defaulttype::Rigid3Types>* m_lcpFF {nullptr};

    // Collision DOFs (contact forces live here, not on liverDofs)
    sofa::core::behavior::MechanicalState
        <sofa::defaulttype::Vec3Types>* m_contactDofs {nullptr};

    // Instrument DOFs (Rigid3d)
    sofa::core::behavior::MechanicalState
        <sofa::defaulttype::Rigid3Types>* m_instrDofs {nullptr};

    bool m_isRecording {false};
    int  m_sessionId   {0};
    
    std::vector<sofa::type::Vec3d> m_prevDeform;  // ← ADD THIS
    std::ofstream m_file;
    int m_step     {0};
    int m_nVertices{0};
    int m_nSurfaceVerts{0};
    bool m_headerWritten{false};

    std::vector<sofa::type::Vec3d> m_accStress;
    bool m_accStressInit {false};
    double m_alpha {0.9};  // Exponential moving average factor for stress estimation

    // Real (Hooke's law) per-vertex stress/strain, from TetrahedronFEMForceField
    sofa::component::solidmechanics::fem::elastic::TetrahedronFEMForceField
        <sofa::defaulttype::Vec3Types>* m_femFF {nullptr};
    sofa::core::topology::BaseMeshTopology* m_topo {nullptr};

    void computeRealStressStrain(std::vector<sofa::type::Vec3d>& vertStress,
                                  std::vector<sofa::type::Vec3d>& vertStrain);

    // Store previous tool position to compute velocity
    sofa::type::Vec3d m_prevToolPos {0.0, 0.0, 0.0};
    bool              m_firstStep   {true};
    double            m_dt          {0.005}; // must match scene dt

    // Force delay buffer — fixes timing mismatch
    sofa::type::Vec3d m_prevToolForce {0.0, 0.0, 0.0};
    bool              m_hasForce      {false};

    // Simulation time of the previous sample, used to compute dt_since_last
    double m_lastSimTime {0.0};

    // Wall-clock reference point (set in init()), used to report real_time
    // as elapsed seconds since recording started rather than raw epoch time
    std::chrono::high_resolution_clock::time_point m_startRealTime;

    // Ax=b spike: the solver that assembles/solves A x = b each step
    sofa::core::behavior::LinearSolver* m_solver {nullptr};

    void writeHeader();
    void writeSample();
    void dumpVector(sofa::linearalgebra::BaseVector* v, int frameIdx, const char* prefix);
};

} // namespace pinn