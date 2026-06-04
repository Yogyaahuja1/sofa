#pragma once

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/core/objectmodel/Data.h>
#include <fstream>
#include <string>
#include <vector>

namespace pinn
{

class DataCollector : public sofa::core::objectmodel::BaseObject
{
public:
    SOFA_CLASS(DataCollector, sofa::core::objectmodel::BaseObject);

    sofa::core::objectmodel::Data<std::string> d_outputFile;
    sofa::core::objectmodel::Data<int>         d_collectEvery;
    sofa::core::objectmodel::Data<std::string> d_toolPath;

    DataCollector();
    ~DataCollector() override;

    void init() override;
    void handleEvent(sofa::core::objectmodel::Event* e) override;

private:
    // Liver DOFs

    sofa::core::behavior::MechanicalState
        <sofa::defaulttype::Vec3Types>* m_liverDofs {nullptr};

    // Collision DOFs (contact forces live here, not on liverDofs)
    sofa::core::behavior::MechanicalState
        <sofa::defaulttype::Vec3Types>* m_contactDofs {nullptr};

    // Instrument DOFs (Rigid3d)
    sofa::core::behavior::MechanicalState
        <sofa::defaulttype::Rigid3Types>* m_instrDofs {nullptr};

    std::vector<sofa::type::Vec3d> m_prevDeform;  // ← ADD THIS
    std::ofstream m_file;
    int m_step     {0};
    int m_nVertices{0};
    int m_nSurfaceVerts{0};
    bool m_headerWritten{false};

    // Store previous tool position to compute velocity
    sofa::type::Vec3d m_prevToolPos {0.0, 0.0, 0.0};
    bool              m_firstStep   {true};
    double            m_dt          {0.005}; // must match scene dt

    // Force delay buffer — fixes timing mismatch
    sofa::type::Vec3d m_prevToolForce {0.0, 0.0, 0.0};  // ← ADD
    bool              m_hasForce      {false};       
         // ← ADD
    void writeHeader();
    void writeSample();
};

} // namespace pinn