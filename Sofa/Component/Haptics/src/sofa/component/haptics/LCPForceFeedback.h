/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#pragma once
#include <sofa/component/haptics/config.h>

#include <sofa/component/haptics/MechanicalStateForceFeedback.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/helper/system/thread/CTime.h>
#include <mutex>
#include <vector>  // <--- ADD THIS
#include <chrono>  // <--- ADD THIS

#include <sofa/component/constraint/lagrangian/solver/ConstraintSolverImpl.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/component/solidmechanics/fem/elastic/TetrahedronFEMForceField.h>
#include "/home/yogyaahuja/sofa/pinn_project/cpp/PINNPredictor.h"

namespace sofa::component::haptics
{

/**
* LCP force field
*/
template <class TDataTypes>
class LCPForceFeedback : public MechanicalStateForceFeedback<TDataTypes>
{
public:

    SOFA_CLASS(SOFA_TEMPLATE(LCPForceFeedback,TDataTypes),MechanicalStateForceFeedback<TDataTypes>);

    typedef TDataTypes DataTypes;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::MatrixDeriv MatrixDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::MatrixDeriv::RowConstIterator MatrixDerivRowConstIterator;
    typedef typename DataTypes::MatrixDeriv::ColConstIterator MatrixDerivColConstIterator;
    typedef typename DataTypes::MatrixDeriv::RowIterator MatrixDerivRowIterator;
    typedef typename DataTypes::MatrixDeriv::ColIterator MatrixDerivColIterator;

    void init() override;

    void draw( const core::visual::VisualParams* ) override
    {
        dmsg_info() << "haptic_freq = " << std::fixed << haptic_freq << " Hz   " << '\xd';
    }

    Data< double > forceCoef; ///< multiply haptic force by this coef.
    Data< bool >   d_usePINN; ///< Set false in data-collection scenes to skip PINN forward pass (saves 3-5ms/step)

    Data< double > solverTimeout; ///< max time to spend solving constraints.

    Data< int > d_solverMaxIt; ///< max iteration to spend solving constraints.

    // deriv (or not) the rotations when updating the violations
    Data <bool> d_derivRotations; ///< if true, deriv the rotations when updating the violations

    // Enable/disable constraint haptic influence from all frames
    Data< bool > d_localHapticConstraintAllFrames; ///< Flag to enable/disable constraint haptic influence from all frames

    void computeForce(SReal x, SReal y, SReal z,
                      SReal u, SReal v, SReal w,
                      SReal q, SReal& fx, SReal& fy, SReal& fz) override;
    void computeWrench(const sofa::defaulttype::SolidTypes<SReal>::Transform &world_H_tool,
                       const sofa::defaulttype::SolidTypes<SReal>::SpatialVector &V_tool_world,
                       sofa::defaulttype::SolidTypes<SReal>::SpatialVector &W_tool_world ) override;
    void computeForce(const  VecCoord& state,  VecDeriv& forces) override;

protected:
    LCPForceFeedback();
    ~LCPForceFeedback() override
    {
        delete(_timer);
    }

    virtual void updateStats();
    virtual bool updateConstraintProblem();
    virtual void doComputeForce(const  VecCoord& state,  VecDeriv& forces);


public:
    void handleEvent(sofa::core::objectmodel::Event *event) override;


    /// Pre-construction check method called by ObjectFactory.
    /// Check that DataTypes matches the MechanicalState.
    template<class T>
    static bool canCreate(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        if (dynamic_cast< core::behavior::MechanicalState<DataTypes>* >(context->getMechanicalState()) == nullptr) {
            arg->logError(std::string("No mechanical state with the datatype '") + DataTypes::Name() + "' found in the context node.");
            return false;
        }

        return core::objectmodel::BaseObject::canCreate(obj, context, arg);
    }
    struct forcetime {
        sofa::type::Vec3d force;
        std::chrono::high_resolution_clock::time_point real_time;
        double sim_time; // Added for your PINN synchronization
    };

    forcetime getForce() const {
        return ft;
    }

    /// Overide method to lock or unlock the force feedback computation. According to parameter, value == true (resp. false) will lock (resp. unlock) mutex @sa lockForce
    void setLock(bool value) override;

protected:
    core::behavior::MechanicalState<DataTypes> *mState; ///< The device try to follow this mechanical state.
    VecCoord mVal[3];
    MatrixDeriv mConstraints[3];
    std::vector<int> mId_buf[3];
    struct forcetime ft{};
    component::constraint::lagrangian::solver::ConstraintProblem* mCP[3];
    sofa::type::Vec3d currentForce {0.0, 0.0, 0.0};
   
     std::vector<forcetime> force_history;

     // For external access to current constraint problem (e.g. for logging)
    unsigned char mNextBufferId; // Next buffer id to be use
    unsigned char mCurBufferId; // Current buffer id in use
    bool mIsCuBufferInUse; // Is current buffer currently in use right now

    sofa::component::constraint::lagrangian::solver::ConstraintSolverImpl* constraintSolver;

    /// timer: verifies the time rates of the haptic loop
    helper::system::thread::CTime *_timer;
    helper::system::thread::ctime_t time_buf;
    int timer_iterations;
    double haptic_freq;
    unsigned int num_constraints;

    /// mutex used in method @doComputeForce which can be touched from outside using method @sa setLock if components are modified in another thread.
    std::mutex lockForce;

    // ── PINN real-time predictor ─────────────────────────────────────────────
    PINNPredictor* m_pinn        {nullptr};
    bool           m_usePINN     {false};

    // Liver DOFs + FEM force field (same as DataCollector, found in init)
    sofa::core::behavior::MechanicalState<sofa::defaulttype::Vec3Types>* m_liverDofs {nullptr};
    sofa::component::solidmechanics::fem::elastic::TetrahedronFEMForceField
        <sofa::defaulttype::Vec3Types>*                                   m_femFF     {nullptr};
    sofa::core::topology::BaseMeshTopology*                               m_liverTopo {nullptr};

    // Previous liver deformation (curPos - restPos) for computing delta
    std::vector<sofa::type::Vec3d> m_prevDeform;

    // Contact-proxy EMA (same alpha as DataCollector)
    std::vector<sofa::type::Vec3d> m_accStress;
    double m_stressAlpha {0.9};
    bool   m_accStressInit {false};

    // Liver rest positions precomputed at init — safe to read from any thread
    std::vector<sofa::type::Vec3d> m_liverRestPos;

    // PINN runs on SOFA thread (AnimateEndEvent); haptic thread reads this cache
    sofa::type::Vec3d m_pinnCachedForce {0.0, 0.0, 0.0};
    std::mutex        m_pinnCacheMutex;

    // Deterministic step counter — NOT wall-clock, NOT elapsed sim-time. Must match
    // DataCollector's collectEvery exactly (currently 3): call predictForce once every
    // 3 AnimateEndEvents, same basis GeomagicDriver's REPLAY_SIM_STRIDE uses to advance
    // the replay trajectory. No clock arithmetic anywhere -> no drift possible.
    static constexpr int PINN_CALL_STEP_STRIDE = 3;
    int    m_pinnStepCounter {0};
    bool   m_pinnTimerSet {false};  // true after first predictForce call (for dt_pred base)
    double m_pinnStartSimTime {0.0};
    double m_lastPINNCallElapsed {-1.0};
    float  m_prevTxForVel {0.f}, m_prevTyForVel {0.f}, m_prevTzForVel {0.f};
};

// ── KEEP ONLY THESE DECLARATIONS AT THE BOTTOM ──
template <typename DataTypes>
void LCPForceFeedback<DataTypes>::computeForce(SReal, SReal, SReal, SReal, SReal, SReal, SReal, SReal&, SReal&, SReal&) {}

template <typename DataTypes>
void LCPForceFeedback<DataTypes>::computeWrench(const sofa::defaulttype::SolidTypes<SReal>::Transform &,
        const sofa::defaulttype::SolidTypes<SReal>::SpatialVector &,
        sofa::defaulttype::SolidTypes<SReal>::SpatialVector & ) {}

template <>
void SOFA_COMPONENT_HAPTICS_API LCPForceFeedback< sofa::defaulttype::Rigid3Types >::computeForce(SReal x, SReal y, SReal z, SReal, SReal, SReal, SReal, SReal& fx, SReal& fy, SReal& fz);

template <>
void SOFA_COMPONENT_HAPTICS_API LCPForceFeedback< sofa::defaulttype::Rigid3Types >::computeWrench(const sofa::defaulttype::SolidTypes<SReal>::Transform &world_H_tool,
        const sofa::defaulttype::SolidTypes<SReal>::SpatialVector &/*V_tool_world*/,
        sofa::defaulttype::SolidTypes<SReal>::SpatialVector &W_tool_world );

#if !defined(SOFA_COMPONENT_CONTROLLER_LCPFORCEFEEDBACK_CPP)
extern template class SOFA_COMPONENT_HAPTICS_API LCPForceFeedback<defaulttype::Vec1Types>;
extern template class SOFA_COMPONENT_HAPTICS_API LCPForceFeedback<defaulttype::Rigid3Types>;
#endif

} // namespace sofa::component::haptics
