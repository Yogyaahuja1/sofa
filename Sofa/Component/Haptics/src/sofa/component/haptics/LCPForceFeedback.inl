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

#include <sofa/component/haptics/LCPForceFeedback.h>

#include <sofa/component/constraint/lagrangian/solver/ConstraintSolverImpl.h>

#include <sofa/simulation/AnimateEndEvent.h>

#include <algorithm>
#include <mutex>
#include <iostream>
#include <fstream>

namespace
{

template <typename DataTypes>
bool derivVectors(const typename DataTypes::VecCoord& x0, const typename DataTypes::VecCoord& x1, typename DataTypes::VecDeriv& d, bool /*derivRotation*/)
{
    size_t sz0 = x0.size();
    const size_t szmin = std::min(sz0,x1.size());

    d.resize(sz0);
    for(size_t i=0; i<szmin; ++i)
    {
        d[i]=x1[i]-x0[i];
    }
    for(size_t i=szmin; i<sz0; ++i) // not sure in what case this is applicable...
    {
        d[i]=-x0[i];
    }
    return true;
}


template <typename DataTypes>
bool derivRigid3Vectors(const typename DataTypes::VecCoord& x0, const typename DataTypes::VecCoord& x1, typename DataTypes::VecDeriv& d, bool derivRotation=false)
{
    size_t sz0 = x0.size();
    const size_t szmin = std::min(sz0,x1.size());

    d.resize(sz0);
    for(size_t i=0; i<szmin; ++i)
    {
        getVCenter(d[i]) = x1[i].getCenter() - x0[i].getCenter();
        if (derivRotation)
        {
            // rotations are taken into account to compute the violations
            sofa::type::Quat<SReal> q;
            getVOrientation(d[i]) = x0[i].rotate(q.angularDisplacement(x1[i].getOrientation(), x0[i].getOrientation() ) ); // angularDisplacement compute the rotation vector btw the two quaternions
        }
        else
            getVOrientation(d[i]) *= 0;
    }

    for(size_t i=szmin; i<sz0; ++i) // not sure in what case this is applicable..
    {
        getVCenter(d[i]) = - x0[i].getCenter();

        if (derivRotation)
        {
            // rotations are taken into account to compute the violations
            sofa::type::Quat<SReal> q= x0[i].getOrientation();
            getVOrientation(d[i]) = -x0[i].rotate( q.quatToRotationVector() );  // Use of quatToRotationVector instead of toEulerVector:
                                                                                // this is done to keep the old behavior (before the
                                                                                // correction of the toEulerVector  function). If the
                                                                                // purpose was to obtain the Eulerian vector and not the
                                                                                // rotation vector please use the following line instead
        }
        else
            getVOrientation(d[i]) *= 0;
    }

    return true;
}


template <typename DataTypes>
double computeDot(const typename DataTypes::Deriv& v0, const typename DataTypes::Deriv& v1)
{
    return dot(v0,v1);
}


template<>
bool derivVectors<sofa::defaulttype::Rigid3Types>(const sofa::defaulttype::Rigid3Types::VecCoord& x0, const sofa::defaulttype::Rigid3Types::VecCoord& x1, sofa::defaulttype::Rigid3Types::VecDeriv& d, bool derivRotation )
{
    return derivRigid3Vectors<sofa::defaulttype::Rigid3Types>(x0,x1,d, derivRotation);
}
template <>
double computeDot<sofa::defaulttype::Rigid3Types>(const sofa::defaulttype::Rigid3Types::Deriv& v0, const sofa::defaulttype::Rigid3Types::Deriv& v1)
{
    return dot(getVCenter(v0),getVCenter(v1)) + dot(getVOrientation(v0), getVOrientation(v1));
}



} // anonymous namespace

namespace sofa::component::haptics
{

template <class DataTypes>
LCPForceFeedback<DataTypes>::LCPForceFeedback()
    : forceCoef(initData(&forceCoef, 0.03, "forceCoef","multiply haptic force by this coef."))
    , d_usePINN(initData(&d_usePINN, true, "usePINN", "Set false in data-collection scenes to skip PINN forward pass"))
    , d_liveComparisonLog(initData(&d_liveComparisonLog, std::string(""), "liveComparisonLog",
        "Path to a CSV logging PINN's prediction vs the real force at the same live instant. Empty = no logging."))
    , solverTimeout(initData(&solverTimeout, 0.0008, "solverTimeout","max time to spend solving constraints."))
    , d_solverMaxIt(initData(&d_solverMaxIt, 100, "solverMaxIt", "max iteration to spend solving constraints"))
    , d_derivRotations(initData(&d_derivRotations, false, "derivRotations", "if true, deriv the rotations when updating the violations"))
    , d_localHapticConstraintAllFrames(initData(&d_localHapticConstraintAllFrames, false, "localHapticConstraintAllFrames", "Flag to enable/disable constraint haptic influence from all frames"))
    , mState(nullptr)
    , mNextBufferId(0)
    , mCurBufferId(0)
    , mIsCuBufferInUse(false)
    , constraintSolver(nullptr)
    , _timer(nullptr)
    , time_buf(0)
    , timer_iterations(0)
    , haptic_freq(0.0)
    , num_constraints(0)
{
    this->f_listening.setValue(true);
    mCP[0] = nullptr;
    mCP[1] = nullptr;
    mCP[2] = nullptr;
    _timer = new helper::system::thread::CTime();
    time_buf = _timer->getTime();
    timer_iterations = 0;
}


template <class DataTypes>
void LCPForceFeedback<DataTypes>::init()
{
    const core::objectmodel::BaseContext* c = this->getContext();

    this->ForceFeedback::init();
    if(!c)
    {
        msg_error() << "LCPForceFeedback has no current context. Initialisation failed.";
        return;
    }

    c->get(constraintSolver);

    if (!constraintSolver)
    {
        msg_error() << "LCPForceFeedback has no binding ConstraintSolver. Initialisation failed.";
        return;
    }

    mState = dynamic_cast<core::behavior::MechanicalState<DataTypes> *> (c->getMechanicalState());
    if (!mState)
    {
        msg_error() << "LCPForceFeedback has no binding MechanicalState. Initialisation failed.";
        return;
    }

    // ── PINN predictor init ──────────────────────────────────────────────────
    if (!d_usePINN.getValue())
    {
        msg_info() << "PINN disabled via usePINN=false — skipping model load (data-collection mode).";
    }
    else
    {
        m_pinn = new PINNPredictor();
        if (m_pinn->init(
            "/home/yogyaahuja/sofa/pinn_project/cpp/pinn_model_traced.pt",
            "/home/yogyaahuja/sofa/pinn_project/cpp/normalization_stats.csv",
            "/home/yogyaahuja/sofa/pinn_project/cpp/liver_vertices.csv"))
        {
            m_usePINN = true;
            msg_info() << "PINN predictor initialized — using PINN force instead of LCP solver.";
        }
        else
        {
            msg_warning() << "PINN predictor init failed — falling back to LCP solver.";
            delete m_pinn; m_pinn = nullptr;
        }
    }

    // Find liver DOFs by name to avoid picking up instrument collision meshes
    {
        std::vector<sofa::core::behavior::MechanicalState<sofa::defaulttype::Vec3Types>*> allStates;
        c->get<sofa::core::behavior::MechanicalState<sofa::defaulttype::Vec3Types>>(
            &allStates, core::objectmodel::BaseContext::SearchRoot);
        for (auto* ms : allStates)
            if (ms->getName() == "liverDofs") { m_liverDofs = ms; break; }
    }
    c->get(m_femFF,     core::objectmodel::BaseContext::SearchRoot);
    c->get(m_liverTopo, core::objectmodel::BaseContext::SearchRoot);
    if (m_liverDofs)
    {
        int nv = (int)m_liverDofs->getSize();
        m_prevDeform.assign(nv, sofa::type::Vec3d(0.0, 0.0, 0.0));
        m_accStress.assign(nv,  sofa::type::Vec3d(0.0, 0.0, 0.0));
        m_accStressInit = true;
        // Precompute rest positions once — safe to read from any thread
        const auto& rest = m_liverDofs->read(sofa::core::ConstVecCoordId::restPosition())->getValue();
        m_liverRestPos.assign(rest.begin(), rest.end());
        msg_info() << "PINN: Found liver MechanicalState '" << m_liverDofs->getName()
                   << "' with " << nv << " vertices.";
    }
    else
        msg_warning() << "PINN: 'liverDofs' MechanicalState not found — FEM buffer updates disabled.";
    if (!m_femFF)
        msg_warning() << "PINN: TetrahedronFEMForceField not found — real strain will be zero.";
}

template <class DataTypes>
void LCPForceFeedback<DataTypes>::setLock(bool value)
{
    value == true ? lockForce.lock() : lockForce.unlock();
}


static std::mutex s_mtx;

template <class DataTypes>
void LCPForceFeedback<DataTypes>::computeForce(const VecCoord& state,  VecDeriv& forces)
{
    if (!this->d_activate.getValue())
    {
        return;
    }
    updateStats();

    lockForce.lock(); // check if computation has not been locked using setLock method.
    updateConstraintProblem();
    doComputeForce(state, forces);
    lockForce.unlock();
}
template <class DataTypes>
void LCPForceFeedback<DataTypes>::updateStats()
{
    using namespace helper::system::thread;

    const ctime_t actualTime = _timer->getTime();
    ++timer_iterations;
    if (actualTime - time_buf >= sofa::helper::system::thread::CTime::getTicksPerSec())
    {
        haptic_freq = (double)(timer_iterations*sofa::helper::system::thread::CTime::getTicksPerSec())/ (double)( actualTime - time_buf) ;
        time_buf = actualTime;
        timer_iterations = 0;
    }
}

template <class DataTypes>
bool LCPForceFeedback<DataTypes>::updateConstraintProblem()
{
    const int prevId = mCurBufferId;

    //
    // Retrieve the last LCP and constraints computed by the Sofa thread.
    //
    mIsCuBufferInUse = true;

    {
        // TODO: Lock and/or memory barrier HERE
        mCurBufferId = mNextBufferId;
    }

    const bool changed = (prevId != mCurBufferId);

    const sofa::component::constraint::lagrangian::solver::ConstraintProblem* cp = mCP[mCurBufferId];

    if(!cp)
    {
        mIsCuBufferInUse = false;
    }

    return changed;
}

template <class DataTypes>
void LCPForceFeedback<DataTypes>::doComputeForce(const VecCoord& state,  VecDeriv& forces)
{
    const unsigned int stateSize = state.size();
    forces.resize(stateSize);
    for (unsigned int i = 0; i < forces.size(); ++i)
        forces[i].clear();

    // ── PINN force path: haptic thread reads cached force only (no CUDA here) ─
    if constexpr (std::is_same_v<DataTypes, sofa::defaulttype::Rigid3Types>)
    {
        if (m_usePINN && m_pinn && m_pinn->isInitialized() && !state.empty())
        {
            // Still compute and cache the REAL force here, into a separate member
            // never touched by the PINN path — ft.force (read by getForce()) gets
            // overwritten with the PINN's own cached output two lines below this
            // function returns, in the SReal computeForce(x,y,z,...) overload. If
            // computeRealForceForPINNFeedback read ft.force, it would be reading
            // PINN's own prior prediction once usePINN is on — a closed feedback
            // loop with no real anchor, confirmed by direct measurement: GT and PINN
            // caches tracked closely for the first ~100 rows, then diverged sharply
            // as the contamination compounded. This separate cache is computed fresh
            // every call, exactly like training's force was, but is immune to that.
            VecDeriv realForcesForCache;
            computeRealLCPForce(state, realForcesForCache);
            if (!realForcesForCache.empty())
                m_realForceCache = sofa::defaulttype::getVCenter(realForcesForCache[0]);

            std::lock_guard<std::mutex> lk(m_pinnCacheMutex);
            sofa::defaulttype::getVCenter(forces[0]) = m_pinnCachedForce;
            return;
        }
    }
    // ─────────────────────────────────────────────────────────────────────────

    computeRealLCPForce(state, forces);
    if constexpr (std::is_same_v<DataTypes, sofa::defaulttype::Rigid3Types>)
    {
        if (!forces.empty())
            m_realForceCache = sofa::defaulttype::getVCenter(forces[0]);
    }
}

template <class DataTypes>
void LCPForceFeedback<DataTypes>::computeRealLCPForce(const VecCoord& state, VecDeriv& forces)
{
    const unsigned int stateSize = state.size();
    forces.resize(stateSize);
    for (unsigned int i = 0; i < forces.size(); ++i)
        forces[i].clear();

    if(!constraintSolver||!mState)
        return;

    const MatrixDeriv& constraints = mConstraints[mCurBufferId];
    VecCoord &val = mVal[mCurBufferId];
    sofa::component::constraint::lagrangian::solver::ConstraintProblem* cp = mCP[mCurBufferId];

    if(!cp)
    {
        return;
    }

    if(!constraints.empty())
    {
        VecDeriv dx;

        derivVectors< DataTypes >(val, state, dx, d_derivRotations.getValue());

        const bool localHapticConstraintAllFrames = d_localHapticConstraintAllFrames.getValue();

        // Modify Dfree
        MatrixDerivRowConstIterator rowItEnd = constraints.end();
        num_constraints = constraints.size();

        for (MatrixDerivRowConstIterator rowIt = constraints.begin(); rowIt != rowItEnd; ++rowIt)
        {
            MatrixDerivColConstIterator colItEnd = rowIt.end();

            for (MatrixDerivColConstIterator colIt = rowIt.begin(); colIt != colItEnd; ++colIt)
            {
                cp->getDfree()[rowIt.index()] += computeDot<DataTypes>(colIt.val(), dx[localHapticConstraintAllFrames ? 0 : colIt.index()]);
            }
        }

        s_mtx.lock();

        // Solving constraints
        cp->solveTimed(cp->tolerance * 0.001, d_solverMaxIt.getValue(), solverTimeout.getValue());	// tol, maxIt, timeout

        // Restore Dfree
        for (MatrixDerivRowConstIterator rowIt = constraints.begin(); rowIt != rowItEnd; ++rowIt)
        {
            MatrixDerivColConstIterator colItEnd = rowIt.end();

            for (MatrixDerivColConstIterator colIt = rowIt.begin(); colIt != colItEnd; ++colIt)
            {
                cp->getDfree()[rowIt.index()] -= computeDot<DataTypes>(colIt.val(), dx[localHapticConstraintAllFrames ? 0 : colIt.index()]);
            }
        }

        s_mtx.unlock();

        VecDeriv tempForces;
        tempForces.resize(val.size());

        for (MatrixDerivRowConstIterator rowIt = constraints.begin(); rowIt != rowItEnd; ++rowIt)
        {
            if (cp->getF()[rowIt.index()] != 0.0)
            {
                MatrixDerivColConstIterator colItEnd = rowIt.end();

                for (MatrixDerivColConstIterator colIt = rowIt.begin(); colIt != colItEnd; ++colIt)
                {
                    tempForces[localHapticConstraintAllFrames ? 0 : colIt.index()] += colIt.val() * cp->getF()[rowIt.index()];
                }
            }
        }

        for(unsigned int i = 0; i < stateSize; ++i)
        {
            forces[i] = tempForces[i] * forceCoef.getValue();
        }
    }
}

template <class DataTypes>
void LCPForceFeedback<DataTypes>::computeRealForceForPINNFeedback(const VecCoord& state, VecDeriv& forces)
{
    // Reads m_realForceCache — computed fresh every doComputeForce call (see there),
    // matching training's access pattern (read a cache kept up to date by whatever
    // drives the device) without the self-referential contamination ft.force has
    // once usePINN is on (ft.force becomes PINN's own cached output in that case).
    // Confirmed by direct measurement: reading ft.force tracked ground truth closely
    // for ~100 rows then diverged sharply as the feedback loop compounded.
    forces.resize(state.size());
    for (auto& f : forces) f.clear();
    if constexpr (std::is_same_v<DataTypes, sofa::defaulttype::Rigid3Types>)
    {
        if (!forces.empty())
            sofa::defaulttype::getVCenter(forces[0]) = m_realForceCache;
    }
}


template <typename DataTypes>
void LCPForceFeedback<DataTypes>::handleEvent(sofa::core::objectmodel::Event *event)
{
    if (!sofa::simulation::AnimateEndEvent::checkEventType(event))
        return;

    if (!constraintSolver)
        return;

    if (!mState)
        return;

    sofa::component::constraint::lagrangian::solver::ConstraintProblem* new_cp = constraintSolver->getConstraintProblem();

    if (!new_cp)
        return;

    // Find available buffer

    unsigned char buf_index=0;
    const unsigned char cbuf_index=mCurBufferId;
    const unsigned char nbuf_index=mNextBufferId;

    if (buf_index == cbuf_index || buf_index == nbuf_index)
    {
        buf_index++;
        if (buf_index == cbuf_index || buf_index == nbuf_index)
            buf_index++;
    }

    // Compute constraints, id_buf lcp and val for the current lcp.

    MatrixDeriv& constraints = mConstraints[buf_index];

    //	std::vector<int>& id_buf = mId_buf[buf_index];
    VecCoord& val = mVal[buf_index];

    // Update LCP
    mCP[buf_index] = new_cp;

    // Update Val
    val = mState->read(sofa::core::VecCoordId::freePosition())->getValue();

    // Update constraints and id_buf
    constraints.clear();
    //	id_buf.clear();

    const MatrixDeriv& c = mState->read(core::ConstMatrixDerivId::constraintJacobian())->getValue()   ;

    MatrixDerivRowConstIterator rowItEnd = c.end();

    for (MatrixDerivRowConstIterator rowIt = c.begin(); rowIt != rowItEnd; ++rowIt)
    {
        constraints.addLine(rowIt.index(), rowIt.row());
    }

    // make sure the MatrixDeriv has been compressed
    constraints.compress();

    // valid buffer

    {
        // TODO: Lock and/or memory barrier HERE
        mNextBufferId = buf_index;
    }

    // Lock lcp to prevent its use by the SOFA thread while it is used by haptic thread
    if(mIsCuBufferInUse)
        constraintSolver->lockConstraintProblem(this, mCP[mCurBufferId], mCP[mNextBufferId]);
    else
        constraintSolver->lockConstraintProblem(this, mCP[mNextBufferId]);

    // ── PINN: read FEM result and push to predictor buffer ───────────────────
    if (m_usePINN && m_pinn && m_liverDofs)
    {
        static int s_hev = 0;
        ++s_hev;
        const bool log = (s_hev <= 3 || s_hev % 500 == 0 || s_hev >= 3000);

        const auto& curPos  = m_liverDofs->read(sofa::core::ConstVecCoordId::position())->getValue();
        const auto& freePos = m_liverDofs->read(sofa::core::ConstVecCoordId::freePosition())->getValue();
        const auto& restPos = m_liverDofs->read(sofa::core::ConstVecCoordId::restPosition())->getValue();
        int nv = (int)curPos.size();

        if (log) std::cerr << "[PINN] handleEvent #" << s_hev
                           << " A: nv=" << nv
                           << " prevDeform=" << m_prevDeform.size()
                           << " freePos=" << freePos.size()
                           << " restPos=" << restPos.size() << std::endl;

        // Safety: if sizes don't match init, skip to avoid heap corruption
        if (nv != (int)m_prevDeform.size() || nv != (int)restPos.size())
        {
            std::cerr << "[PINN] handleEvent #" << s_hev
                      << " SIZE MISMATCH — skipping (nv=" << nv
                      << " prevDeform=" << m_prevDeform.size()
                      << " restPos=" << restPos.size() << ")" << std::endl;
            return;
        }

        // ── GATE THE WHOLE BLOCK, not just predictForce ─────────────────────────
        // DataCollector's EMA (m_accStress) and deform-delta (m_prevDeform) updates
        // were gated by collectEvery=3 during training — only recomputed once every
        // 3 sim steps. This entire block was running unconditionally every single
        // step instead, so the EMA decayed 3x faster (alpha applied 3x per training
        // "tick") and the deform delta was computed against 1-step-old data instead
        // of 3-step-old — both producing systematically different magnitudes than
        // anything the model saw in training. Gate everything here, once, so ddx/
        // sax/strain/updateFEM/predictForce all advance at the exact training rate.
        ++m_pinnStepCounter;
        // Contact-onset fast-path: if the tool just entered contact (real force went from
        // zero to nonzero), force a PINN call immediately rather than waiting up to 2 more
        // steps. Without this, the device gives ~0 resistance for up to 15ms at every
        // contact onset, the tool plunges in freely, and FEM logs a phantom 14-17N spike.
        // At the moment of first touch, deformation is near-zero regardless of delta window,
        // so the 1-step deform-delta here is equivalent to the normal 3-step delta.
        {
            bool in_contact = (m_realForceCache.norm() > 0.05);
            bool contact_onset = in_contact && !m_prevInContact;
            // Debounce contact loss: LCP force briefly dips below threshold mid-contact
            // (~23/36 gaps were <3 PINN calls). Require 2 consecutive no-contact calls
            // before zeroing the cache — eliminates false loss→onset cache-jump jerks.
            if (in_contact) {
                m_noContactCount = 0;
            } else {
                ++m_noContactCount;
            }
            bool contact_loss = !in_contact && m_prevInContact && (m_noContactCount >= 4);
            m_prevInContact = in_contact;
            if (contact_onset)
                m_pinnStepCounter = 0; // restart stride so next regular call is in 3 steps
            if (contact_loss) {
                std::lock_guard<std::mutex> lk(m_pinnCacheMutex);
                m_pinnCachedForce = {0.0, 0.0, 0.0};
            }
            if (!contact_onset && (m_pinnStepCounter % PINN_CALL_STEP_STRIDE) != 0)
                return;
        }

        // Delta deformation: (curPos-restPos) - prevDeform
        // MAX_DEFORM_MM: maximum plausible liver deformation (~30mm). Anything beyond
        // signals simulation divergence — clamp to 0 so PINN buffer stays clean.
        static constexpr double MAX_DEFORM_MM = 30.0;
        std::vector<float> ddx(nv), ddy(nv), ddz(nv);
        for (int i = 0; i < nv; ++i)
        {
            double cx = curPos[i][0] - restPos[i][0];
            double cy = curPos[i][1] - restPos[i][1];
            double cz = curPos[i][2] - restPos[i][2];
            if (!std::isfinite(cx) || !std::isfinite(cy) || !std::isfinite(cz) ||
                std::abs(cx) > MAX_DEFORM_MM || std::abs(cy) > MAX_DEFORM_MM || std::abs(cz) > MAX_DEFORM_MM)
            {
                ddx[i] = 0.f; ddy[i] = 0.f; ddz[i] = 0.f;
                continue; // don't update m_prevDeform — keep last valid value
            }
            ddx[i] = (float)(cx - m_prevDeform[i][0]);
            ddy[i] = (float)(cy - m_prevDeform[i][1]);
            ddz[i] = (float)(cz - m_prevDeform[i][2]);
            m_prevDeform[i] = sofa::type::Vec3d(cx, cy, cz);
        }
        if (log) std::cerr << "[PINN] handleEvent #" << s_hev << " B: ddx built" << std::endl;

        // Contact-proxy EMA (sax/say/saz = smoothed freePos - curPos)
        std::vector<float> sax(nv), say(nv), saz(nv);
        for (int i = 0; i < nv && i < (int)freePos.size(); ++i)
        {
            auto proxy = freePos[i] - curPos[i];
            if (std::isfinite(proxy[0]) && std::isfinite(proxy[1]) && std::isfinite(proxy[2]))
            {
                // Reset EMA if previously poisoned by NaN
                if (!std::isfinite(m_accStress[i][0]))
                    m_accStress[i] = sofa::type::Vec3d(proxy[0], proxy[1], proxy[2]);
                else
                {
                    m_accStress[i][0] = m_stressAlpha * m_accStress[i][0] + (1-m_stressAlpha) * proxy[0];
                    m_accStress[i][1] = m_stressAlpha * m_accStress[i][1] + (1-m_stressAlpha) * proxy[1];
                    m_accStress[i][2] = m_stressAlpha * m_accStress[i][2] + (1-m_stressAlpha) * proxy[2];
                }
            }
            else if (!std::isfinite(m_accStress[i][0]))
                m_accStress[i] = sofa::type::Vec3d(0, 0, 0);
            sax[i] = (float)m_accStress[i][0];
            say[i] = (float)m_accStress[i][1];
            saz[i] = (float)m_accStress[i][2];
        }
        if (log) std::cerr << "[PINN] handleEvent #" << s_hev << " C: sax built" << std::endl;

        // Real Hooke's-law strain from TetrahedronFEMForceField
        // Skipped during PrecomputedConstraintCorrection precomputation (extreme D values).
        // Read the FEM's own cached per-tet strain (computed during its addForce()
        // pass this step) — the exact same accessor DataCollector.cpp used to label
        // training data. An earlier version of this code manually re-derived strain
        // via rotation matrices and a hand-rolled D_MAX cutoff; that path was never
        // validated against the cached values and produced numbers the model had
        // never seen, which silently wrecked a third of the input (nb_realstrain).
        std::vector<float> rxx(nv, 0.f), ryy(nv, 0.f), rzz(nv, 0.f);
        if (m_femFF && m_liverTopo)
        {
            const auto& tets = m_liverTopo->getTetrahedra();
            const int nTets = (int)m_femFF->getNumTetra();
            std::vector<int> cnt(nv, 0);
            for (int t = 0; t < nTets && t < (int)tets.size(); ++t)
            {
                const auto& tet    = tets[t];
                bool valid = true;
                for (int i = 0; i < 4; ++i)
                    if (tet[i] >= (unsigned int)nv) { valid = false; break; }
                if (!valid) continue;

                const auto& strain = m_femFF->getLastStrain((unsigned int)t);
                for (int i = 0; i < 4; ++i)
                {
                    rxx[tet[i]] += (float)strain[0];
                    ryy[tet[i]] += (float)strain[1];
                    rzz[tet[i]] += (float)strain[2];
                    cnt[tet[i]]++;
                }
            }
            for (int v = 0; v < nv; ++v)
                if (cnt[v] > 0) { rxx[v] /= cnt[v]; ryy[v] /= cnt[v]; rzz[v] /= cnt[v]; }
        }
        // ── DIAG-2: rxx non-zero means strain is being computed (not all filtered by D_MAX) ──
        {
            float max_rxx = 0.f;
            for (int v = 0; v < nv; ++v) max_rxx = std::max(max_rxx, std::abs(rxx[v]));
            if (log) std::cerr << "[PINN] handleEvent #" << s_hev << " E: strain done"
                               << " rxx[0]=" << rxx[0] << " max|rxx|=" << max_rxx;
            if (max_rxx == 0.f && s_hev > 5)
                std::cerr << "  << ALL ZERO — D_MAX filtering everything or no contact";
            std::cerr << std::endl;
        }

        m_pinn->updateFEM(ddx.data(), ddy.data(), ddz.data(),
                          sax.data(), say.data(), saz.data(),
                          rxx.data(), ryy.data(), rzz.data());
        if (log) std::cerr << "[PINN] handleEvent #" << s_hev << " F: updateFEM done" << std::endl;

        // ── Run PINN inference here on SOFA thread ───────────────────────────
        if constexpr (std::is_same_v<DataTypes, sofa::defaulttype::Rigid3Types>)
        {
            if (mState)
            {
                const auto& toolCoords = mState->read(sofa::core::ConstVecCoordId::position())->getValue();
                const auto& toolVels   = mState->read(sofa::core::ConstVecDerivId::velocity())->getValue();
                if (!toolCoords.empty())
                {
                    const auto& center = toolCoords[0].getCenter();
                    float tx = (float)center[0], ty = (float)center[1], tz = (float)center[2];
                    // Real SOFA-integrated rigid-body velocity (mass + spring + damping),
                    // not a manual position finite-difference — training's tool_vx/vy/vz
                    // came from the same MechanicalState velocity read in DataCollector.cpp,
                    // never from differencing positions. A spring-coupled body's actual
                    // velocity can differ substantially from naive position-derivative,
                    // especially right after the coupling spring pulls it toward a new target.
                    float realTvx = 0.f, realTvy = 0.f, realTvz = 0.f;
                    if (!toolVels.empty())
                    {
                        const auto& vel = sofa::defaulttype::getVCenter(toolVels[0]);
                        realTvx = (float)vel[0]; realTvy = (float)vel[1]; realTvz = (float)vel[2];
                    }

                    if (!std::isfinite(tx) || !std::isfinite(ty) || !std::isfinite(tz))
                    {
                        // ALWAYS log NaN tool pos — exceptional event
                        std::cerr << "[PINN-GUARD] handleEvent #" << s_hev
                                  << " NaN/inf tool pos tx=" << tx << " ty=" << ty
                                  << " tz=" << tz << " — skipping" << std::endl;
                        if (log) std::cerr << "[PINN] handleEvent #" << s_hev << " DONE" << std::endl;
                        return;
                    }

                    // ── DIAG-3: ty distribution check (training mean=4.23, std=1.36) ──
                    // Normalized ty should be within ±4 for valid predictions.
                    // Values beyond ±6 are far out-of-distribution → likely NaN output.
                    {
                        float ty_norm = (ty - 4.2251f) / 1.3605f;
                        if (log) std::cerr << "[PINN] handleEvent #" << s_hev
                                           << " G: tx=" << tx << " ty=" << ty
                                           << " tz=" << tz
                                           << " ty_norm=" << ty_norm;
                        if (std::abs(ty_norm) > 5.f)
                            std::cerr << "  << OUT-OF-DIST (|ty_norm|=" << std::abs(ty_norm) << ">5) — expect NaN";
                        std::cerr << std::endl;
                    }

                    float min_dist = 1e9f;
                    for (const auto& v : m_liverRestPos)
                    {
                        float dx = (float)(v[0]-tx), dy = (float)(v[1]-ty), dz = (float)(v[2]-tz);
                        float d = std::sqrt(dx*dx + dy*dy + dz*dz);
                        if (d < min_dist) min_dist = d;
                    }

                    if (!m_pinnTimerSet)
                    {
                        m_pinnStartWallTime = std::chrono::high_resolution_clock::now();
                        m_pinnTimerSet  = true;
                    }
                    double elapsed = std::chrono::duration<double>(
                        std::chrono::high_resolution_clock::now() - m_pinnStartWallTime).count();

                    // Gating already happened once, at the top of this PINN block (we returned
                    // early if not due) — this whole section, including the FEM/EMA reads above,
                    // only ever runs on the correct every-3rd-step cadence now.
                    {
                        // ── Feed the predictor REAL force, not its own last guess ──────────
                        // Training (and every Python-side validation, including FEM_SKIP
                        // robustness tests) always used the true measured force for this
                        // input slot, never a self-prediction. m_pinnCachedForce is the
                        // PINN's own prior output — using it here would create a closed
                        // feedback loop with no anchor to reality, compounding error
                        // indefinitely. Run the real LCP solve (same computation that
                        // produced training's ground-truth labels) instead.
                        VecCoord realState; realState.resize(1);
                        realState[0].getCenter() = sofa::type::Vec3(tx, ty, tz);
                        VecDeriv realForces;
                        this->computeRealForceForPINNFeedback(realState, realForces);
                        float pfx = 0.f, pfy = 0.f, pfz = 0.f;
                        if (!realForces.empty())
                        {
                            const auto& fc = sofa::defaulttype::getVCenter(realForces[0]);
                            pfx = (float)fc[0]; pfy = (float)fc[1]; pfz = (float)fc[2];
                        }
                        // Real velocity (read above from mState), clamped to the same range
                        // training's actual recorded velocities fell within (p1/p99: -20.09,
                        // +20.83) — guards against the rare case of a transient spike in the
                        // real integrated velocity pushing inputs out of the trained distribution.
                        auto clamp20 = [](float v){ return std::max(-20.f, std::min(20.f, v)); };
                        float tvx = clamp20(realTvx), tvy = clamp20(realTvy), tvz = clamp20(realTvz);

                        if (log) std::cerr << "[PINN] handleEvent #" << s_hev
                                           << " G: predictForce tx=" << tx << " ty=" << ty << " tz=" << tz
                                           << " tvx=" << tvx << " tvy=" << tvy << " tvz=" << tvz
                                           << " elapsed=" << elapsed << "s" << std::endl;

                        auto f = m_pinn->predictForce(tx, ty, tz, tvx, tvy, tvz,
                                                       pfx, pfy, pfz, min_dist, elapsed);

                        // ── DIAG-4: always log NaN output ────────────────────────────────────
                        bool f_nan = !std::isfinite(f[0]) || !std::isfinite(f[1]) || !std::isfinite(f[2]);
                        if (log || f_nan)
                            std::cerr << "[PINN] handleEvent #" << s_hev
                                      << " H: predictForce returned ["
                                      << f[0] << "," << f[1] << "," << f[2] << "]"
                                      << (f_nan ? "  << NAN — check ty_norm above" : "") << std::endl;

                        std::lock_guard<std::mutex> lk(m_pinnCacheMutex);
                        // EMA smoothing — blends new prediction with previous cached value
                        // to eliminate the staircase/jerk from stride-3 updates.
                        // Scale factor: 1.0 = no scaling (model predicts correctly at ~7% error).
                        constexpr double scale = 1.0;
                        constexpr double alpha = 0.20;
                        m_pinnCachedForce[0] = alpha * (scale * f[0]) + (1.0 - alpha) * m_pinnCachedForce[0];
                        m_pinnCachedForce[1] = alpha * (scale * f[1]) + (1.0 - alpha) * m_pinnCachedForce[1];
                        m_pinnCachedForce[2] = alpha * (scale * f[2]) + (1.0 - alpha) * m_pinnCachedForce[2];

                        // ── Live comparison log: PINN's prediction vs the real force at
                        // this exact instant, both from genuine continuous real-time data
                        // (no replay involved) — see d_liveComparisonLog comment.
                        const std::string& cmpPath = d_liveComparisonLog.getValue();
                        if (!cmpPath.empty())
                        {
                            static std::ofstream cmpLog;
                            static bool cmpLogOpened = false;
                            if (!cmpLogOpened)
                            {
                                cmpLog.open(cmpPath);
                                cmpLog << "elapsed,real_fx,real_fy,real_fz,pinn_fx,pinn_fy,pinn_fz\n";
                                cmpLogOpened = true;
                            }
                            cmpLog << elapsed << "," << pfx << "," << pfy << "," << pfz << ","
                                   << f[0] << "," << f[1] << "," << f[2] << "\n";
                            cmpLog.flush();
                        }
                    }
                }
            }
        }
        if (log) std::cerr << "[PINN] handleEvent #" << s_hev << " DONE" << std::endl;
    }
}


//
// Those functions are here for compatibility with the Forcefeedback scheme
//

template <>
void LCPForceFeedback< sofa::defaulttype::Rigid3Types >::computeForce(SReal x, SReal y, SReal z, SReal, SReal, SReal, SReal, SReal& fx, SReal& fy, SReal& fz)
{
    fx = fy = fz = 0.0;
    if (!this->d_activate.getValue()) return;

    sofa::defaulttype::Rigid3Types::VecCoord state;
    sofa::defaulttype::Rigid3Types::VecDeriv forces;

    state.resize(1);
    state[0].getCenter() = sofa::type::Vec3(x, y, z);
    computeForce(state, forces);

    if (forces.empty()) return;

    fx = getVCenter(forces[0]).x();
    fy = getVCenter(forces[0]).y();
    fz = getVCenter(forces[0]).z();
    
    this->ft.force = sofa::type::Vec3d(fx, fy, fz);
    this->ft.real_time = std::chrono::high_resolution_clock::now();
    this->ft.sim_time = this->getContext()->getTime();
}

template <>
void LCPForceFeedback< sofa::defaulttype::Rigid3Types >::computeWrench(const sofa::defaulttype::SolidTypes<SReal>::Transform &world_H_tool,
        const sofa::defaulttype::SolidTypes<SReal>::SpatialVector &/*V_tool_world*/,
        sofa::defaulttype::SolidTypes<SReal>::SpatialVector &W_tool_world )
{
    if (!this->d_activate.getValue())
    {
        return;
    }

    sofa::defaulttype::Rigid3Types::VecCoord state;
    sofa::defaulttype::Rigid3Types::VecDeriv forces;
    state.resize(1);
    state[0].getCenter()      = world_H_tool.getOrigin();
    state[0].getOrientation() = world_H_tool.getOrientation();

    computeForce(state, forces);

    W_tool_world.setForce(getVCenter(forces[0]));
    W_tool_world.setTorque(getVOrientation(forces[0]));

    double fx = getVCenter(forces[0]).x();
    double fy = getVCenter(forces[0]).y();
    double fz = getVCenter(forces[0]).z();
    
    this->ft.force = sofa::type::Vec3d(fx, fy, fz);
    this->ft.real_time = std::chrono::high_resolution_clock::now();
    this->ft.sim_time = this->getContext()->getTime();
    
    this->currentForce = this->ft.force;
}



} // namespace sofa::component::haptics
