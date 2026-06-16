#define SOFA_COMPONENT_CONTROLLER_LCPFORCEFEEDBACK_CPP

#include <sofa/component/haptics/LCPForceFeedback.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/RigidTypes.h>

namespace sofa::component::haptics
{

int lCPForceFeedbackClass = sofa::core::RegisterObject("LCP force feedback for the device")
        .add< LCPForceFeedback<defaulttype::Vec1Types> >()
        .add< LCPForceFeedback<defaulttype::Rigid3Types> >();

template class SOFA_COMPONENT_HAPTICS_API LCPForceFeedback<defaulttype::Vec1Types>;
template class SOFA_COMPONENT_HAPTICS_API LCPForceFeedback<defaulttype::Rigid3Types>;

} // namespace sofa::component::haptics
