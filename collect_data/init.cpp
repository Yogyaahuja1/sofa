#include "DataCollector.h"

#include <sofa/core/ObjectFactory.h>

namespace pinn
{

int DataCollectorClass = sofa::core::RegisterObject(
    "Collects training data for PINN")
    .add<DataCollector>();

}

