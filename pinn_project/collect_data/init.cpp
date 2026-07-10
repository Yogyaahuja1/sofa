#include "DataCollector.h"
#include "MembraneDataCollector.h"

#include <sofa/core/ObjectFactory.h>

namespace pinn
{

int DataCollectorClass = sofa::core::RegisterObject(
    "Collects training data for PINN")
    .add<DataCollector>();

// MembraneDataCollector registered in its own .cpp

}

