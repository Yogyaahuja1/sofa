cmake_minimum_required(VERSION 3.12)
project(PINNDataCollector)

# Find SOFA — relative to this clone's own build/install, not a hardcoded absolute
# path. A hardcoded path here would silently link against a DIFFERENT, possibly
# mismatched SOFA build on machines where the repo isn't cloned to that exact
# location — causing exactly the kind of "undefined reference" linker error that
# headers-vs-library version mismatches produce (confirmed: hit this on a second
# machine with a clone at a different path).
get_filename_component(SOFA_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(CMAKE_PREFIX_PATH
    "${SOFA_REPO_ROOT}/build/install/lib/cmake"
    ${CMAKE_PREFIX_PATH})

find_package(Sofa.Core REQUIRED)
find_package(Sofa.Simulation.Core REQUIRED)
find_package(Sofa.Component.StateContainer REQUIRED)
find_package(Sofa.Component.Haptics REQUIRED)
find_package(Sofa.Component.SolidMechanics.FEM.Elastic REQUIRED)

add_library(${PROJECT_NAME} SHARED
    DataCollector.cpp
    DataCollector.h
    init.cpp
)

target_link_libraries(${PROJECT_NAME}
    Sofa.Core
    Sofa.Simulation.Core
    Sofa.Component.StateContainer
    Sofa.Component.Haptics
    Sofa.Component.SolidMechanics.FEM.Elastic
)

target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)