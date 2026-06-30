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
find_package(Sofa.Component.SolidMechanics.FEM.Elastic REQUIRED)
find_package(Sofa.Component.Haptics REQUIRED)

# LCPForceFeedback.h pulls in PINNPredictor.h, which needs torch/script.h directly —
# the main Sofa.Component.Haptics build only exposes its own Torch includes as
# PRIVATE, so they don't propagate here even though we link against that target.
# Without this, the build silently never recompiled against header changes (the
# error only surfaced on a forced rebuild) instead of failing loudly every time.
list(APPEND CMAKE_PREFIX_PATH "/home/yogyaahuja/.local/lib/python3.12/site-packages/torch/share/cmake")
# Same fix as the main build's CMakeLists.txt: LibTorch's CUDA-enabled build
# triggers CMake's CUDA language detection even though this project only ever
# runs on CPU — without a fixed architecture, configuration fails outright.
set(CMAKE_CUDA_ARCHITECTURES "86" CACHE STRING "" FORCE)
find_package(Torch REQUIRED)

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
    ${TORCH_LIBRARIES}
)

target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_include_directories(${PROJECT_NAME} PRIVATE ${TORCH_INCLUDE_DIRS})