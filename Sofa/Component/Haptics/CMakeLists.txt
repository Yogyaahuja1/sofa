cmake_minimum_required(VERSION 3.12)
project(Sofa.Component.Haptics LANGUAGES CXX)

set(SOFACOMPONENTHAPTICS_SOURCE_DIR "src/sofa/component/haptics")

set(HEADER_FILES
    ${SOFACOMPONENTHAPTICS_SOURCE_DIR}/config.h.in
    ${SOFACOMPONENTHAPTICS_SOURCE_DIR}/init.h
    ${SOFACOMPONENTHAPTICS_SOURCE_DIR}/ForceFeedback.h
    ${SOFACOMPONENTHAPTICS_SOURCE_DIR}/LCPForceFeedback.h
    ${SOFACOMPONENTHAPTICS_SOURCE_DIR}/LCPForceFeedback.inl
    ${SOFACOMPONENTHAPTICS_SOURCE_DIR}/MechanicalStateForceFeedback.h
    ${SOFACOMPONENTHAPTICS_SOURCE_DIR}/NullForceFeedback.h
    ${SOFACOMPONENTHAPTICS_SOURCE_DIR}/NullForceFeedbackT.h
)

set(SOURCE_FILES
    ${SOFACOMPONENTHAPTICS_SOURCE_DIR}/init.cpp
    ${SOFACOMPONENTHAPTICS_SOURCE_DIR}/ForceFeedback.cpp
    ${SOFACOMPONENTHAPTICS_SOURCE_DIR}/LCPForceFeedback.cpp
    ${SOFACOMPONENTHAPTICS_SOURCE_DIR}/NullForceFeedback.cpp
    ${SOFACOMPONENTHAPTICS_SOURCE_DIR}/NullForceFeedbackT.cpp
    /home/yogyaahuja/sofa/pinn_project/cpp/PINNPredictor.cpp
)

sofa_find_package(Sofa.Simulation.Core REQUIRED)
sofa_find_package(Sofa.Component.Constraint.Lagrangian.Solver REQUIRED)
sofa_find_package(Sofa.Component.SolidMechanics.FEM.Elastic REQUIRED)

list(APPEND CMAKE_PREFIX_PATH "/home/yogyaahuja/.local/lib/python3.12/site-packages/torch/share/cmake")
set(CMAKE_CUDA_ARCHITECTURES "86" CACHE STRING "" FORCE)
find_package(Torch REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Simulation.Core)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Constraint.Lagrangian.Solver)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.SolidMechanics.FEM.Elastic)
target_link_libraries(${PROJECT_NAME} PUBLIC ${TORCH_LIBRARIES})
target_include_directories(${PROJECT_NAME} PRIVATE ${TORCH_INCLUDE_DIRS})

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)

# Tests
# If SOFA_BUILD_TESTS exists and is OFF, then these tests will be auto-disabled
cmake_dependent_option(SOFA_COMPONENT_HAPTICS_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFA_COMPONENT_HAPTICS_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
