cmake_minimum_required(VERSION 3.12)
project(PINNDataCollector)

# Find SOFA — point to your build
set(CMAKE_PREFIX_PATH 
    "/home/yogyaahuja/sofa/build/install/lib/cmake"
    ${CMAKE_PREFIX_PATH})

find_package(Sofa.Core REQUIRED)
find_package(Sofa.Simulation.Core REQUIRED)
find_package(Sofa.Component.StateContainer REQUIRED)

add_library(${PROJECT_NAME} SHARED
    DataCollector.cpp
    DataCollector.h
    init.cpp
)

target_link_libraries(${PROJECT_NAME}
    Sofa.Core
    Sofa.Simulation.Core
    Sofa.Component.StateContainer
)

target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)