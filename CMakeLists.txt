cmake_minimum_required(VERSION 3.10)

project(clCPP LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCLHeaders CONFIG REQUIRED)
find_package(OpenCLHeadersCpp CONFIG REQUIRED)

add_library(clCPP INTERFACE)
target_include_directories(clCPP INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(clCPP INTERFACE OpenCL::Headers INTERFACE OpenCL::HeadersCpp)