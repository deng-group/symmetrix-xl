cmake_minimum_required(VERSION 3.27)

# The Metal runtime layer is configurable on its own so that it can be built
# and qualified without Kokkos, BLAS, or the Python extension.
if(PROJECT_IS_TOP_LEVEL OR NOT DEFINED PROJECT_NAME)
    project(SymmetrixMetal LANGUAGES CXX OBJCXX)
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
else()
    enable_language(OBJCXX)
endif()

if(NOT APPLE)
    message(FATAL_ERROR "The Symmetrix Metal runtime requires an Apple platform.")
endif()

option(SYMMETRIX_METAL_SELFTEST "Build the Metal runtime self-test executable." ON)

add_library(symmetrix_metal STATIC
    metal_runtime.mm metal_m0_module.cpp metal_r0_module.cpp metal_r1_module.cpp)
set_target_properties(symmetrix_metal PROPERTIES
    OBJCXX_STANDARD 20
    OBJCXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)
target_compile_options(symmetrix_metal PRIVATE -fobjc-arc)
target_include_directories(symmetrix_metal
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}"
           "${CMAKE_CURRENT_SOURCE_DIR}/../../../symmetrix/source/symmetrix")
target_link_libraries(symmetrix_metal PUBLIC
    "-framework Metal"
    "-framework Foundation")
target_compile_definitions(symmetrix_metal PUBLIC SYMMETRIX_ENABLE_METAL)

if(SYMMETRIX_METAL_SELFTEST)
    add_executable(symmetrix_metal_selftest metal_runtime_selftest.cpp)
    target_link_libraries(symmetrix_metal_selftest PRIVATE symmetrix_metal)

    add_executable(symmetrix_metal_cpu_microbenchmark
        "${CMAKE_CURRENT_SOURCE_DIR}/../../../benchmarks/metal_cpu_microbenchmark.cpp")
    target_compile_options(symmetrix_metal_cpu_microbenchmark PRIVATE
        -O3 -mcpu=native)
    target_compile_definitions(symmetrix_metal_cpu_microbenchmark PRIVATE
        ACCELERATE_NEW_LAPACK)
    target_link_libraries(symmetrix_metal_cpu_microbenchmark PRIVATE
        symmetrix_metal "-framework Accelerate")
endif()
