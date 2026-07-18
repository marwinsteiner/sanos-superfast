cmake_minimum_required(VERSION 3.20)
project(sanos_superfast LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# volfi requires strict FP semantics
if(MSVC)
    add_compile_options(/fp:precise /O2 /arch:AVX2)
else()
    add_compile_options(-O3 -march=native -ffp-contract=off -fno-fast-math)
endif()

# volfi is header-only
add_library(volfi INTERFACE)
target_include_directories(volfi INTERFACE external/volfi/include)

# sanos library
add_library(sanos STATIC
    src/bs_kernel.cpp
    src/qp_solver.cpp
    src/surface.cpp
)
target_include_directories(sanos PUBLIC include)
target_link_libraries(sanos PUBLIC volfi)

# benchmark
add_executable(bench_surface bench/bench_surface.cpp)
target_link_libraries(bench_surface PRIVATE sanos)

# tests
add_executable(test_sanos test/test_main.cpp)
target_link_libraries(test_sanos PRIVATE sanos)
