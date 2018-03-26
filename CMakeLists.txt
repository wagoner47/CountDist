cmake_minimum_required(VERSION 3.8)
project(CountDistCPP)
set(CMAKE_CXX_STANDARD 11)
# Version
set(CountDistCPP_VERSION_MAJOR 3)
set(CountDistCPP_VERSION_MINOR 1)
# Compiler
set(CMAKE_CXX_COMPILER g++)
# Compiler flags
set(CMAKE_CXX_FLAGS "-Wall -g -std=c++11")
set(CMAKE_EXE_LINKER_FLAGS "-lm -l sqlite3 -lstdc++fs")

file(MAKE_DIRECTORY ${CountDistCPP_SOURCE_DIR}/build)
file(MAKE_DIRECTORY ${CountDistCPP_SOURCE_DIR}/build/bin)
set(RUNTIME_OUTPUT_DIRECTORY ${CountDistCPP_SOURCE_DIR}/build)

if (DEFINED ENV{OMP})
    set(omp $ENV{OMP})
else()
    set(omp 0)
endif()
add_definitions(-Domp_num_threads=${omp})

if (omp GREATER 0)
    add_compile_options(-fopenmp)
    add_compile_options(-lgomp)
endif()

subdirs(src)