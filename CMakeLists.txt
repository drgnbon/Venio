cmake_minimum_required(VERSION 3.15)

project(Venio)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()


# Add the Eigen library
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/eigen-3.4.0")

# Create your executable
add_executable(Venio main.cxx)

# Link Eigen library to the executable
target_link_libraries(Venio PRIVATE Eigen3::Eigen)

