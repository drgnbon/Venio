cmake_minimum_required(VERSION 3.15)

project(Venio)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add the Eigen library
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/eigen-3.4.0")

# Create your executable
add_executable(Venio main.cxx)

# Link Eigen library to the executable
target_link_libraries(Venio PRIVATE Eigen3::Eigen)

