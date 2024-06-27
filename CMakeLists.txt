cmake_minimum_required(VERSION 3.18) 

project(Shell) # Название проекта

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/eigen-3.4.0")

add_executable(Shell Shell.cxx)
target_link_libraries(Shell PUBLIC Eigen3::Eigen)

   




