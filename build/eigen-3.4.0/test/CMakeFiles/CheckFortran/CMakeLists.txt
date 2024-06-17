cmake_minimum_required(VERSION 3.30.0-rc1)
set(CMAKE_MODULE_PATH "D:/Projects/Venio/eigen-3.4.0/cmake")
project(CheckFortran Fortran)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/result.cmake"
  "set(CMAKE_Fortran_COMPILER \"${CMAKE_Fortran_COMPILER}\")\n"
  
  )