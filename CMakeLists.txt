cmake_minimum_required(VERSION 3.10)

project(Venio CUDA CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Найти и настроить CUDA
find_package(CUDAToolkit REQUIRED)

# Добавить библиотеку Eigen
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/eigen-3.4.0")

# Добавить исходные файлы CUDA
set(CUDA_SOURCES
  CUDAlib/GPUcalculation.h
  CUDAlib/GPUCalculation.cu
)

# Добавление и настройка библиотеки с CUDA
add_library(CUDAlib STATIC ${CUDA_SOURCES})
target_include_directories(CUDAlib PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/CUDAlib>"
  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
target_link_libraries(
  CUDAlib
    PUBLIC Eigen3::Eigen
    PRIVATE CUDA::cudart CUDA::cublas)

# CMake version detecting and comparing. See also:
# - https://stackoverflow.com/questions/43697154/detect-current-cmake-version-using-cmake
if(${CMAKE_VERSION} VERSION_LESS "3.24.0")
    message(WARNING "Native CUDA Architecture detecting was introduced in CMake 3.24, please update for use this.")
    set_property(TARGET CUDAlib PROPERTY CUDA_ARCHITECTURES OFF)
else()
    # Detecting CUDA acrchitecture native by CMake. See also:
    # - https://stackoverflow.com/a/72332793/17157999
    # - https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html
    set_property(TARGET CUDAlib PROPERTY CUDA_ARCHITECTURES NATIVE)
endif()

# Создать исполняемый файл
add_executable(Venio main.cxx)

# Связать библиотеки с исполняемым файлом
target_link_libraries(Venio PRIVATE CUDAlib)
