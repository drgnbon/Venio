cmake_minimum_required(VERSION 3.18)

project(Venio CUDA CXX) # Название проекта

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(Venio_headers_dir "${CMAKE_CURRENT_SOURCE_DIR}/Venio")
set(Venio_sources_dir "${CMAKE_CURRENT_SOURCE_DIR}/Venio")

# Найти и настроить CUDA
find_package(CUDAToolkit REQUIRED)

set(CUDA_SOURCES
  Venio/CudaKernel/Kernel.hxx
  Venio/CudaKernel/Kernel.cu
)

add_library(Kernel STATIC ${CUDA_SOURCES})

# Добавление include директории CUDA
target_include_directories(Kernel PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Venio/CudaKernel>"
  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
  ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(
  Kernel
  PUBLIC Eigen3::Eigen
  PRIVATE CUDA::cudart CUDA::cublas
)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

file(GLOB_RECURSE INCLUDES "${Venio_headers_dir}/*.hxx")
file(GLOB_RECURSE SOURCES "${Venio_sources_dir}/*.cxx")

add_library(Venio ${INCLUDES} ${SOURCES})

file(GLOB_RECURSE HEADER_DIRS LIST_DIRECTORIES true "${Venio_headers_dir}/*")
set(ALL_HEADER_DIRS "")
foreach(HEADER_DIR ${HEADER_DIRS})
    if(IS_DIRECTORY ${HEADER_DIR})
        list(APPEND ALL_HEADER_DIRS ${HEADER_DIR})
    endif()
endforeach()

target_include_directories(Venio PUBLIC
    "$<BUILD_INTERFACE:${Venio_headers_dir}>"
    ${ALL_HEADER_DIRS}
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/dependencies/eigen-3.4.0")
target_link_libraries(Venio PUBLIC Eigen3::Eigen)
target_link_libraries(Venio PUBLIC Kernel)

add_executable(Main Main.cxx)
target_link_libraries(Main PRIVATE Venio)
target_link_libraries(Main PUBLIC Kernel)
