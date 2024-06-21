cmake_minimum_required(VERSION 3.10)

project(Venio CUDA CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Найти и настроить CUDA
find_package(CUDAToolkit REQUIRED)

# Добавить библиотеку Eigen
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/eigen-3.4.0")

# Создать исполняемый файл
add_executable(Venio main.cxx ${CUDA_SOURCES})

# Добавить исходные файлы CUDA
set(CUDA_SOURCES
        GPUCalculation.cu
)
target_sources(Venio PRIVATE
        ${CUDA_SOURCES}
)

# Установить стандарт CUDA и C++
set_property(TARGET Venio PROPERTY CUDA_STANDARD 11)
set_property(TARGET Venio PROPERTY CXX_STANDARD 11)

# Связать библиотеки с исполняемым файлом
target_link_libraries(Venio PRIVATE Eigen3::Eigen CUDA::cudart)
