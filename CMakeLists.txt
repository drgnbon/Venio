cmake_minimum_required(VERSION 3.18)

project(Venio)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_EXE_LINKER_FLAGS "-static -static-libgcc")

set(Venio_headers_dir "${CMAKE_CURRENT_SOURCE_DIR}/Venio")
set(Venio_sources_dir "${CMAKE_CURRENT_SOURCE_DIR}/Venio")

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(Venio_public_headers

  ${Venio_headers_dir}/Venio.hxx
  ${Venio_headers_dir}/Config.hxx

  ${Venio_headers_dir}/ActivationFunctions/ActivationFunction/ActivationFunction.hxx
  ${Venio_headers_dir}/ActivationFunctions/GhFunction/GhFunction.hxx
  ${Venio_headers_dir}/ActivationFunctions/LinearFunction/LinearFunction.hxx
  ${Venio_headers_dir}/ActivationFunctions/LogisticFunction/LogisticFunction.hxx
  ${Venio_headers_dir}/ActivationFunctions/SincFunction/SincFunction.hxx
  ${Venio_headers_dir}/ActivationFunctions/SoftSignFunction/SoftSignFunction.hxx

  ${Venio_headers_dir}/CPUKernel/CPUKernel.hxx

  ${Venio_headers_dir}/ErrorLogger/ErrorLogger.hxx

  ${Venio_headers_dir}/Layers/Layer/Layer.hxx
  ${Venio_headers_dir}/Layers/SequentialLayer/SequentialLayer.hxx
  ${Venio_headers_dir}/Layers/ConvolutionLayer/ConvolutionLayer.hxx

  ${Venio_headers_dir}/LossFunctions/LossFunction/LossFunction.hxx
  ${Venio_headers_dir}/LossFunctions/SquareErrorFunction/SquareErrorFunction.hxx


  ${Venio_headers_dir}/Model/Model.hxx

  ${Venio_headers_dir}/Optimizers/Optimizer/Optimizer.hxx
  ${Venio_headers_dir}/Optimizers/ADAM/ADAM.hxx
  ${Venio_headers_dir}/Optimizers/GD/GD.hxx

  ${Venio_headers_dir}/RandomGenerator/RandomGenerator.hxx

  

)

set(Venio_sources

${Venio_headers_dir}/ActivationFunctions/ActivationFunction/ActivationFunction.cxx
${Venio_headers_dir}/ActivationFunctions/GhFunction/GhFunction.cxx
${Venio_headers_dir}/ActivationFunctions/LinearFunction/LinearFunction.cxx
${Venio_headers_dir}/ActivationFunctions/LogisticFunction/LogisticFunction.cxx
${Venio_headers_dir}/ActivationFunctions/SincFunction/SincFunction.cxx
${Venio_headers_dir}/ActivationFunctions/SoftSignFunction/SoftSignFunction.cxx

${Venio_headers_dir}/CPUKernel/CPUKernel.cxx

${Venio_headers_dir}/ErrorLogger/ErrorLogger.cxx

${Venio_headers_dir}/Layers/Layer/Layer.cxx
${Venio_headers_dir}/Layers/SequentialLayer/SequentialLayer.cxx
${Venio_headers_dir}/Layers/ConvolutionLayer/ConvolutionLayer.cxx

${Venio_headers_dir}/LossFunctions/LossFunction/LossFunction.cxx
${Venio_headers_dir}/LossFunctions/SquareErrorFunction/SquareErrorFunction.cxx


${Venio_headers_dir}/Model/Model.cxx

${Venio_headers_dir}/Optimizers/Optimizer/Optimizer.cxx
${Venio_headers_dir}/Optimizers/ADAM/ADAM.cxx
${Venio_headers_dir}/Optimizers/GD/GD.cxx

${Venio_headers_dir}/RandomGenerator/RandomGenerator.cxx


)

add_library(Venio ${Venio_public_headers} ${Venio_sources})

target_include_directories(Venio PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/dependencies/eigen-3.4.0")
target_link_libraries(Venio PUBLIC Eigen3::Eigen)

add_executable(Main Main.cxx)
target_link_libraries(Main PRIVATE Venio)

#target_compile_features(Venio PUBLIC cxx_std_23)
#target_include_directories(Venio PUBLIC
#        "dependencies/eigen-3.4.0/Eigen"
#)
#target_include_directories(Main PRIVATE
#        "dependencies/eigen-3.4.0/Eigen"
#)