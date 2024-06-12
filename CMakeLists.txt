cmake_minimum_required(VERSION 3.15)

project(Venio)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_EXE_LINKER_FLAGS "-static -static-libgcc")
set(SFML_STATIC_LIBRARIES TRUE)

set(Venio_headers_dir "${CMAKE_CURRENT_SOURCE_DIR}/include/Venio")
set(Venio_sources_dir "${CMAKE_CURRENT_SOURCE_DIR}/src")

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(Venio_public_headers

  ${Venio_headers_dir}/ActivateFunction.hxx
  ${Venio_headers_dir}/ActivateFunctions/Sigmoid.hxx
  ${Venio_headers_dir}/ActivateFunctions/Th.hxx
  ${Venio_headers_dir}/ActivateFunctions/ReLU.hxx
  ${Venio_headers_dir}/ActivateFunctions/Linear.hxx
  ${Venio_headers_dir}/ActivateFunctions/Arctg.hxx
  ${Venio_headers_dir}/ActivateFunctions/Softsign.hxx
  ${Venio_headers_dir}/ActivateFunctions/ISRU.hxx
  ${Venio_headers_dir}/ActivateFunctions/LReLU.hxx
  ${Venio_headers_dir}/ActivateFunctions/ELU.hxx
  ${Venio_headers_dir}/ActivateFunctions/SELU.hxx
  ${Venio_headers_dir}/ActivateFunctions/ISRLU.hxx
  ${Venio_headers_dir}/ActivateFunctions/SoftPlus.hxx
  ${Venio_headers_dir}/ActivateFunctions/Benti.hxx
  ${Venio_headers_dir}/ActivateFunctions/SiLU.hxx
  ${Venio_headers_dir}/ActivateFunctions/Sin.hxx
  ${Venio_headers_dir}/ActivateFunctions/Sinc.hxx
  ${Venio_headers_dir}/ActivateFunctions/Gh.hxx
  ${Venio_headers_dir}/ActivateFunctions/Arctg.hxx

  ${Venio_headers_dir}/Layer.hxx
  ${Venio_headers_dir}/Layers/PerceptronLayer.hxx

  ${Venio_headers_dir}/LossFunction.hxx
  ${Venio_headers_dir}/LossFunctions/SquareError.hxx


  ${Venio_headers_dir}/NeuralNetwork.hxx

  ${Venio_headers_dir}/NeuralNetworks/FunctionalModel.hxx
  ${Venio_headers_dir}/NeuralNetworks/SequentionalModel.hxx

  ${Venio_headers_dir}/Optimizer.hxx
  ${Venio_headers_dir}/Optimizers/ADAM.hxx
  ${Venio_headers_dir}/Optimizers/BFGS.hxx
  ${Venio_headers_dir}/Optimizers/GD.hxx

  ${Venio_headers_dir}/Venio.hxx
  ${Venio_headers_dir}/Logging.hxx

  ${Venio_headers_dir}/Config.hxx
  ${Venio_headers_dir}/TrainerStrategy.hxx
  ${Venio_headers_dir}/Branch.hxx
  ${Venio_headers_dir}/Optimizers/Adagrad.hxx
  ${Venio_headers_dir}/Optimizers/Adadelta.hxx
  ${Venio_headers_dir}/Optimizers/RMSProp.hxx
)

set(Venio_sources


  ${Venio_sources_dir}/Layers/PerceptronLayer.cxx


  ${Venio_sources_dir}/LossFunctions/SquareError.cxx


  ${Venio_sources_dir}/NeuralNetworks/FunctionalModel.cxx
  ${Venio_sources_dir}/NeuralNetworks/SequentionalModel.cxx

  ${Venio_sources_dir}/Optimizers/ADAM.cxx
  ${Venio_sources_dir}/Optimizers/BFGS.cxx
  ${Venio_sources_dir}/Optimizers/GD.cxx

  ${Venio_sources_dir}/NeuralNetwork.cxx
  ${Venio_sources_dir}/Logging.cxx

  ${Venio_sources_dir}/TrainerStrategy.cxx
  ${Venio_sources_dir}/Branch.cxx

  ${Venio_sources_dir}/ActivateFunctions/Sigmoid.cxx
  ${Venio_sources_dir}/ActivateFunctions/Th.cxx
  ${Venio_sources_dir}/ActivateFunctions/ReLU.cxx
  ${Venio_sources_dir}/ActivateFunctions/Linear.cxx
  ${Venio_sources_dir}/ActivateFunctions/Arctg.cxx
  ${Venio_sources_dir}/ActivateFunctions/Softsign.cxx
  ${Venio_sources_dir}/ActivateFunctions/ISRU.cxx
  ${Venio_sources_dir}/ActivateFunctions/LReLU.cxx
  ${Venio_sources_dir}/ActivateFunctions/ELU.cxx
  ${Venio_sources_dir}/ActivateFunctions/SELU.cxx
  ${Venio_sources_dir}/ActivateFunctions/ISRLU.cxx
  ${Venio_sources_dir}/ActivateFunctions/SoftPlus.cxx
  ${Venio_sources_dir}/ActivateFunctions/Benti.cxx
  ${Venio_sources_dir}/ActivateFunctions/SiLU.cxx
  ${Venio_sources_dir}/ActivateFunctions/Sin.cxx
  ${Venio_sources_dir}/ActivateFunctions/Sinc.cxx
  ${Venio_sources_dir}/ActivateFunctions/Gh.cxx
  ${Venio_sources_dir}/ActivateFunctions/Arctg.cxx
  ${Venio_sources_dir}/Optimizer.cxx
  ${Venio_sources_dir}/Optimizers/Adagrad.cxx
  ${Venio_sources_dir}/Optimizers/Adadelta.cxx
  ${Venio_sources_dir}/Optimizers/RMSProp.cxx

)

add_library(Venio ${Venio_public_headers} ${Venio_sources})

target_include_directories(Venio PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/dependencies/eigen-3.4.0")
target_link_libraries(Venio PUBLIC Eigen3::Eigen)


target_link_libraries(Venio PUBLIC sfml-graphics)
set(SFML_DIR "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/SFML-2.6.1/lib/cmake/SFML")
find_package(SFML 2.6 REQUIRED graphics network audio window system)


add_executable(Main Main.cxx)
target_link_libraries(Main PRIVATE Venio)

#target_compile_features(Venio PUBLIC cxx_std_23)
#target_include_directories(Venio PUBLIC
#        "dependencies/eigen-3.4.0/Eigen"
#)
#target_include_directories(Main PRIVATE
#        "dependencies/eigen-3.4.0/Eigen"
#)
