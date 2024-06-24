#pragma once

#include "ActivationFunctions/LinearFunction/LinearFunction.hxx"
#include "ActivationFunctions/LogisticFunction/LogisticFunction.hxx"
#include "ActivationFunctions/SincFunction/SincFunction.hxx"
#include "ActivationFunctions/SoftSignFunction/SoftSignFunction.hxx"

#include "ErrorLogger/ErrorLogger.hxx" //delete later

#include "Layers/ConvolutionLayer/ConvolutionLayer.hxx"
#include "Layers/SequentialLayer/SequentialLayer.hxx"

#include "CPUKernel/CPUKernel.hxx" //delete later

#include "LossFunctions/SquareErrorFunction/SquareErrorFunction.hxx"

#include "Model/Model.hxx"

#include "Optimizers/ADAM/ADAM.hxx"
#include "Optimizers/GD/GD.hxx"

#include "RandomGenerator/RandomGenerator.hxx" //delete later

// #include "Config.hxx"

// #include <Eigen\Core>
// #include <iostream>
// #include <initializer_list>
// #include <memory>
// #include <utility>
// #include <vector>
// #include <fstream>
// #include <random>
// #include <filesystem>
// #include <utility>
