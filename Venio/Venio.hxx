#pragma once

#include "Config.hxx"

#include "ActivationFunctions/ActivationFunction/ActivationFunction.hxx"
#include "ErrorLogger/ErrorLogger.hxx"
#include "Layers/Layer/Layer.hxx"
#include "CPUKernel/CPUKernel.hxx"
#include "LossFunctions/LossFunction/LossFunction.hxx"
#include "Model/Model.hxx"
#include "Optimizers/Optimizer/Optimizer.hxx"
#include "RandomGenerator/RandomGenerator.hxx"

#include <Eigen\Core>
#include <iostream>
#include <initializer_list>
#include <memory>
#include <utility>
#include <vector>
#include <fstream>
#include <random>
#include <filesystem>
#include <utility>

typedef Eigen::MatrixXd Matrixd;