#pragma once

#include "ErrorLogger/ErrorLogger.hxx"         //delete later
#include "CPUKernel/CPUKernel.hxx"             //delete later
#include "RandomGenerator/RandomGenerator.hxx" //delete later
#include "Kernel.hxx"                          //delete later

#include "ActivationFunctions/LinearFunction/LinearFunction.hxx"
#include "ActivationFunctions/LogisticFunction/LogisticFunction.hxx"
#include "ActivationFunctions/SincFunction/SincFunction.hxx"
#include "ActivationFunctions/SoftSignFunction/SoftSignFunction.hxx"
#include "ActivationFunctions/ReLU/ReLU.hxx"
#include "ActivationFunctions/LReLU/LReLU.hxx"
#include "ActivationFunctions/ArcTg/ArcTg.hxx"
#include "ActivationFunctions/Benti/Benti.hxx"
#include "ActivationFunctions/ELU/ELU.hxx"
#include "ActivationFunctions/GH/GH.hxx"
#include "ActivationFunctions/ISRLU/ISRLU.hxx"
#include "ActivationFunctions/ISRU/ISRU.hxx"
#include "ActivationFunctions/SELU/SELU.hxx"
#include "ActivationFunctions/SiLU/SiLU.hxx"
#include "ActivationFunctions/TH/TH.hxx"
#include "ActivationFunctions/SoftPlus/SoftPlus.hxx"
#include "ActivationFunctions/Sin/Sin.hxx"

#include "Layers/ConvolutionLayer/ConvolutionLayer.hxx"
#include "Layers/SequentialLayer/SequentialLayer.hxx"

#include "LossFunctions/SquareErrorFunction/SquareErrorFunction.hxx"

#include "Model/Model.hxx"

#include "Optimizers/ADAM/ADAM.hxx"
#include "Optimizers/GD/GD.hxx"
#include "Optimizers/Adadelta/Adadelta.hxx"
#include "Optimizers/Adagrad/Adagrad.hxx"
#include "Optimizers/BFGS/BFGS.hxx"
#include "Optimizers/RMSProp/RMSProp.hxx"
