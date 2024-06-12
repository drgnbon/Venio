#ifndef Venio_CONFIG_HXX
#define Venio_CONFIG_HXX


//для Андрея это конфиг для самой либы это подключать ток в network;

//Venio.hxx и этот файл должны быть идентичны на данном этапе, но нельзя путать подключение


#include <Venio/ActivateFunction.hxx>
#include <Venio/LossFunction.hxx>
#include <Venio/Layer.hxx>
#include <Venio/NeuralNetwork.hxx>


#include <Venio/ActivateFunctions/Sigmoid.hxx>
#include <Venio/ActivateFunctions/Th.hxx>
#include <Venio/LossFunctions/SquareError.hxx>
#include <Venio/Layers/PerceptronLayer.hxx>

//typedef Eigen::MatrixXd Matrixd;
//typedef Eigen::MatrixXf Matrixf;
//typedef Eigen::MatrixXi Matrixi;


#endif
