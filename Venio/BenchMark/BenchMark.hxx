/*
This class was created for developers and is intended
to be used by them for performance testing or bug checking.
It is not recommended for use by ordinary users,
due to unexpected errors or bugs in the future.
@drgnbon
*/
#pragma once
#include "ActivationFunction.hxx"
#include "LossFunction.hxx"
#include "Optimizer.hxx"

#include "LogisticFunction.hxx"
#include "SquareErrorFunction.hxx"
#include "SequentialLayer.hxx"
#include "GD.hxx"
#include "Layer.hxx"
#include "Model.hxx"
#include "Config.hxx"
#include <vector>
#include <windows.h>
#include <psapi.h>


class BenchMark
{
public:
    static void benchSequentialLayer();
};