#ifndef Venio_LOSSFUNCTION_HXX
#define Venio_LOSSFUNCTION_HXX

#include "Eigen/Core"

typedef Eigen::MatrixXd Matrixd;

class LossFunction
{
public:
    virtual double getMediumLoss(Matrixd active_value,Matrixd right_answer) = 0;
    virtual Matrixd getDerivationLoss(Matrixd active_value,Matrixd right_answer) = 0;
};

#endif
