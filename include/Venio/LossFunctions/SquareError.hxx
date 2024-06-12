#ifndef Venio_SQUAREERROR_HXX
#define Venio_SQUAREERROR_HXX

#include <Venio/LossFunction.hxx>

class SquareError : public LossFunction
{
public:
  double getMediumLoss(Matrixd active_value,Matrixd right_answer) override;

  Matrixd getDerivationLoss(Matrixd active_value,Matrixd right_answer) override;
};

#endif
