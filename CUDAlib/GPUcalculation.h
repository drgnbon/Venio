#ifndef GPUCALCULATION_H
#define GPUCALCULATION_H

#include <Eigen/Dense>

class GPUcalculation {
public:
  GPUcalculation();
  ~GPUcalculation();

  void getMatrixMultiply(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B);
  Eigen::MatrixXd getMatrixTranspose(const Eigen::MatrixXd &A);

private:
  typedef void *internal_handle;
  internal_handle handle;
};

#endif // GPUCALCULATION_H
