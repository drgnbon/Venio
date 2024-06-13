#include <Eigen>
#include <iostream>

int main()
{
    Eigen::MatrixXd wqeqwe = Eigen::MatrixXd::Random(4,3);
    Eigen::MatrixXd wqeqwe2 = Eigen::MatrixXd::Random(3,2);
    Eigen::MatrixXd  cock = wqeqwe*wqeqwe2;
    std::cout << cock << "";
}