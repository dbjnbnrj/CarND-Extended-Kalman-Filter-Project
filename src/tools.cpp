#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse(4);
  rmse << 0,0,0,0;

  if(estimations.size() != ground_truth.size()
			|| estimations.size() == 0){
        cout << "Invalid ground truth size";
        return rmse;
  }

  //accumulate squared residuals
	for(unsigned int i=0; i < estimations.size(); ++i){
		VectorXd residual = estimations[i] - ground_truth[i];
		residual = residual.array()*residual.array();
		rmse += residual;
	}

	rmse = rmse/estimations.size();
	//calculate the squared root
	rmse = rmse.array().sqrt();

	//return the result
	return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  MatrixXd Hj(3,4);

  //recover state parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);


  float px2 = px * px;
  float py2 = py * py;
  float pxpy2 = px2 + py2;
  float pxpy_2 = sqrt(pxpy2);
  float pxpy_3_2 = pxpy_2 * pxpy2;

  if (pxpy2 == 0 || pxpy_2 == 0 || pxpy_3_2 == 0) {
     cout << "Don't divide by zero silly ";
     return  Hj;
  }

  Hj << px /pxpy_2, py /pxpy_2, 0, 0,
        -py / pxpy2, px  / pxpy2, 0, 0,
         py * (vx *py - vy * px) /  pxpy_3_2,
         px * (vy *px - vx * py) /  pxpy_3_2,
         px /pxpy_2, py /pxpy_2;

  return Hj;
}
