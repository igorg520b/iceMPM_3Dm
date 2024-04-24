#ifndef PARTICLE_H
#define PARTICLE_H

#include <cmath>
#include <utility>

#include <Eigen/Core>

#include <cuda.h>
#include <cuda_runtime.h>

#include "parameters_sim_3d.h"

struct Point3D
{
    Eigen::Vector3d pos, velocity;
    Eigen::Matrix3d Bp, Fe; // refer to "The Material Point Method for Simulating Continuum Materials"

    double Jp_inv; // track the change in det(Fp)
    short grain;

    double p_tr, q_tr, Je_tr;
    Eigen::Matrix3d U, V;
    Eigen::Vector3d vSigma, vSigmaSquared, v_s_hat_tr;

    long long utility_data;
};


#endif // PARTICLE_H
