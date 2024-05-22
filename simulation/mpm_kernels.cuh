#ifndef MPM_KERNELS_CUH
#define MPM_KERNELS_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include "point_3d.h"

using namespace Eigen;

//constexpr double d = 2; // dimensions
constexpr double coeff1 = 1.224744871391589; // sqrt((6-d)/2.);
constexpr double coeff1_inv = 0.8164965809277260;
constexpr long long status_crushed = 0x10000;
constexpr long long status_disabled = 0x20000;

__device__ uint8_t gpu_error_indicator;
__constant__ SimParams3D gprms;

__global__ void partition_kernel_p2g(const int gridX, const int gridX_offset, const int pitch_grid,
                                     const int count_pts, const int pitch_pts,
                                     const double *buffer_pts, double *buffer_grid)
{
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(pt_idx >= count_pts) return;

    const long long* ptr = reinterpret_cast<const long long*>(&buffer_pts[pitch_pts*SimParams3D::idx_utility_data]);
    long long utility_data = ptr[pt_idx];
    if(utility_data & status_disabled) return; // point is disabled

    const double &h = gprms.cellsize;
    const double &h_inv = gprms.cellsize_inv;
    const double &particle_mass = gprms.ParticleMass;
    const int &gridY = gprms.GridY;
    const int &gridZ = gprms.GridZ;
    const int &halo = gprms.GridHaloSize;
    const int &offset = gprms.gbOffset;

    // pull point data from SOA
    Vector3d pos, velocity;
    Matrix3d Bp, Fe;

    for(int i=0; i<SimParams3D::dim; i++)
    {
        pos[i] = buffer_pts[pt_idx + pitch_pts*(SimParams3D::posx+i)];
        velocity[i] = buffer_pts[pt_idx + pitch_pts*(SimParams3D::velx+i)];
        for(int j=0; j<SimParams3D::dim; j++)
        {
            Fe(i,j) = buffer_pts[pt_idx + pitch_pts*(SimParams3D::Fe00 + i*SimParams3D::dim + j)];
            Bp(i,j) = buffer_pts[pt_idx + pitch_pts*(SimParams3D::Bp00 + i*SimParams3D::dim + j)];
        }
    }

    Matrix3d PFt = KirchhoffStress_Wolper(Fe);
    Matrix3d subterm2 = particle_mass*Bp - (gprms.dt_vol_Dpinv)*PFt;

    Eigen::Vector3i base_coord_i = (pos*h_inv - Vector3d::Constant(0.5)).cast<int>(); // coords of base grid node for point
    Vector3d base_coord = base_coord_i.cast<double>();
    Vector3d fx = pos*h_inv - base_coord;

    // check if a point is out of boundaries
    if(base_coord.x()- gridX_offset < (-halo)) gpu_error_indicator = 70;
    if(base_coord.y()<0) gpu_error_indicator = 71;
    if(base_coord.z()<0) gpu_error_indicator = 72;
    if(base_coord.x()- gridX_offset > (gridX+halo-3)) gpu_error_indicator = 73;
    if(base_coord.y()>gridY-3) gpu_error_indicator = 74;
    if(base_coord.z()>gridZ-3) gpu_error_indicator = 75;

    // optimized method of computing the quadratic (!) weight function (no conditional operators)
    Array3d arr_v0 = 1.5-fx.array();
    Array3d arr_v1 = fx.array() - 1.0;
    Array3d arr_v2 = fx.array() - 0.5;
    Array3d ww[3] = {0.5*arr_v0*arr_v0, 0.75-arr_v1*arr_v1, 0.5*arr_v2*arr_v2};

    for (int i=0; i<3; i++)
        for (int j=0; j<3; j++)
            for (int k=0; k<3; k++)
            {
                double Wip = ww[i][0]*ww[j][1]*ww[k][2];
                Vector3d dpos((i-fx[0])*h, (j-fx[1])*h, (k-fx[2])*h);
                Vector3d incV = Wip*(velocity*particle_mass + subterm2*dpos);
                double incM = Wip*particle_mass;

                int idx_gridnode = (i+base_coord_i[0]-gridX_offset)*(gridY*gridZ) +
                                   (j+base_coord_i[1])*gridZ + (k+base_coord_i[2]);

                // Udpate mass, velocity and force
                atomicAdd(&buffer_grid[0*pitch_grid + idx_gridnode + offset], incM);
                for(int m=0; m<SimParams3D::dim; m++)
                    atomicAdd(&buffer_grid[(m+1)*pitch_grid + idx_gridnode + offset], incV[m]);
            }
}


__global__ void partition_kernel_receive_halos_left(const int haloElementCount, const int gridX_partition,
                                               const int pitch_grid, double *buffer_grid,
                                               const double *halo0, const double *halo1)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= haloElementCount) return;

//    const int &gridY = gprms.GridY;
//    const int &gridZ = gprms.GridZ;

    for(int i=0; i<SimParams3D::nGridArrays; i++)
    {
        buffer_grid[idx + i*pitch_grid] += halo0[idx + i*pitch_grid];
//        buffer_grid[idx + i*pitch_grid + gridZ*gridY*gridX_partition] += halo1[idx + i*pitch_grid];
    }
}

__global__ void partition_kernel_receive_halos_right(const int haloElementCount, const int gridX_partition,
                                               const int pitch_grid, double *buffer_grid,
                                               const double *halo0, const double *halo1)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= haloElementCount) return;

    const int &gridY = gprms.GridY;
    const int &gridZ = gprms.GridZ;

    for(int i=0; i<SimParams3D::nGridArrays; i++)
    {
//        buffer_grid[idx + i*pitch_grid] += halo0[idx + i*pitch_grid];
        buffer_grid[idx + i*pitch_grid + gridZ*gridY*gridX_partition] += halo1[idx + i*pitch_grid];
    }
}


__global__ void partition_kernel_update_nodes(const Eigen::Vector2d indCenter,
                                              const int nNodes, const int gridX_offset, const int pitch_grid,
                                              double *buffer_grid, double *indenter_force_accumulator)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= nNodes) return;

    const int &halo = gprms.GridHaloSize;
    const int &gridY = gprms.GridY;
    const int &gridZ = gprms.GridZ;
    const int &gridXTotal = gprms.GridXTotal;
    const double &indRsq = gprms.IndRSq;
    const double &dt = gprms.InitialTimeStep;
    const double &cellsize = gprms.cellsize;
    const double &vmax = gprms.vmax;
    const double &vmax_squared = gprms.vmax_squared;
    const Vector3d vco(gprms.IndVelocity,0,0);  // velocity of the collision object (indenter)

    double mass = buffer_grid[idx];
    if(mass == 0) return;

    Vector3d velocity;
    for(int i=0;i<SimParams3D::dim;i++)
        velocity[i] = buffer_grid[(i+1)*pitch_grid + idx];

    int idx_z = idx % gridZ;
    int idx_y = (idx / gridZ) % gridY;
    int idx_x = idx / (gridZ*gridY);
    Vector3i gi(idx_x+gridX_offset-halo, idx_y, idx_z);   // integer x-y index of the grid node
    Vector3d gnpos = gi.cast<double>()*cellsize;    // position of the grid node in the whole grid

    velocity /= mass;
    velocity[1] -= gprms.dt_Gravity;
    if(velocity.squaredNorm() > vmax_squared) velocity = velocity.normalized()*vmax;


    // indenter
    Vector2d gnpos2d(gnpos.x(), gnpos.y());    // position of the grid node
    Vector2d n = gnpos2d - indCenter;    // vector pointing at the node from indenter's center

    if(n.squaredNorm() < indRsq)
    {
        // grid node is inside the indenter
        Vector3d vrel = velocity - vco;
        n.normalize();
        Vector3d n3d(n[0],n[1],0);
        double vn = vrel.dot(n3d);   // normal component of the velocity
        if(vn < 0)
        {
            Vector3d vt = vrel - n3d*vn;   // tangential portion of relative velocity
            Vector3d prev_velocity = velocity;
            velocity = vco + vt;

            // force on the indenter
            Vector3d force = (prev_velocity-velocity)*mass/dt;
            float angle = atan2f((float)n[0],(float)n[1]);
            angle += SimParams3D::pi;
            angle *= gprms.IndenterSubdivisions/(2*SimParams3D::pi);

            int index_angle = min(max((int)angle, 0), gprms.IndenterSubdivisions-1);
            int index_z = min(max(idx_z,0),gridZ-1);
            int index = index_z + index_angle*gridZ;

            for(int i=0;i<SimParams3D::dim;i++) atomicAdd(&indenter_force_accumulator[i+SimParams3D::dim*index], force[i]);
        }
    }

    // attached bottom layer
    if(gi.y() <= 2) velocity.setZero();
    else if(gi.y() >= gridY-4 && velocity.y()>0) velocity.y() = 0;
    if(gi.x() <= 2 && velocity.x()<0) velocity.x() = 0;
    else if(gi.x() >= gridXTotal-4 && velocity.x()>0) velocity.x() = 0;
    if(gi.z() <= 2 && velocity.z()<0) velocity.z() = 0;
    else if(gi.z() >= gridZ-4 && velocity.z()>0) velocity.z() = 0;

    // side boundary conditions would go here

    // write the updated grid velocity back to memory
    for(int i=0;i<SimParams3D::dim;i++)
        buffer_grid[(i+1)*pitch_grid + idx] = velocity[i];
}




__global__ void partition_kernel_g2p(const bool recordPQ, const bool enablePointTransfer,
                                     const int gridX, const int gridX_offset, const int pitch_grid,
                                     const int count_pts, const int pitch_pts,
                                     double *buffer_pts, const double *buffer_grid,
                                     int *utility_data,
                                     const int VectorCapacity_transfer,
                                     double *point_buffer_left, double *point_buffer_right)
{

    const int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(pt_idx >= count_pts) return;

    // skip if a point is disabled
    Point3D p;
    long long* ptr = reinterpret_cast<long long*>(&buffer_pts[pt_idx + pitch_pts*SimParams3D::idx_utility_data]);
    p.utility_data = *ptr;
    if(p.utility_data & status_disabled) return; // point is disabled

    const int &halo = gprms.GridHaloSize;
    const double &h_inv = gprms.cellsize_inv;
    const double &dt = gprms.InitialTimeStep;
    const int &gridY = gprms.GridY;
    const int &gridZ = gprms.GridZ;
    const double &mu = gprms.mu;
    const double &kappa = gprms.kappa;
    const int &offset = gprms.gbOffset;

    // pull point data from SOA
    for(int i=0; i<SimParams3D::dim; i++)
    {
        p.pos[i] = buffer_pts[pt_idx + pitch_pts*(SimParams3D::posx+i)];
        for(int j=0; j<SimParams3D::dim; j++)
        {
            p.Fe(i,j) = buffer_pts[pt_idx + pitch_pts*(SimParams3D::Fe00 + i*SimParams3D::dim + j)];
        }
    }
    p.Jp_inv = buffer_pts[pt_idx + pitch_pts*SimParams3D::idx_Jp_inv];
    p.grain = (short)p.utility_data;

    // coords of base grid node for point
    Eigen::Vector3i base_coord_i = (p.pos*h_inv - Vector3d::Constant(0.5)).cast<int>();
    Vector3d base_coord = base_coord_i.cast<double>();
    Vector3d fx = p.pos*h_inv - base_coord;

    // optimized method of computing the quadratic weight function without conditional operators
    Array3d arr_v0 = 1.5 - fx.array();
    Array3d arr_v1 = fx.array() - 1.0;
    Array3d arr_v2 = fx.array() - 0.5;
    Array3d ww[3] = {0.5*arr_v0*arr_v0, 0.75-arr_v1*arr_v1, 0.5*arr_v2*arr_v2};

    p.velocity.setZero();
    p.Bp.setZero();

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++)
        {
            Vector3d dpos = Vector3d(i, j, k) - fx;
            double weight = ww[i][0]*ww[j][1]*ww[k][2];

            int idx_gridnode = (i+base_coord_i[0]-gridX_offset)*(gridY*gridZ) + (j+base_coord_i[1])*gridZ + (k+base_coord_i[2]);

            Vector3d node_velocity;
            for(int m=0;m<SimParams3D::dim;m++)
                node_velocity[m] = buffer_grid[(1+m)*pitch_grid + idx_gridnode + offset];
            p.velocity += weight * node_velocity;
            p.Bp += (4.*h_inv)*weight *(node_velocity*dpos.transpose());
        }

    // Advection and update of the deformation gradient
    p.pos += p.velocity * dt;
    p.Fe = (Matrix3d::Identity() + dt*p.Bp) * p.Fe;     // p.Bp is the gradient of the velocity vector (it seems)

    ComputePQ(p, kappa, mu);    // pre-computes USV, p, q, etc.

    if(!(p.utility_data & status_crushed)) CheckIfPointIsInsideFailureSurface(p);
    if(p.utility_data & status_crushed) Wolper_Drucker_Prager(p);

    // distribute the values of p back into GPU memory: pos, velocity, BP, Fe, Jp_inv, PQ
    for(int i=0; i<SimParams3D::dim; i++)
    {
        buffer_pts[pt_idx + pitch_pts*(SimParams3D::posx+i)] = p.pos[i];
        buffer_pts[pt_idx + pitch_pts*(SimParams3D::velx+i)] = p.velocity[i];
        for(int j=0; j<SimParams3D::dim; j++)
        {
            buffer_pts[pt_idx + pitch_pts*(SimParams3D::Fe00 + i*SimParams3D::dim + j)] = p.Fe(i,j);
            buffer_pts[pt_idx + pitch_pts*(SimParams3D::Bp00 + i*SimParams3D::dim + j)] = p.Bp(i,j);
        }
    }

    buffer_pts[pt_idx + pitch_pts*SimParams3D::idx_Jp_inv] = p.Jp_inv;
    *ptr = p.utility_data; // includes crushed/disable status and grain number

    // at the end of each cycle, PQ are recorded for visualization
    if(recordPQ)
    {
        buffer_pts[pt_idx + pitch_pts*SimParams3D::idx_P] = p.p_tr;
        buffer_pts[pt_idx + pitch_pts*SimParams3D::idx_Q] = p.q_tr;
    }

    // check if a points needs to be transferred to adjacent partition
    int base_coord_x = (int)(p.pos.x()*h_inv - 0.5); // updated after the point has moved


    // only tranfer the points if this feature is enabled this particular step
    constexpr int fly_threshold = 3;
    if(enablePointTransfer)
    {
        const int keep_track_threshold = halo/2-1;
        if((base_coord_x - gridX_offset) < -keep_track_threshold)
        {
            int deviation = -(base_coord_x - gridX_offset);
            atomicMax(&utility_data[GPU_Partition_3D::idx_pts_max_extent], deviation);
        }
        else if(base_coord_x - (gridX_offset+gridX-3) > keep_track_threshold)
        {
            int deviation = base_coord_x - (gridX_offset+gridX-3);
            atomicMax(&utility_data[GPU_Partition_3D::idx_pts_max_extent], deviation);
        }

        if((base_coord_x - gridX_offset) < -fly_threshold)
        {
            // point transfers to the left
            int fly_idx = atomicAdd(&utility_data[GPU_Partition_3D::idx_transfer_to_left], 1);  // reserve buffer index
            if(fly_idx < VectorCapacity_transfer)
            {
                // only perform this procedure if there is space in the buffer
                PreparePointForTransfer(pt_idx, fly_idx, point_buffer_left, pitch_pts, buffer_pts);
                *ptr = status_disabled; // includes crushed/disable status and grain number
            }
            else
                utility_data[GPU_Partition_3D::idx_transfer_to_left] = VectorCapacity_transfer;
        }
        else if(base_coord_x - (gridX_offset+gridX-3) > fly_threshold)
        {
            // point transfers to the right
            int fly_idx = atomicAdd(&utility_data[GPU_Partition_3D::idx_transfer_to_right], 1);  // reserve buffer index
            if(fly_idx < VectorCapacity_transfer)
            {
                PreparePointForTransfer(pt_idx, fly_idx, point_buffer_right, pitch_pts, buffer_pts);
                *ptr = status_disabled; // includes crushed/disable status and grain number
            }
            else
                utility_data[GPU_Partition_3D::idx_transfer_to_right] = VectorCapacity_transfer;
        }
    }
}

__device__ void PreparePointForTransfer(const int pt_idx, const int index_in_transfer_buffer,
                                        double *point_transfer_buffer, const int pitch_pts,
                                        const double *buffer_pts)
{
    // check buffer boundary
    for(int i=0;i<SimParams3D::nPtsArrays;i++)
        point_transfer_buffer[i + index_in_transfer_buffer*SimParams3D::nPtsArrays] = buffer_pts[pt_idx + pitch_pts*i];
}


__global__ void partition_kernel_receive_points(const int count_transfer,
                                                const int count_pts, const int pitch_pts,
                                                double *buffer_pts,
                                                double *transfer_buffer)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= count_transfer) return;

    int idx_in_soa = count_pts + idx;
    if(idx_in_soa >= pitch_pts) { gpu_error_indicator = 5; return; } // no space for incoming points

    // copy point data
    for(int i=0;i<SimParams3D::nPtsArrays;i++)
    {
        buffer_pts[idx_in_soa + i*pitch_pts] = transfer_buffer[i + SimParams3D::nPtsArrays*idx];
    }
}






__device__ void GetParametersForGrain(short grain, double &pmin, double &pmax, double &qmax, double &beta, double &mSq, double &pmin2)
{
    //    double var1 = 1.0 + gprms.GrainVariability*0.05*(-10 + grain%21);
    double var2 = 1.0 + gprms.GrainVariability*0.033*(-15 + (grain+3)%30);
    double var3 = 1.0 + gprms.GrainVariability*0.1*(-10 + (grain+4)%11);

    pmax = gprms.IceCompressiveStrength;// * var1;
    pmin = -gprms.IceTensileStrength;// * var2;
    qmax = gprms.IceShearStrength * var3;
    pmin2 = -gprms.IceTensileStrength2 * var2;

    beta = gprms.NACC_beta;
    //    beta = -pmin / pmax;
    double NACC_M = (2*qmax*sqrt(1+2*beta))/(pmax-pmin);
    mSq = NACC_M*NACC_M;
//    mSq = (4*qmax*qmax*(1+2*beta))/((pmax*(1+beta))*(pmax*(1+beta)));
}


__device__ void CheckIfPointIsInsideFailureSurface(Point3D &p)
{
    double beta, M_sq, pmin, pmax, qmax, pmin2;
    GetParametersForGrain(p.grain, pmin, pmax, qmax, beta, M_sq, pmin2);

    if(p.p_tr<0)
    {
        if(p.p_tr<pmin2) {p.utility_data |= status_crushed; return;}
        double q0 = 2*sqrt(-pmax*pmin)*qmax/(pmax-pmin);
        double k = -q0/pmin2;
        double q_limit = k*(p.p_tr-pmin2);
        if(p.q_tr > q_limit) {p.utility_data |= status_crushed; return;}
    }
    else
    {
        double y = (1.+2.*beta)*p.q_tr*p.q_tr + M_sq*(p.p_tr + beta*pmax)*(p.p_tr - pmax);
        if(y > 0)
        {
            p.utility_data |= status_crushed;
        }
    }
}


__device__ void ComputePQ(Point3D &p, const double &kappa, const double &mu)
{

    svd3x3(p.Fe, p.U, p.vSigma, p.V);
    p.Je_tr = p.vSigma.prod();         // product of elements of vSigma (representation of diagonal matrix)
    p.p_tr = -(kappa/2.) * (p.Je_tr*p.Je_tr - 1.);
    p.vSigmaSquared = p.vSigma.array().square().matrix();
    const double Je_tr_sq = p.Je_tr*p.Je_tr;
    p.v_s_hat_tr = mu*rcbrt(Je_tr_sq) * dev_d(p.vSigmaSquared); //mu * pow(Je_tr,-2./d)* dev(SigmaSquared);
    p.q_tr = coeff1*p.v_s_hat_tr.norm();
}


__device__ void Wolper_Drucker_Prager(Point3D &p)
{
    const double &mu = gprms.mu;
    const double &kappa = gprms.kappa;
    const double &tan_phi = gprms.DP_tan_phi;
    const double &DP_threshold_p = gprms.DP_threshold_p;

    const double &pmax = gprms.IceCompressiveStrength;
    const double &qmax = gprms.IceShearStrength;


    if(p.p_tr < -DP_threshold_p || p.Jp_inv < 1)
    {
        // tear in tension or compress until original state
        double p_new = -DP_threshold_p;
        double Je_new = sqrt(-2.*p_new/kappa + 1.);
        double cbrt_Je_new = cbrt(Je_new);
        Vector3d vSigma_new(cbrt_Je_new,cbrt_Je_new,cbrt_Je_new); // Vector3d::Constant(1.)*pow(Je_new, 1./(double)d);
        p.Fe = p.U*vSigma_new.asDiagonal()*p.V.transpose();
        p.Jp_inv *= Je_new/p.Je_tr;
    }
    else
    {
        double q_n_1;

        if(p.p_tr > pmax)
        {
            q_n_1 = 0;
        }
        else
        {
            double q_from_dp = (p.p_tr+DP_threshold_p)*tan_phi;
            //q_n_1 = min(q_from_dp,qmax);
            const double pmin = -gprms.IceTensileStrength;
            double q_from_failure_surface = 2*sqrt((pmax-p.p_tr)*(p.p_tr-pmin))*qmax/(pmax-pmin);
            q_n_1 = min(q_from_failure_surface, q_from_dp);
        }

        if(p.q_tr >= q_n_1)
        {
            // project onto YS
            double s_hat_n_1_norm = q_n_1*coeff1_inv;
            double Je_tr_sq = p.Je_tr*p.Je_tr;
            //Matrix2d B_hat_E_new = s_hat_n_1_norm*(pow(Je_tr,2./d)/mu)*s_hat_tr.normalized() + Matrix2d::Identity()*(SigmaSquared.trace()/d);
            Vector3d vB_hat_E_new = (s_hat_n_1_norm*cbrt(Je_tr_sq)/mu)*p.v_s_hat_tr.normalized() +
                                    Vector3d::Constant(1.)*(p.vSigmaSquared.sum()/SimParams3D::dim);
            Vector3d vSigma_new = vB_hat_E_new.array().sqrt().matrix();
            p.Fe = p.U*vSigma_new.asDiagonal()*p.V.transpose();
        }
    }
}


__device__ Matrix3d KirchhoffStress_Wolper(const Matrix3d &F)
{
    const double &kappa = gprms.kappa;
    const double &mu = gprms.mu;

    // Kirchhoff stress as per Wolper (2019)
    double Je = F.determinant();
    Matrix3d b = F*F.transpose();
    Matrix3d PFt = mu*pow(Je, -2./SimParams3D::dim)*dev(b) + kappa*0.5*(Je*Je-1.)*Matrix3d::Identity();
    return PFt;
}


// deviatoric part of a diagonal matrix

__device__ Vector3d dev_d(Vector3d Adiag)
{
    return Adiag - Adiag.sum()/3*Vector3d::Constant(1.);
}

__device__ Eigen::Matrix3d dev(Eigen::Matrix3d A)
{
    return A - A.trace()/3*Eigen::Matrix3d::Identity();
}


__device__ void svd3x3(const Eigen::Matrix3d &A, Eigen::Matrix3d &_U, Eigen::Vector3d &_S, Eigen::Matrix3d &_V)
{
    double U[9] = {};
    double S[3] = {};
    double V[9] = {};
    svd(A(0,0), A(0,1), A(0,2), A(1,0), A(1,1), A(1,2), A(2,0), A(2,1), A(2,2),
        U[0], U[3], U[6], U[1], U[4], U[7], U[2], U[5], U[8],
        S[0], S[1], S[2],
        V[0], V[3], V[6], V[1], V[4], V[7], V[2], V[5], V[8]);
    _U << U[0], U[3], U[6],
        U[1], U[4], U[7],
        U[2], U[5], U[8];
    _S << S[0], S[1], S[2];
    _V << V[0], V[3], V[6],
        V[1], V[4], V[7],
        V[2], V[5], V[8];
}


#endif // MPM_KERNELS_CUH
