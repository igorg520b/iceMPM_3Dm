#ifndef GPU_PARTITION_H
#define GPU_PARTITION_H

#include <Eigen/Core>
#include <Eigen/LU>
#include <spdlog/spdlog.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <functional>
#include <vector>

#include "parameters_sim_3d.h"
#include "point_3d.h"
#include "host_side_soa.h"


// kernels
__global__ void partition_kernel_p2g(const int gridX, const int gridX_offset, const int pitch_grid,
                              const int count_pts, const int pitch_pts,
                                     const double *buffer_pts, double *buffer_grid);


// receive grid data from adjacent partitions
__global__ void partition_kernel_receive_halos_left(const int haloElementCount, const int gridX_partition,
                                               const int pitch_grid, double *buffer_grid,
                                               const double *halo0, const double *halo1);

__global__ void partition_kernel_receive_halos_right(const int haloElementCount, const int gridX_partition,
                                               const int pitch_grid, double *buffer_grid,
                                               const double *halo0, const double *halo1);




__global__ void partition_kernel_update_nodes(const Eigen::Vector2d indCenter,
                                              const int nNodes, const int gridX_offset, const int pitch_grid,
                                              double *_buffer_grid, double *indenter_force_accumulator);


__global__ void partition_kernel_g2p(const bool recordPQ, const bool enablePointTransfer,
                                     const int gridX, const int gridX_offset, const int pitch_grid,
                                     const int count_pts, const int pitch_pts,
                                     double *buffer_pts, const double *buffer_grid,
                                     int *utility_data,
                                     const int VectorCapacity_transfer,
                                     double *point_buffer_left, double *point_buffer_right);


// take points from the receive buffer and add them to the list
__global__ void partition_kernel_receive_points(const int count_transfer,
                                                const int count_pts, const int pitch_pts,
                                                double *buffer_pts,
                                                double *point_transfer_buffer);

// write point from SOA into a tanfer buffer
__device__ void PreparePointForTransfer(const int pt_idx, const int index_in_transfer_buffer,
                                        double *point_transfer_buffer, const int pitch_pts,
                                        const double *buffer_pts);


__device__ void Wolper_Drucker_Prager(Point3D &p);
__device__ void CheckIfPointIsInsideFailureSurface(Point3D &p);
__device__ Eigen::Matrix3d KirchhoffStress_Wolper(const Eigen::Matrix3d &F);
__device__ Eigen::Matrix3d Water(const double J);

__device__ void ComputePQ(Point3D  &p, const double &kappa, const double &mu);
__device__ void ComputeSVD(Point3D  &p, const double &kappa, const double &mu);
__device__ void GetParametersForGrain(short grain, double &pmin, double &pmax, double &qmax, double &beta, double &mSq, double &pmin2);

__device__ Eigen::Vector3d dev_d(Eigen::Vector3d Adiag);
__device__ Eigen::Matrix3d dev(Eigen::Matrix3d A);
__device__ void svd3x3(const Eigen::Matrix3d &A, Eigen::Matrix3d &_U, Eigen::Vector3d &_S, Eigen::Matrix3d &_V);


struct GPU_Partition_3D
{
    // these are indices in utility_data array
    constexpr static int idx_transfer_to_left = 0;
    constexpr static int idx_transfer_to_right = 1;
    constexpr static int idx_pts_max_extent = 2;
    constexpr static size_t utility_data_size = 3;

    GPU_Partition_3D();
    ~GPU_Partition_3D();

    // preparation
    void initialize(int device, int partition);
    void allocate(int n_points_capacity, int grid_x_capacity);
    void transfer_points_from_soa_to_device(HostSideSOA &hssoa, int point_idx_offset);
    void update_constants();
    void transfer_from_device(HostSideSOA &hssoa, int point_idx_offset);

    // calculation
    void reset_grid();
    void reset_indenter_force_accumulator();
    void p2g();
    void receive_halos();   // neightbour halos were copied, but we need to incorporate them into the grid
    void update_nodes();
    void g2p(const bool recordPQ, const bool enablePointTransfer);
    void receive_points(int nFromLeft, int nFromRight);

    // analysis
    void reset_timings();
    void record_timings(const bool enablePointTransfer);
    void normalize_timings(int cycles);

    // helper functions
    int getLeftBufferCount() {return host_side_utility_data[idx_transfer_to_left];}
    int getRightBufferCount() {return host_side_utility_data[idx_transfer_to_right];}
    int getMaxDeviationValue() {return host_side_utility_data[idx_pts_max_extent];}

    // host-side data
    int PartitionID;
    int Device;
    static SimParams3D *prms;

    size_t nPtsPitch, nGridPitch; // in number of elements(!), for coalesced access on the device
    int nPts_partition;    // actual number of points (including disabled)
    int nPts_disabled;      // count the disabled points in this partition
    int GridX_partition;   // size of the portion of the grid for which this partition is "responsible"
    int GridX_offset;      // index where the grid starts in this partition

    double *host_side_indenter_force_accumulator;
    int *host_side_utility_data; // sizes of outbound pt tranfer buffers (2), disabled pts (1)

    // stream and events
    cudaStream_t streamCompute;

    cudaEvent_t event_10_cycle_start;
    cudaEvent_t event_20_grid_halo_sent;
    cudaEvent_t event_30_halo_accepted;
    cudaEvent_t event_40_grid_updated;
    cudaEvent_t event_50_g2p_completed;
    cudaEvent_t event_70_pts_sent;
    cudaEvent_t event_80_pts_accepted;

    bool initialized = false;
    uint8_t error_code = 0;

    // device-side data
    int *device_side_utility_data;
    double *pts_array, *grid_array, *indenter_force_accumulator;

    // Four GPU-side vectors to keep track of points that escape and arrive
    // points that fly to/from the adjacent partitions (left-out, right-out, left-in, right-in)
    double *point_transfer_buffer[4];

    // testing
    double *halo_transfer_buffer[2];

    // frame analysis
    float timing_10_P2GAndHalo;
    float timing_20_acceptHalo;
    float timing_30_updateGrid;
    float timing_40_G2P;
    float timing_60_ptsSent;
    float timing_70_ptsAccepted;
    float timing_stepTotal;
    int max_pts_sent;
    int max_pt_deviation;
};


#endif // GPU_PARTITION_H
