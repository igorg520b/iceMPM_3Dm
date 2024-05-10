#ifndef P_SIM_H
#define P_SIM_H

#include <iostream>
#include <string>
#include <filesystem>
#include <fstream>

#include <Eigen/Core>
#include "rapidjson/reader.h"
#include "rapidjson/document.h"
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include <H5Cpp.h>
// various settings and constants of the model


struct SimParams3D
{
public:
    constexpr static double pi = 3.14159265358979323846;
    constexpr static int dim = 3;
    constexpr static int nGridArrays = 4;

    // index of the corresponding array in SoA
    constexpr static size_t idx_utility_data = 0;
    constexpr static size_t idx_P = idx_utility_data + 1;
    constexpr static size_t idx_Q = idx_P + 1;

    constexpr static size_t idx_Jp_inv = idx_Q + 1;
    constexpr static size_t posx = idx_Jp_inv + 1;
    constexpr static size_t velx = posx + 3;
    constexpr static size_t Fe00 = velx+3;
    constexpr static size_t Bp00 = Fe00+9;
    constexpr static size_t nPtsArrays = Bp00 + 9;

    int SetupType;  // 0 - ice block horizontal indentation; 1 - cone uniaxial compression

    int nPtsTotal;
    int GridXTotal, GridY, GridZ, GridTotal;
    int IndenterSubdivisions; // array dimensions for indenter force
    double DomainDimensionX;    // physical size of the simulation domain in x-direction

    double InitialTimeStep, SimulationEndTime;
    double SimulationTime;
    int UpdateEveryNthStep; // run N steps without update
    int SnapshotPeriod;     // take full snapshot every N animation frames
    int SimulationStep;

    // material properties
    double Gravity, Density, PoissonsRatio, YoungsModulus;
    double lambda, mu, kappa; // Lame

    double IceCompressiveStrength, IceTensileStrength, IceShearStrength, IceTensileStrength2;
    double NACC_beta;
    double DP_tan_phi, DP_threshold_p;
    double GrainVariability;

    // indentation params
    double IndDiameter, IndRSq, IndVelocity, IndDepth;

    double cellsize;
    double xmin, xmax, ymin, ymax, zmin, zmax;            // bounding box of the material

    double ParticleVolume, ParticleMass, ParticleViewSize;

    double indenter_x, indenter_x_initial, indenter_y, indenter_y_initial;
    double Volume;  // total volume (area) of the object

    // multi-GPU params
    int nPartitions; // number of partitions (ideally, one partition per device)
    int GridHaloSize;  // number of grid slices (perpendicular to the x-axis) for "halo" transfers
    int PointTransferFrequency; // n times per full cycle
    int VectorCapacity_transfer;   // vector capacity for points that fly to another partition
    double ExtraSpaceForIncomingPoints;     // percentage of points per partition
    double PointsTransferBufferFraction;    // space for points that can "fly over" per simulation step
    double RebalanceThresholdFreeSpaceRemaining;     // % of the total space
    double RebalanceThresholdDisabledPercentage;

    // computed parameters/properties
    double dt_vol_Dpinv, dt_Gravity, vmax, vmax_squared;
    int gbOffset;
    double cellsize_inv, Dp_inv;
    int tpb_P2G, tpb_Upd, tpb_G2P;  // threads per block for each operation
    double animation_threshold_pos;

    void Reset();
    std::string ParseFile(std::string fileName);
    void SaveParametersAsAttributes(H5::DataSet &dataset);
    void ReadParametersFromAttributes(H5::DataSet &dataset);

    void ComputeLame();
    void ComputeCamClayParams();
    void ComputeHelperVariables();

    double PointsPerCell() {return nPtsTotal/(Volume/(cellsize*cellsize*cellsize));}
    int AnimationFrameNumber() { return SimulationStep / UpdateEveryNthStep;}
    size_t IndenterArraySize() { return sizeof(double)*dim*IndenterSubdivisions*GridZ; }

    // grid cell from point's coordinates
    int CellIdx(float x) { return (int)(x*cellsize_inv+0.5); }
};

#endif
