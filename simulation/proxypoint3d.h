#ifndef PROXYPOINT3D_H
#define PROXYPOINT3D_H

#include <Eigen/Core>
#include "parameters_sim_3d.h"

struct ProxyPoint3D
{
    bool isReference = false;
    unsigned pos, pitch;    // element # and capacity of each array in SOA
    double *soa;            // reference to SOA (assume contiguous space of size nArrays*pitch)
    double data[SimParams3D::nPtsArrays];    // local copy of the data when isReference==true

    ProxyPoint3D() { isReference = false; }
    ProxyPoint3D(const ProxyPoint3D &other);
    ProxyPoint3D& operator=(const ProxyPoint3D &other);

    // access data
    double getValue(size_t valueIdx) const;   // valueIdx < nArrays
    void setValue(size_t valueIdx, double value);
    Eigen::Vector3d getPos() const;
    Eigen::Vector3d getVelocity() const;
    bool getCrushedStatus();
    bool getDisabledStatus();
    uint16_t getGrain();
    int getCellIndex(double hinv, int GridY, int GridZ);  // index of the grid cell at the point's location
    int getXIndex(double hinv) const;                     // x-index of the grid cell
    void setPartition(uint8_t PartitionID);
    uint8_t getPartition();
};

#endif // PROXYPOINT3D_H
