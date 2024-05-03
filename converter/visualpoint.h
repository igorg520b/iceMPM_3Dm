#ifndef VISUALPOINT_H
#define VISUALPOINT_H

#include <H5Cpp.h>
#include <Eigen/Core>

struct VisualPoint
{
    float pos[3], vel[3];
    float Jp_inv;
    uint8_t status;

    static H5::CompType ctUpdPV, ctUpdJp, ctUpdS, ctVisualPoint;
    static void InitializeStatic();

    Eigen::Map<Eigen::Vector3f> getVel() { return Eigen::Map<Eigen::Vector3f>(vel);}
    Eigen::Map<Eigen::Vector3f> getPos() { return Eigen::Map<Eigen::Vector3f>(pos);}
    int getCrushedStatus() {return (status & 0b10000000)==0 ? 0 : 1;}
    int getPartition() {return (int)(status & 0b01111111);}
};

#endif // VISUALPOINT_H
