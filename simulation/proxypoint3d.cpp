#include "proxypoint3d.h"



ProxyPoint3D::ProxyPoint3D(const ProxyPoint3D &other)
{
    isReference = false;
    *this = other;
}

ProxyPoint3D& ProxyPoint3D::operator=(const ProxyPoint3D &other)
{
    if(isReference)
    {
        // distribute into soa
        if(other.isReference)
        {
            for(int i=0;i<SimParams3D::nPtsArrays;i++) soa[pos + i*pitch] = other.soa[other.pos + i*other.pitch];
        }
        else
        {
            for(int i=0;i<SimParams3D::nPtsArrays;i++) soa[pos + i*pitch] = other.data[i];
        }
    }
    else
    {
        // local copy
        if(other.isReference)
        {
            for(int i=0;i<SimParams3D::nPtsArrays;i++) data[i] = other.soa[other.pos + i*other.pitch];
        }
        else
        {
            for(int i=0;i<SimParams3D::nPtsArrays;i++) data[i] = other.data[i];
        }
    }
    return *this;
}

Eigen::Vector3d ProxyPoint3D::getPos() const
{
    Eigen::Vector3d result;
    if(isReference)
    {
        for(int i=0; i<SimParams3D::dim;i++)
            result[i] = soa[pos + pitch*(SimParams3D::posx+i)];
    }
    else
    {
        for(int i=0; i<SimParams3D::dim;i++)
            result[i] = data[SimParams3D::posx+i];
    }
    return result;
}

double ProxyPoint3D::getValue(size_t valueIdx) const
{
    if(isReference)
        return soa[pos + pitch*valueIdx];
    else
        return data[valueIdx];
}

void ProxyPoint3D::setValue(size_t valueIdx, double value)
{
    if(isReference)
        soa[pos + pitch*valueIdx] = value;
    else
        data[valueIdx] = value;
}

void ProxyPoint3D::setPartition(uint8_t PartitionID)
{
    // retrieve the existing value
    double dval = getValue(SimParams3D::idx_utility_data);
    long long val = *reinterpret_cast<long long*>(&dval);

    long long _pid = (long long)PartitionID;
    _pid <<= 24;
    val &= 0xffffffff00ffffffll;
    val |= _pid;

    long long *ptr;
    if(isReference)
    {
        ptr = (long long*)&soa[pos + pitch*SimParams3D::idx_utility_data];
    }
    else
    {
        ptr = (long long*)&data[SimParams3D::idx_utility_data];
    }
    *ptr = val;
}

uint8_t ProxyPoint3D::getPartition()
{
    double dval = getValue(SimParams3D::idx_utility_data);
    long long val = *reinterpret_cast<long long*>(&dval);
    val >>= 24;
    return (uint8_t)(val & 0xff);
}

bool ProxyPoint3D::getCrushedStatus()
{
    double dval = getValue(SimParams3D::idx_utility_data);
    long long val = *reinterpret_cast<long long*>(&dval);
    return (val & 0x10000);
}

bool ProxyPoint3D::getDisabledStatus()
{
    double dval = getValue(SimParams3D::idx_utility_data);
    long long val = *reinterpret_cast<long long*>(&dval);
    return (val & 0x20000ll) == 0x20000ll;
}

uint16_t ProxyPoint3D::getGrain()
{
    double dval = getValue(SimParams3D::idx_utility_data);
    long long val = *reinterpret_cast<long long*>(&dval);
    return (val & 0xffff);
}

int ProxyPoint3D::getCellIndex(double hinv, unsigned GridY, int GridZ)
{
    Eigen::Vector3d v = getPos();
    Eigen::Vector3i idx = (v*hinv + Eigen::Vector3d::Constant(0.5)).cast<int>();
    return idx.x()*GridY*GridZ + idx.y()*GridZ + idx.z();
}

int ProxyPoint3D::getXIndex(double hinv) const
{
    double x = getValue(SimParams3D::posx);
    int x_idx = (int)(x*hinv + 0.5);
    return x_idx;
}

