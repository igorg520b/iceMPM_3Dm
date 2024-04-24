#include "host_side_soa.h"



std::pair<Eigen::Vector3d, Eigen::Vector3d> HostSideSOA::getBlockDimensions()
{
    Eigen::Vector3d result[2];
    for(int k=0;k<SimParams3D::dim;k++)
    {
        std::pair<SOAIterator, SOAIterator> it_res =
            std::minmax_element(begin(), end(),
                                [k](ProxyPoint3D &p1, ProxyPoint3D &p2)
                                {return p1.getValue(SimParams3D::posx+k)<p2.getValue(SimParams3D::posx+k);});
        result[0][k] = (*it_res.first).getValue(SimParams3D::posx+k);
        result[1][k] = (*it_res.second).getValue(SimParams3D::posx+k);
    }
    return {result[0], result[1]};
}

void HostSideSOA::offsetBlock(Eigen::Vector3d offset)
{
    for(SOAIterator it = begin(); it!=end(); ++it)
    {
        ProxyPoint3D &p = *it;
        Eigen::Vector3d pos = p.getPos();
        pos += offset;
        p.setValue(SimParams3D::posx, pos.x());
        p.setValue(SimParams3D::posx+1, pos.y());
        p.setValue(SimParams3D::posx+2, pos.z());
    }
}

void HostSideSOA::RemoveDisabledAndSort(double hinv, int GridY)
{
    spdlog::info("RemoveDisabledAndSort; nPtsArrays {}", SimParams3D::nPtsArrays);
    unsigned size_before = size;
    SOAIterator it_result = std::remove_if(begin(), end(), [](ProxyPoint3D &p){return p.getDisabledStatus();});
    size = it_result.m_point.pos;
    spdlog::info("RemoveDisabledAndSort: {} removed; new size {}", size_before-size, size);
    std::sort(begin(), end(),
              [&hinv,&GridY](ProxyPoint &p1, ProxyPoint &p2)
              {return p1.getCellIndex(hinv,GridY)<p2.getCellIndex(hinv,GridY);});
    spdlog::info("RemoveDisabledAndSort done");
}


void HostSideSOA::Allocate(unsigned capacity)
{
    cudaFreeHost(host_buffer);
    this->capacity = capacity;
    size_t allocation_size = sizeof(double)*capacity*icy::SimParams::nPtsArrays;
    cudaError_t err = cudaMallocHost(&host_buffer, allocation_size);
    if(err != cudaSuccess)
    {
        const char *description = cudaGetErrorString(err);
        spdlog::critical("allocating host buffer of size {}: {}",allocation_size,description);
        throw std::runtime_error("allocating host buffer for points");
    }
    size = 0;
    memset(host_buffer, 0, allocation_size);
    spdlog::info("HSSOA allocate capacity {} pt; toal {} Gb", capacity, (double)allocation_size/(1024.*1024.*1024.));
}

unsigned HostSideSOA::FindFirstPointAtGridXIndex(const int index_grid_x, const double hinv)
{
    SOAIterator it = std::lower_bound(begin(),end(),index_grid_x,
                                      [hinv](const ProxyPoint &p, const int val)
                                      {return p.getXIndex(hinv)<val;});

    unsigned result_pos = it.m_point.pos;
//    int xindex = it.m_point.getXIndex(hinv);
//    spdlog::info("FindFirstPointAtGridXIndex: index_grid_x {} at pos {}; found cell_index {}", index_grid_x, result_pos, xindex);
    return result_pos;
}


void HostSideSOA::InitializeBlock()
{
    Eigen::Matrix2d identity = Eigen::Matrix2d::Identity();
    for(SOAIterator it = begin(); it!=end(); ++it)
    {
        ProxyPoint &p = *it;
        p.setValue(icy::SimParams::idx_Jp_inv,1);
        for(int i=0; i<icy::SimParams::dim; i++)
            for(int j=0; j<icy::SimParams::dim; j++)
                p.setValue(icy::SimParams::Fe00+i*2+j, identity(i,j));
    }

}


// ====================================================== ProxyPoint
ProxyPoint::ProxyPoint(const ProxyPoint &other)
{
    isReference = false;
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

ProxyPoint& ProxyPoint::operator=(const ProxyPoint &other)
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

double ProxyPoint::getValue(size_t valueIdx) const
{
    if(isReference)
        return soa[pos + pitch*valueIdx];
    else
        return data[valueIdx];
}


void ProxyPoint::setValue(size_t valueIdx, double value)
{
    if(isReference)
        soa[pos + pitch*valueIdx] = value;
    else
        data[valueIdx] = value;
}


void ProxyPoint::setPartition(uint8_t PartitionID)
{
    // retrieve the existing value
    double dval;
    if(isReference)
    {
        dval = soa[pos + pitch*icy::SimParams::idx_utility_data];
    }
    else
    {
        dval = data[icy::SimParams::idx_utility_data];
    }
    long long val = *reinterpret_cast<long long*>(&dval);

    long long _pid = (long long)PartitionID;
    _pid <<= 24;
    val &= 0xffffffff00ffffffll;
    val |= _pid;

    long long *ptr;
    if(isReference)
    {
        ptr = (long long*)&soa[pos + pitch*icy::SimParams::idx_utility_data];
    }
    else
    {
        ptr = (long long*)&data[icy::SimParams::idx_utility_data];
    }
    *ptr = val;
}


uint8_t ProxyPoint::getPartition()
{
    double dval;
    if(isReference)
    {
        dval = soa[pos + pitch*icy::SimParams::idx_utility_data];
    }
    else
    {
        dval = data[icy::SimParams::idx_utility_data];
    }
    long long val = *reinterpret_cast<long long*>(&dval);
    val >>= 24;
    return (uint8_t)(val & 0xff);
}



bool ProxyPoint::getCrushedStatus()
{
    double dval;
    if(isReference)
    {
        dval = soa[pos + pitch*icy::SimParams::idx_utility_data];
    }
    else
    {
        dval = data[icy::SimParams::idx_utility_data];
    }
    long long val = *reinterpret_cast<long long*>(&dval);
    return (val & 0x10000);
}

bool ProxyPoint::getDisabledStatus()
{
    double dval;
    if(isReference)
    {
        dval = soa[pos + pitch*icy::SimParams::idx_utility_data];
    }
    else
    {
        dval = data[icy::SimParams::idx_utility_data];
    }
    long long val = *reinterpret_cast<long long*>(&dval);
    return (val & 0x20000ll) == 0x20000ll;
}

uint16_t ProxyPoint3D::getGrain()
{
    double dval;
    if(isReference)
    {
        dval = soa[pos + pitch*SimParams3D::idx_utility_data];
    }
    else
    {
        dval = data[SimParams3D::idx_utility_data];
    }
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


// ==================================================== SOAIterator

SOAIterator::SOAIterator(unsigned pos, double *soa_data, unsigned pitch)
{
    m_point.isReference = true;
    m_point.pos = pos;
    m_point.soa = soa_data;
    m_point.pitch = pitch;
}

SOAIterator::SOAIterator(const SOAIterator& other)
{
    m_point.isReference = other.m_point.isReference;
    m_point.pos = other.m_point.pos;
    m_point.soa = other.m_point.soa;
    m_point.pitch = other.m_point.pitch;
}

SOAIterator& SOAIterator::operator=(const SOAIterator& other)
{
    m_point.isReference = other.m_point.isReference;
    m_point.pos = other.m_point.pos;
    m_point.soa = other.m_point.soa;
    m_point.pitch = other.m_point.pitch;
    return *this;
}

