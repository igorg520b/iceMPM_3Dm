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

void HostSideSOA::MarkSolidAndLiquid(Eigen::Vector3d blockDimensions, Eigen::Vector3d blockCenter)
{
    int count = 0;
    Eigen::Vector3d from = blockCenter - 0.5*blockDimensions;
    Eigen::Vector3d to = blockCenter + 0.5*blockDimensions;

    for(SOAIterator it = begin(); it!=end(); ++it)
    {
        ProxyPoint3D &p = *it;
        Eigen::Vector3d pos = p.getPos();
        if(pos.x() < from.x()) { p.setLiquidStatus(true); continue; }
        if(pos.y() < from.y()) { p.setLiquidStatus(true); continue; }
        if(pos.z() < from.z()) { p.setLiquidStatus(true); continue; }
        if(pos.x() > to.x()) { p.setLiquidStatus(true); continue; }
        if(pos.y() > to.y()) { p.setLiquidStatus(true); continue; }
        if(pos.z() > to.z()) { p.setLiquidStatus(true); continue; }
        count++;
    }
    spdlog::info("MarkSolidAndLiquid from [{},{},{}] to [{},{},{}]; marked {} pts",
                 from[0],from[1],from[2], to[0],to[1],to[2],count);
}


void HostSideSOA::RemoveDisabledAndSort(double hinv, int GridY, int GridZ)
{
    spdlog::info("RemoveDisabledAndSort; nPtsArrays {}", SimParams3D::nPtsArrays);
    unsigned size_before = size;
    SOAIterator it_result = std::remove_if(begin(), end(), [](ProxyPoint3D &p){return p.getDisabledStatus();});
    size = it_result.m_point.pos;
    spdlog::info("RemoveDisabledAndSort: {} removed; new size {}", size_before-size, size);
    std::sort(begin(), end(),
              [=](ProxyPoint3D &p1, ProxyPoint3D &p2)
              {return p1.getCellIndex(hinv,GridY,GridZ)<p2.getCellIndex(hinv,GridY,GridZ);});
    spdlog::info("RemoveDisabledAndSort done");
}


void HostSideSOA::Allocate(unsigned capacity)
{
    cudaFreeHost(host_buffer);
    this->capacity = capacity;
    size_t allocation_size = sizeof(double)*capacity*SimParams3D::nPtsArrays;
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


void HostSideSOA::InitializeBlock()
{
    for(SOAIterator it = begin(); it!=end(); ++it)
    {
        ProxyPoint3D &p = *it;
        p.setValue(SimParams3D::idx_Jp_inv,1);
        for(int i=0; i<SimParams3D::dim; i++)
                p.setValue(SimParams3D::Fe00+i*3+i, 1.0);
    }
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

