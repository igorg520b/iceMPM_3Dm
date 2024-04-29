#include "snapshotmanager.h"
#include "model_3d.h"

#include <spdlog/spdlog.h>
#include <H5Cpp.h>
#include <filesystem>
#include <string>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <utility>


void SnapshotManager::SaveSnapshot(std::string outputDirectory, bool compress)
{
    std::filesystem::path odp(outputDirectory);
    if(!std::filesystem::is_directory(odp) || !std::filesystem::exists(odp)) std::filesystem::create_directory(odp);
    std::filesystem::path odp2(outputDirectory+"/"+directory_snapshots);
    if(!std::filesystem::is_directory(odp2) || !std::filesystem::exists(odp2)) std::filesystem::create_directory(odp2);

    const int current_frame_number = model->prms.AnimationFrameNumber();
    char fileName[20];
    snprintf(fileName, sizeof(fileName), "d%05d.h5", current_frame_number);
    std::string filePath = outputDirectory + "/" + directory_snapshots + "/" + fileName;
    spdlog::info("saving NC frame {} to file {}", current_frame_number, filePath);


    H5::H5File file(filePath, H5F_ACC_TRUNC);

    // indenter
    hsize_t dims_indenter = model->prms.IndenterSubdivisions * model->prms.GridZ * SimParams3D::dim;
    H5::DataSpace dataspace_indenter(1, &dims_indenter);
    H5::DataSet dataset_indenter = file.createDataSet("Indenter", H5::PredType::NATIVE_DOUBLE, dataspace_indenter);
    dataset_indenter.write(model->gpu.indenter_sensor_total.data(), H5::PredType::NATIVE_DOUBLE);

    // points
    hsize_t dims_points = model->gpu.hssoa.capacity * SimParams3D::nPtsArrays;
    H5::DataSpace dataspace_points(1, &dims_points);
    H5::DSetCreatPropList proplist;
    if(compress)
    {
        hsize_t chunk_dims = (hsize_t)std::min((unsigned)(1024*1024), (unsigned)model->gpu.hssoa.size);
        proplist.setChunk(1, &chunk_dims);
        proplist.setDeflate(7);
    }
    H5::DataSet dataset_points = file.createDataSet("Points", H5::PredType::NATIVE_DOUBLE, dataspace_points, proplist);
    dataset_points.write(model->gpu.hssoa.host_buffer, H5::PredType::NATIVE_DOUBLE);

    hsize_t att_dim = 1;
    H5::DataSpace att_dspace(1, &att_dim);
    H5::Attribute att_HSSOA_capacity = dataset_points.createAttribute("HSSOA_capacity", H5::PredType::NATIVE_UINT, att_dspace);
    H5::Attribute att_HSSOA_size = dataset_points.createAttribute("HSSOA_size", H5::PredType::NATIVE_UINT, att_dspace);
    att_HSSOA_capacity.write(H5::PredType::NATIVE_UINT, &model->gpu.hssoa.capacity);
    att_HSSOA_size.write(H5::PredType::NATIVE_UINT, &model->gpu.hssoa.size);

    model->prms.SaveParametersAsAttributes(dataset_points);

    file.close();
}



void SnapshotManager::LoadRawPoints(std::string fileName)
{
    spdlog::info("reading raw points file {}",fileName);
    if(!std::filesystem::exists(fileName)) throw std::runtime_error("error reading raw points file - no file");;

    H5::H5File file(fileName, H5F_ACC_RDONLY);

    H5::DataSet dataset_grains = file.openDataSet("llGrainIDs");
    hsize_t nPoints;
    dataset_grains.getSpace().getSimpleExtentDims(&nPoints, NULL);
    model->prms.nPtsTotal = nPoints;

    // allocate space host-side
    model->gpu.hssoa.Allocate(nPoints*(1+model->prms.ExtraSpaceForIncomingPoints));
    model->gpu.hssoa.size = nPoints;

    // read
    file.openDataSet("x").read(model->gpu.hssoa.getPointerToLine(SimParams3D::posx), H5::PredType::NATIVE_DOUBLE);
    file.openDataSet("y").read(model->gpu.hssoa.getPointerToLine(SimParams3D::posx+1), H5::PredType::NATIVE_DOUBLE);
    file.openDataSet("z").read(model->gpu.hssoa.getPointerToLine(SimParams3D::posx+2), H5::PredType::NATIVE_DOUBLE);
    dataset_grains.read(model->gpu.hssoa.host_buffer, H5::PredType::NATIVE_UINT64);

    // read volume attribute
    H5::Attribute att_volume = dataset_grains.openAttribute("volume");
    att_volume.read(H5::PredType::NATIVE_DOUBLE, &model->prms.Volume);
    file.close();

    // get block dimensions
    std::pair<Eigen::Vector3d, Eigen::Vector3d> boundaries = model->gpu.hssoa.getBlockDimensions();
    model->prms.xmin = boundaries.first.x();
    model->prms.ymin = boundaries.first.y();
    model->prms.zmin = boundaries.first.z();
    model->prms.xmax = boundaries.second.x();
    model->prms.ymax = boundaries.second.y();
    model->prms.zmax = boundaries.second.z();


    const double &h = model->prms.cellsize;
    const double box_x = model->prms.GridXTotal*h;
    const double box_z = model->prms.GridZ*h;
    const double length = model->prms.xmax - model->prms.xmin;
    const double width = model->prms.zmax - model->prms.zmin;
    const double x_offset = (box_x - length)/2;
    const double y_offset = 2*h;
    const double z_offset = (box_z - width)/2;

    Eigen::Vector3d offset(x_offset, y_offset, z_offset);
    model->gpu.hssoa.offsetBlock(offset);
    model->gpu.hssoa.RemoveDisabledAndSort(model->prms.cellsize_inv, model->prms.GridY, model->prms.GridZ);
    model->gpu.hssoa.InitializeBlock();

    // set indenter starting position
    const double block_left = x_offset;
    const double block_top = model->prms.ymax + y_offset;

    const double r = model->prms.IndDiameter/2;
    const double ht = r - model->prms.IndDepth;
    const double x_ind_offset = sqrt(r*r - ht*ht);

    model->prms.indenter_x = floor((block_left-x_ind_offset)/h)*h;
    if(model->prms.SetupType == 0)
        model->prms.indenter_y = block_top + ht;
    else if(model->prms.SetupType == 1)
        model->prms.indenter_y = ceil(block_top/h)*h;

    model->prms.indenter_x_initial = model->prms.indenter_x;
    model->prms.indenter_y_initial = model->prms.indenter_y;

    // particle volume and mass
    model->prms.ParticleVolume = model->prms.Volume/nPoints;
    model->prms.ParticleMass = model->prms.ParticleVolume * model->prms.Density;
    model->prms.ComputeHelperVariables();

    // allocate GPU partitions
    model->gpu.initialize();
    model->gpu.split_hssoa_into_partitions();
    model->gpu.allocate_arrays();
    model->gpu.transfer_ponts_to_device();

    model->Reset();
    model->Prepare();

    spdlog::info("LoadRawPoints done; nPoitns {}",nPoints);
}


void SnapshotManager::ReadSnapshot(std::string fileName, int partitions)
{
    if(!std::filesystem::exists(fileName)) return;
    spdlog::info("reading snapshot {}", fileName);

    H5::H5File file(fileName, H5F_ACC_RDONLY);

    H5::DataSet dataset_points = file.openDataSet("Points");
    model->prms.ReadParametersFromAttributes(dataset_points);
    if(partitions > 0)
    {
        model->prms.nPartitions = partitions;
        model->prms.ComputeHelperVariables();
    }

    // allocate HSOA
    unsigned capacity, size;
    H5::Attribute att_capacity = dataset_points.openAttribute("HSSOA_capacity");
    H5::Attribute att_size = dataset_points.openAttribute("HSSOA_size");
    att_capacity.read(H5::PredType::NATIVE_INT, &capacity);
    att_size.read(H5::PredType::NATIVE_INT, &size);
    model->gpu.hssoa.Allocate(capacity);
    model->gpu.hssoa.size = size;

    // read point data
    dataset_points.read(model->gpu.hssoa.host_buffer, H5::PredType::NATIVE_DOUBLE);

    file.close();

    // distribute into partitions
    model->gpu.hssoa.RemoveDisabledAndSort(model->prms.cellsize_inv, model->prms.GridY, model->prms.GridZ);

    // allocate GPU partitions
    model->gpu.initialize();
    model->gpu.split_hssoa_into_partitions();
    model->gpu.allocate_arrays();
    model->gpu.transfer_ponts_to_device();

    model->max_points_transferred = 0;
    model->SyncTopologyRequired = true;

    model->Prepare();

    spdlog::info("SnapshotManager::ReadSnapshot() done");
    return;
}
