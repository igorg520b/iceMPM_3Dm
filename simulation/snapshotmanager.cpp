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
#include <chrono>

SnapshotManager::SnapshotManager()
{
    VisualPoint::InitializeStatic();
}

void SnapshotManager::SaveFrame(std::string outputDirectory)
{
    // populate saved_frame
    const int frame = model->prms.AnimationFrameNumber();
    char fileName[20];
    snprintf(fileName, sizeof(fileName), "v%05d.h5", frame);
    std::string fullFilePath = outputDirectory + "/" + fileName;
    spdlog::info("SnapshotManager::SaveFrame(): frame {} to file {}", frame, fullFilePath);

    // ensure that directory exists
    std::filesystem::path od(outputDirectory);
    if(!std::filesystem::is_directory(od) || !std::filesystem::exists(od)) std::filesystem::create_directory(od);

    // file
    H5::H5File file(fullFilePath, H5F_ACC_TRUNC);

    // indenter
    hsize_t dims_indenter = model->prms.IndenterSubdivisions * model->prms.GridZ * SimParams3D::dim;
    H5::DataSpace dataspace_indenter(1, &dims_indenter);
    hsize_t chunk_dims_indenter = 10000;
    if(chunk_dims_indenter > dims_indenter) chunk_dims_indenter = dims_indenter;
    H5::DSetCreatPropList proplist2;
    proplist2.setChunk(1, &chunk_dims_indenter);
    proplist2.setDeflate(5);
    H5::DataSet dataset_indenter = file.createDataSet("Indenter", H5::PredType::NATIVE_DOUBLE, dataspace_indenter, proplist2);
    dataset_indenter.write(model->gpu.indenter_sensor_total.data(), H5::PredType::NATIVE_DOUBLE);

    // type of frame
    hsize_t att_dim = 1;
    H5::DataSpace att_dspace(1, &att_dim);
    uint8_t prev_frame = previous_frame_exists ? 1 : 0;
    H5::Attribute att = dataset_indenter.createAttribute("partial_frame", H5::PredType::NATIVE_UINT8, att_dspace);
    att.write(H5::PredType::NATIVE_UINT8, &prev_frame);

    model->prms.SaveParametersAsAttributes(dataset_indenter);

    // time stamp
    auto now = std::chrono::system_clock::now();
    uint64_t seconds = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
    H5::Attribute att_timestamp = dataset_indenter.createAttribute("timestamp", H5::PredType::NATIVE_UINT64, att_dspace);
    att_timestamp.write(H5::PredType::NATIVE_UINT64, &seconds);

    spdlog::info("previous_frame_exists {}", previous_frame_exists);
    if(!previous_frame_exists)
    {
        // save full frame data in a compressed way
        visual_state.resize(model->prms.nPtsTotal);
        last_pos_refresh_frame.resize(model->prms.nPtsTotal);

        int count = 0;
        for(int i=0;i<model->gpu.hssoa.size;i++)
        {
            SOAIterator s = model->gpu.hssoa.begin()+i;
            if(s->getDisabledStatus()) continue;
            VisualPoint &vp = visual_state[count];
            Eigen::Vector3d pos = s->getPos();
            vp.pos[0] = pos[0]; vp.pos[1] = pos[1]; vp.pos[2] = pos[2];
            Eigen::Vector3d vel = s->getVelocity();
            vp.vel[0] = vel[0]; vp.vel[1] = vel[1]; vp.vel[2] = vel[2];
            vp.Jp_inv = s->getValue(SimParams3D::idx_Jp_inv);
            vp.status = s->getPartition();
            if(s->getCrushedStatus()) vp.status |= 0b10000000;
            count++;
        }
        // saved_frame is a full copy of the current_frame
        std::fill(last_pos_refresh_frame.begin(), last_pos_refresh_frame.end(), frame);

        // write the file
        hsize_t dims_points = count;
        H5::DataSpace dataspace_points(1, &dims_points);
        H5::DSetCreatPropList proplist;
        hsize_t chunk_dims = (hsize_t)std::min((unsigned)(1024*1024), (unsigned)count);
        proplist.setChunk(1, &chunk_dims);
        proplist.setDeflate(5);
        H5::DataSet dataset_points = file.createDataSet("VisualPoints", VisualPoint::ctVisualPoint, dataspace_points, proplist);
        dataset_points.write(visual_state.data(), VisualPoint::ctVisualPoint);

        previous_frame_exists = true;
        spdlog::info("writing full frame; pts {}; previous_frame_exists {}", dims_points, previous_frame_exists);
    }
    else
    {
        // save partial frame data
        update_pos_vel.clear();
        update_Jp.clear();
        update_status.clear();

        double &dt = model->prms.InitialTimeStep;
        int &coeff = model->prms.UpdateEveryNthStep;

        int count = 0;
        for(int i=0;i<model->gpu.hssoa.size;i++)
        {
            SOAIterator s = model->gpu.hssoa.begin()+i;
            if(s->getDisabledStatus()) continue;
            VisualPoint vp = visual_state[count];

            Eigen::Vector3d pos = s->getPos();
            Eigen::Vector3d vel = s->getVelocity();
            float Jp_inv = s->getValue(SimParams3D::idx_Jp_inv);
            uint8_t status = s->getPartition();
            if(s->getCrushedStatus()) status |= 0b10000000;

            int elapsed_frames = frame - last_pos_refresh_frame[count];
            Eigen::Vector3f predicted_position = Eigen::Map<Eigen::Vector3f>(vp.pos) + Eigen::Map<Eigen::Vector3f>(vp.vel)*(dt*elapsed_frames*coeff);
            Eigen::Vector3f prediction_error = pos.cast<float>() - predicted_position;
            if(prediction_error.norm() > threshold_pos)
            {
                last_pos_refresh_frame[count] = frame;
                Eigen::Map<Eigen::Vector3f>(vp.pos) = pos.cast<float>();
                Eigen::Map<Eigen::Vector3f>(vp.vel) = vel.cast<float>();
                std::array<float, 6> arr = {vp.pos[0], vp.pos[1], vp.pos[2], vp.vel[0], vp.vel[1], vp.vel[2]};
                update_pos_vel.push_back({count, arr});
            }

            if(abs(vp.Jp_inv - Jp_inv) > threshold_Jp)
            {
                vp.Jp_inv = Jp_inv;
                update_Jp.push_back({count, Jp_inv});
            }

            if(vp.status != status)
            {
                vp.status = status;
                update_status.push_back({count, status});
            }
            count++;
        }

        // write file
        if(update_pos_vel.size() > 0)
        {
            H5::DSetCreatPropList proplist3;
            hsize_t pos_dims = (hsize_t)std::min((size_t)1024*64, update_pos_vel.size());
            proplist3.setChunk(1, &pos_dims);
            proplist3.setDeflate(5);
            hsize_t dims_update_pos_vel = update_pos_vel.size();
            H5::DataSpace dsp_update_vel(1, &dims_update_pos_vel);
            H5::DataSet ds_update_vel = file.createDataSet("UpdatePos", VisualPoint::ctUpdPV, dsp_update_vel, proplist3);
            ds_update_vel.write(update_pos_vel.data(), VisualPoint::ctUpdPV);
        }

        if(update_status.size() > 0)
        {
            H5::DSetCreatPropList proplist2;
            hsize_t status_dims = (hsize_t)std::min((size_t)1024*64, update_status.size());
            proplist2.setChunk(1, &status_dims);
            proplist2.setDeflate(5);
            hsize_t dims_update_status = update_status.size();
            H5::DataSpace dsp_update_status(1, &dims_update_status);
            H5::DataSet ds_update_status = file.createDataSet("UpdateStatus", VisualPoint::ctUpdS, dsp_update_status, proplist2);
            ds_update_status.write(update_status.data(), VisualPoint::ctUpdS);
        }

        if(update_Jp.size() > 0)
        {
            // Jp_inv, P, Q
            H5::DSetCreatPropList proplist4;
            hsize_t jp_dims = (hsize_t)std::min((size_t)1024*64, update_Jp.size());
            proplist4.setChunk(1, &jp_dims);
            proplist4.setDeflate(5);
            hsize_t dims_update_Jp = update_Jp.size();
            H5::DataSpace dsp_update_Jp(1, &dims_update_Jp);
            H5::DataSet ds_update_Jp = file.createDataSet("UpdateJp", VisualPoint::ctUpdJp, dsp_update_Jp, proplist4);
            ds_update_Jp.write(update_Jp.data(), VisualPoint::ctUpdJp);
        }

        spdlog::info("writing iterative frame ");
    }
    file.close();
}





void SnapshotManager::SaveSnapshot(std::string outputDirectory, bool compress)
{
    std::filesystem::path odp(outputDirectory);
    if(!std::filesystem::is_directory(odp) || !std::filesystem::exists(odp)) std::filesystem::create_directory(odp);

    const int current_frame_number = model->prms.AnimationFrameNumber();
    char fileName[20];
    snprintf(fileName, sizeof(fileName), "d%05d.h5", current_frame_number);
    std::string filePath = outputDirectory + "/" + fileName;
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
        proplist.setDeflate(5);
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

    previous_frame_exists = false;
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
    spdlog::info("reading the points buffer");
    dataset_points.read(model->gpu.hssoa.host_buffer, H5::PredType::NATIVE_DOUBLE);
    spdlog::info("reading poings buffer - done");

    file.close();

    // distribute into partitions
    // always squeeze/sort before saving
    if(model->prms.SimulationStep != 0)
        model->gpu.hssoa.RemoveDisabledAndSort(model->prms.cellsize_inv, model->prms.GridY, model->prms.GridZ);

    // allocate GPU partitions
    model->gpu.initialize();
    model->gpu.split_hssoa_into_partitions();
    model->gpu.allocate_arrays();
    model->gpu.transfer_ponts_to_device();

    model->max_points_transferred = 0;
    model->SyncTopologyRequired = true;

    model->Prepare();

    previous_frame_exists = false;
    spdlog::info("SnapshotManager::ReadSnapshot() done");
}
