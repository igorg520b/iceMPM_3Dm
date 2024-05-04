#include "converter.h"

std::mutex *Converter::accessing_indenter_force_file;
H5::DataSet *Converter::dataset_indenter_totals;
//H5::DataSpace *Converter::dataspace;
int Converter::frames_total;


Converter::Converter()
{
//    writer3->SetDataModeToAscii();

    cylinder->SetResolution(33);
    transform->RotateX(90);
    transformFilter->SetTransform(transform);
    transformFilter->SetInputConnection(cylinder->GetOutputPort());
    appendFilter->SetInputConnection(transformFilter->GetOutputPort());

    writer2->SetInputData(unstructuredGrid);


    values->SetName("Pressure");
    structuredGrid->GetCellData()->SetScalars(values);
    structuredGrid->SetPoints(grid_points);
    writer3->SetInputData(structuredGrid);

    values_Jp->SetName("Jp_inv");
    values_status->SetName("status");
    values_partitions->SetName("partitions");

    polydata->SetPoints(points);
    polydata->GetPointData()->AddArray(values_status);
    polydata->GetPointData()->AddArray(values_Jp);
    polydata->GetPointData()->AddArray(values_partitions);
    writer1->SetInputData(polydata);

}

void Converter::read_full_frame(H5::H5File &file, H5::DataSet &dataset_indenter)
{
    // load full frame
    H5::DataSet dataset_full = file.openDataSet("VisualPoints");
    hsize_t nPoints;
    dataset_full.getSpace().getSimpleExtentDims(&nPoints, NULL);

    if(frame == frame_start)
    {
        v.resize(nPoints);
        last_pos_refresh_frame.resize(nPoints);
        H5::Attribute att_IndenterSubdivisions = dataset_indenter.openAttribute("IndenterSubdivisions");
        att_IndenterSubdivisions.read(H5::PredType::NATIVE_INT, &IndenterSubdivisions);
        H5::Attribute att_GridZ = dataset_indenter.openAttribute("GridZ");
        att_GridZ.read(H5::PredType::NATIVE_INT, &GridZ);
        indenter_data.resize(3*GridZ*IndenterSubdivisions);

        H5::Attribute att_UpdateEveryNthStep = dataset_indenter.openAttribute("UpdateEveryNthStep");
        att_UpdateEveryNthStep.read(H5::PredType::NATIVE_INT, &UpdateEveryNthStep);

        H5::Attribute att_ind_diameter = dataset_indenter.openAttribute("IndDiameter");
        att_ind_diameter.read(H5::PredType::NATIVE_DOUBLE, &IndDiameter);
        H5::Attribute att_ind_cellsize = dataset_indenter.openAttribute("cellsize");
        att_ind_cellsize.read(H5::PredType::NATIVE_DOUBLE, &cellsize);
        H5::Attribute att_dt = dataset_indenter.openAttribute("InitialTimeStep");
        att_dt.read(H5::PredType::NATIVE_DOUBLE, &dt);
    }
    dataset_full.read(v.data(), VisualPoint::ctVisualPoint);
    std::fill(last_pos_refresh_frame.begin(), last_pos_refresh_frame.end(),frame);
}


void Converter::read_partial_frame(H5::H5File &file)
{
    update_pos_vel.clear();
    update_Jp.clear();
    update_status.clear();


    // update position and velocity
    if(H5Lexists(file.getId(), "UpdatePos", H5P_DEFAULT ) > 0)
    {
        H5::DataSet ds_pos = file.openDataSet("UpdatePos");
        hsize_t n;
        ds_pos.getSpace().getSimpleExtentDims(&n, NULL);
        update_pos_vel.resize(n);
        ds_pos.read(update_pos_vel.data(), VisualPoint::ctUpdPV);
    }

    for(int i=0; i<update_pos_vel.size(); i++)
    {
        std::pair<int, std::array<float,6>> &p = update_pos_vel[i];
        int pt_idx = p.first;
        std::array<float,6> &vals = p.second;
        last_pos_refresh_frame[pt_idx] = frame;
        VisualPoint &vp = v[pt_idx];
        vp.pos[0] = vals[0];
        vp.pos[1] = vals[1];
        vp.pos[2] = vals[2];
        vp.vel[0] = vals[3];
        vp.vel[1] = vals[4];
        vp.vel[2] = vals[5];
    }

    // update status

    if(H5Lexists(file.getId(), "UpdateStatus", H5P_DEFAULT ) > 0)
    {
        H5::DataSet ds_status = file.openDataSet("UpdateStatus");
        hsize_t n;
        ds_status.getSpace().getSimpleExtentDims(&n, NULL);
        update_status.resize(n);
        ds_status.read(update_status.data(), VisualPoint::ctUpdS);
    }

    for(int i=0; i<update_status.size(); i++)
    {
        std::pair<int, uint8_t> &p = update_status[i];
        int pt_idx = p.first;
        uint8_t val = p.second;
        v[pt_idx].status = val;
    }

    // update Jp
    if(H5Lexists(file.getId(), "UpdateJp", H5P_DEFAULT ) > 0)
    {
        H5::DataSet ds_jp = file.openDataSet("UpdateJp");
        hsize_t n;
        ds_jp.getSpace().getSimpleExtentDims(&n, NULL);
        update_Jp.resize(n);
        ds_jp.read(update_Jp.data(), VisualPoint::ctUpdJp);
    }

    for(int i=0; i<update_Jp.size(); i++)
    {
        std::pair<int, float> &p = update_Jp[i];
        int pt_idx = p.first;
        float Jp_inv = p.second;
        v[pt_idx].Jp_inv = Jp_inv;
    }
}


void Converter::save_points()
{
    spdlog::info("save_points {}; n= {}", frame, v.size());
    int n = v.size();
    points->SetNumberOfPoints(n);
    values_Jp->SetNumberOfValues(n);
    values_status->SetNumberOfValues(n);
    values_partitions->SetNumberOfValues(n);

    for(int i=0;i<n;i++)
    {
        VisualPoint &vp = v[i];
        int frame_difference = frame - last_pos_refresh_frame[i];
        Eigen::Vector3f pos = vp.getPos() + vp.getVel()*(dt*frame_difference*UpdateEveryNthStep);

        points->SetPoint(i, pos[0], pos[1], pos[2]);
        values_Jp->SetValue(i, vp.Jp_inv);
        values_status->SetValue(i, vp.getCrushedStatus());
        values_partitions->SetValue(i, vp.getPartition());
    }
    points->Modified();
    values_Jp->Modified();
    values_status->Modified();
    values_partitions->Modified();

    char pointsFileName[20];

    // INDENTER / cylinder shape
    snprintf(pointsFileName, sizeof(pointsFileName), "p_%05d.vtp", frame);
    std::string savePath = std::string(directory_output) + "/" + std::string(directory_points) + "/"+ pointsFileName;
    spdlog::info("writing points {}", savePath);

    writer1->SetFileName(savePath.c_str());
    writer1->Write();
}

void Converter::save_indenter()
{
    char indenterFileName[20];

    // INDENTER / cylinder shape
    snprintf(indenterFileName, sizeof(indenterFileName), "i_%05d.vtu", frame);
    std::string savePath = std::string(directory_output) + "/" + std::string(directory_indenter) + "/"+ indenterFileName;
    spdlog::info("writing indenger geometry {}", savePath);

    cylinder->SetRadius(IndDiameter/2.f);
    cylinder->SetHeight(GridZ*cellsize);
    double indenter_z = GridZ * cellsize/2;
    cylinder->SetCenter(indenter_x, indenter_z, -indenter_y);

    cylinder->Update();
    transformFilter->Update();
    appendFilter->Update();

    unstructuredGrid->ShallowCopy(appendFilter->GetOutput());

    // Write the unstructured grid.
    writer2->SetFileName(savePath.c_str());
    writer2->Write();
}

void Converter::save_tekscan()
{
    double h = cellsize;
//    double hsq = h*h;

    int nx = GridZ+1;
    int ny = IndenterSubdivisions*0.3+1;

    structuredGrid->SetDimensions(nx, ny, 1);
    grid_points->SetNumberOfPoints(nx*ny);
    for(int idx_y=0; idx_y<ny; idx_y++)
        for(int idx_x=0; idx_x<nx; idx_x++)
        {
            grid_points->SetPoint(idx_x+idx_y*nx, 0, idx_y*h, idx_x*h);
        }
    grid_points->Modified();

    values->SetNumberOfValues((nx-1)*(ny-1));
    for(int idx_y=0; idx_y<(ny-1); idx_y++)
        for(int idx_x=0; idx_x<(nx-1); idx_x++)
        {
            int idx = idx_x + GridZ*(IndenterSubdivisions-idx_y-1);
            Eigen::Map<Eigen::Vector3d> f(&indenter_data[3*idx]);
            float val = f.norm();
            values->SetValue((idx_x+idx_y*(nx-1)), val);
        }
    values->Modified();

    // Write the unstructured grid.
    char outputFileName[20];
    snprintf(outputFileName, sizeof(outputFileName), "t_%05d.vts", frame);
    std::string savePath = std::string(directory_output) + "/" + std::string(directory_sensor) + "/"+ outputFileName;
    spdlog::info("nx {}; ny {}; writing vts file for tekscan-like grid {}", nx, ny, savePath);

    writer3->SetFileName(savePath.c_str());
    writer3->Modified();
    writer3->Update();
    writer3->Write();
}

void Converter::save_bgeo()
{
    spdlog::info("export_bgeo frame {}", frame);

    Partio::ParticlesDataMutable* parts = Partio::create();
    Partio::ParticleAttribute attr_Jp = parts->addAttribute("Jp_inv", Partio::FLOAT, 1);
    Partio::ParticleAttribute attr_pos = parts->addAttribute("position", Partio::VECTOR, 3);

    parts->addParticles(v.size());
    for(int i=0;i<v.size();i++)
    {
        VisualPoint &vp = v[i];
        float* val = parts->dataWrite<float>(attr_pos, i);
        for(int j=0;j<3;j++) val[j] = vp.pos[j];
        float *val_Jp = parts->dataWrite<float>(attr_Jp, i);
        val_Jp[0] = vp.Jp_inv;
    }

    char fileName[20];
    snprintf(fileName, sizeof(fileName), "%05d.bgeo", frame);
    std::string savePath = std::string(directory_output) + "/" + std::string(directory_bgeo) + "/" + fileName;
    spdlog::info("writing bgeo file {}", savePath);
    Partio::write(savePath.c_str(), *parts);
    parts->release();
    spdlog::info("export_bgeo frame done {}", frame);
}



void Converter::read_file(std::string fullFilePath)
{
    H5::H5File file(fullFilePath, H5F_ACC_RDONLY);
    H5::DataSet dataset_indenter = file.openDataSet("Indenter");
    H5::Attribute att_partial_frame = dataset_indenter.openAttribute("partial_frame");
    uint8_t prev_frame;
    att_partial_frame.read(H5::PredType::NATIVE_UINT8, &prev_frame);
    if(frame == frame_start && prev_frame) throw std::runtime_error("partial frame instead of a full frame");

    H5::Attribute att_indx = dataset_indenter.openAttribute("indenter_x");
    att_indx.read(H5::PredType::NATIVE_DOUBLE, &indenter_x);
    H5::Attribute att_indy = dataset_indenter.openAttribute("indenter_y");
    att_indy.read(H5::PredType::NATIVE_DOUBLE, &indenter_y);
    H5::Attribute att_time = dataset_indenter.openAttribute("SimulationTime");
    att_time.read(H5::PredType::NATIVE_DOUBLE, &SimulationTime);


    if(!prev_frame)
        read_full_frame(file, dataset_indenter);
    else
        read_partial_frame(file);

    dataset_indenter.read(indenter_data.data(), H5::PredType::NATIVE_DOUBLE);
    file.close();
}


void Converter::save_indenter_total()
{
    double force[5] {};
    for(int j=0; j<IndenterSubdivisions*GridZ; j++)
        for(int k=0;k<3;k++)
        {
            int idx = j*3+k;
            force[k+1] += indenter_data[idx];
        }
    double hsq = cellsize*cellsize;
    force[0] = SimulationTime;
    force[1] *= hsq;
    force[2] *= hsq;
    force[3] *= hsq;
    force[4] = sqrt(force[1]*force[1]+force[2]*force[2]+force[3]*force[3]);

    // write
    accessing_indenter_force_file->lock();
    hsize_t dims[2] = {(hsize_t)frames_total, 5};
    H5::DataSpace dataspace(2, dims);

    hsize_t count[2] = {1, 5};  // Dimensions of the hyperslab
    hsize_t offset[2] = {(hsize_t)frame, 0};
    dataspace.selectHyperslab(H5S_SELECT_SET, count, offset);

    hsize_t dims_mem[2] = {1,5};
    H5::DataSpace dataspace_mem(2, dims_mem);

    dataset_indenter_totals->write(force, H5::PredType::NATIVE_DOUBLE, dataspace_mem, dataspace);
    accessing_indenter_force_file->unlock();
}



void Converter::process_subset(const int _frame_start, int count, std::string directory,
                               bool bgeo, bool paraview, bool paraview_intact)
{
    this->frame_start = _frame_start;
    spdlog::info("process_subset; frame_start {}; count {}", frame_start, count);

    for(frame=frame_start; frame<frame_start+count; frame++)
    {
        snprintf(fileName, sizeof(fileName), "v%05d.h5", frame);
        std::string fullFilePath = directory + "/" + fileName;
        spdlog::info("reading frame {} from file {}", frame, fullFilePath);
        read_file(fullFilePath);

        if(paraview)
        {
            save_points();
            save_indenter();
            save_tekscan();
        }

        if(paraview_intact) save_points_intact();

        if(bgeo) save_bgeo();
        save_indenter_total();
    }
}

void Converter::save_points_intact()
{
    spdlog::info("save_points {}; n= {}", frame, v.size());
    int n = v.size();

    points->SetNumberOfPoints(0);
    values_Jp->SetNumberOfValues(0);
    values_status->SetNumberOfValues(0);
    values_partitions->SetNumberOfValues(0);

    for(int i=0;i<n;i++)
    {
        VisualPoint &vp = v[i];
        int frame_difference = frame - last_pos_refresh_frame[i];
        Eigen::Vector3f pos = vp.getPos() + vp.getVel()*(dt*frame_difference*UpdateEveryNthStep);

        if(vp.getCrushedStatus()) continue;
        points->InsertNextPoint(pos[0], pos[1], pos[2]);
        values_Jp->InsertNextValue(vp.Jp_inv);
        values_status->InsertNextValue(vp.getCrushedStatus());
        values_partitions->InsertNextValue(vp.getPartition());
    }
    points->Modified();
    values_Jp->Modified();
    values_status->Modified();
    values_partitions->Modified();

    char pointsFileName[20];

    // INDENTER / cylinder shape
    snprintf(pointsFileName, sizeof(pointsFileName), "ip_%05d.vtp", frame);
    std::string savePath = std::string(directory_output) + "/" + std::string(directory_points_intact) + "/"+ pointsFileName;
    spdlog::info("writing points {}", savePath);

    writer1->SetFileName(savePath.c_str());
    writer1->Write();
}

