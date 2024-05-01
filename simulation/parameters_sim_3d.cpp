#include "parameters_sim_3d.h"
#include <spdlog/spdlog.h>


void SimParams3D::Reset()
{
    nPtsTotal = 0;
    SimulationEndTime = 12;

    SimulationStep = 0;
    SimulationTime = 0;

    InitialTimeStep = 3.e-5;
    YoungsModulus = 5.e8;
    GridXTotal = 128;
    GridY = 55;
    GridZ = 55;
    ParticleViewSize = 2.5f;
    DomainDimensionX = 3.33;

    PoissonsRatio = 0.3;
    Gravity = 9.81;
    Density = 980;

    IndDiameter = 0.324;
    IndVelocity = 0.2;
    IndDepth = 0.25;//0.101;

    IceCompressiveStrength = 100e6;
    IceTensileStrength = 10e6;
    IceShearStrength = 6e6;
    IceTensileStrength2 = 5e6;

    DP_tan_phi = std::tan(70*pi/180.);
    DP_threshold_p = 1e2;

    tpb_P2G = 256;
    tpb_Upd = 512;
    tpb_G2P = 128;

    indenter_x = indenter_x_initial = indenter_y = indenter_y_initial = 0;
    SetupType = 0;
    GrainVariability = 0.50;

    GridHaloSize = 15;
    ExtraSpaceForIncomingPoints = 0.2;  // in percentage
    PointsTransferBufferFraction = 0.05; // % of points that could "fly over" during a given cycle
    nPartitions = 4;        // one partition of single-gpu; >1 for multi-gpu

    RebalanceThresholdFreeSpaceRemaining = 0.10;
    RebalanceThresholdDisabledPercentage = 0.05;
    PointTransferFrequency = 2;
    SnapshotPeriod = 100;

    ComputeLame();
    ComputeCamClayParams();
    ComputeHelperVariables();
    spdlog::info("SimParams reset; nPtsArrays {}; sizeof(SimParams3D) {}", nPtsArrays, sizeof(SimParams3D));
}


void SimParams3D::ComputeHelperVariables()
{
    UpdateEveryNthStep = (int)(1.f/(200*InitialTimeStep));
    cellsize = DomainDimensionX/GridXTotal;
    cellsize_inv = 1./cellsize;
    Dp_inv = 4./(cellsize*cellsize);
    IndRSq = IndDiameter*IndDiameter/4.;
    dt_vol_Dpinv = InitialTimeStep*ParticleVolume*Dp_inv;
    dt_Gravity = InitialTimeStep*Gravity;
    vmax = 0.5*cellsize/InitialTimeStep;
    vmax_squared = vmax*vmax;

    VectorCapacity_transfer = nPtsTotal/nPartitions * PointsTransferBufferFraction;

    gbOffset = GridZ * GridY * GridHaloSize;

    IndenterSubdivisions = (int)(pi*IndDiameter / cellsize); // approximately one sensel per grid cell
}


std::string SimParams3D::ParseFile(std::string fileName)
{
    Reset();
    spdlog::info("SimParams3D ParseFile {}",fileName);
    if(!std::filesystem::exists(fileName)) throw std::runtime_error("configuration file is not found");
    std::ifstream fileStream(fileName);
    std::string strConfigFile;
    strConfigFile.resize(std::filesystem::file_size(fileName));
    fileStream.read(strConfigFile.data(), strConfigFile.length());
    fileStream.close();

    rapidjson::Document doc;
    doc.Parse(strConfigFile.data());
    if(!doc.IsObject()) throw std::runtime_error("configuration file is not JSON");

    if(doc.HasMember("InitialTimeStep")) InitialTimeStep = doc["InitialTimeStep"].GetDouble();
    if(doc.HasMember("YoungsModulus")) YoungsModulus = doc["YoungsModulus"].GetDouble();
    if(doc.HasMember("GridX")) GridXTotal = doc["GridX"].GetInt();
    if(doc.HasMember("GridY")) GridY = doc["GridY"].GetInt();
    if(doc.HasMember("GridZ")) GridZ = doc["GridZ"].GetInt();
    if(doc.HasMember("DomainDimensionX")) DomainDimensionX = doc["DomainDimensionX"].GetDouble();
    if(doc.HasMember("ParticleViewSize")) ParticleViewSize = doc["ParticleViewSize"].GetDouble();
    if(doc.HasMember("SimulationEndTime")) SimulationEndTime = doc["SimulationEndTime"].GetDouble();
    if(doc.HasMember("PoissonsRatio")) PoissonsRatio = doc["PoissonsRatio"].GetDouble();
    if(doc.HasMember("Gravity")) Gravity = doc["Gravity"].GetDouble();
    if(doc.HasMember("Density")) Density = doc["Density"].GetDouble();
    if(doc.HasMember("IndDiameter")) IndDiameter = doc["IndDiameter"].GetDouble();
    if(doc.HasMember("IndVelocity")) IndVelocity = doc["IndVelocity"].GetDouble();
    if(doc.HasMember("IndDepth")) IndDepth = doc["IndDepth"].GetDouble();

    if(doc.HasMember("IceCompressiveStrength")) IceCompressiveStrength = doc["IceCompressiveStrength"].GetDouble();
    if(doc.HasMember("IceTensileStrength")) IceTensileStrength = doc["IceTensileStrength"].GetDouble();
    if(doc.HasMember("IceTensileStrength2")) IceTensileStrength2 = doc["IceTensileStrength2"].GetDouble();
    if(doc.HasMember("IceShearStrength")) IceShearStrength = doc["IceShearStrength"].GetDouble();

    if(doc.HasMember("DP_phi")) DP_tan_phi = std::tan(doc["DP_phi"].GetDouble()*pi/180);
    if(doc.HasMember("DP_threshold_p")) DP_threshold_p = doc["DP_threshold_p"].GetDouble();
    if(doc.HasMember("GrainVariability")) GrainVariability = doc["GrainVariability"].GetDouble();

    if(doc.HasMember("tpb_P2G")) tpb_P2G = doc["tpb_P2G"].GetInt();
    if(doc.HasMember("tpb_Upd")) tpb_Upd = doc["tpb_Upd"].GetInt();
    if(doc.HasMember("tpb_G2P")) tpb_G2P = doc["tpb_G2P"].GetInt();
    if(doc.HasMember("GridHaloSize")) GridHaloSize = doc["GridHaloSize"].GetInt();
    if(doc.HasMember("nPartitions")) nPartitions = doc["nPartitions"].GetInt();

    if(doc.HasMember("ExtraSpaceForIncomingPoints")) ExtraSpaceForIncomingPoints = doc["ExtraSpaceForIncomingPoints"].GetDouble();
    if(doc.HasMember("RebalanceThresholdFreeSpaceRemaining")) RebalanceThresholdFreeSpaceRemaining = doc["RebalanceThresholdFreeSpaceRemaining"].GetDouble();
    if(doc.HasMember("RebalanceThresholdDisabledPercentage")) RebalanceThresholdDisabledPercentage = doc["RebalanceThresholdDisabledPercentage"].GetDouble();
    if(doc.HasMember("PointsTransferBufferFraction")) PointsTransferBufferFraction = doc["PointsTransferBufferFraction"].GetDouble();

    ComputeCamClayParams();
    ComputeHelperVariables();

    if(!doc.HasMember("InputRawPoints"))
    {
        spdlog::critical("InputRawPoints entry is missing in JSON config file");
        throw std::runtime_error("config parameter missing");
    }

    std::string result = doc["InputRawPoints"].GetString();
    spdlog::info("Loaded parameters; grid [{} x {} x {}] pointFile {}",GridXTotal, GridY, GridZ, result);
    return result;
}

void SimParams3D::ComputeLame()
{
    lambda = YoungsModulus*PoissonsRatio/((1+PoissonsRatio)*(1-2*PoissonsRatio));
    mu = YoungsModulus/(2*(1+PoissonsRatio));
    kappa = mu*2./3. + lambda;
}

void SimParams3D::ComputeCamClayParams()
{
    ComputeLame();
    NACC_beta = IceTensileStrength/IceCompressiveStrength;
}


void SimParams3D::SaveParametersAsAttributes(H5::DataSet &ds)
{
    hsize_t att_dim = 1;
    H5::DataSpace att_dspace(1, &att_dim);

    H5::Attribute att_SetupType = ds.createAttribute("SetupType", H5::PredType::NATIVE_INT, att_dspace);
    att_SetupType.write(H5::PredType::NATIVE_INT, &SetupType);

    // points, grid, and indenter
    H5::Attribute att_nPtsTotal = ds.createAttribute("nPtsTotal", H5::PredType::NATIVE_INT, att_dspace);
    H5::Attribute att_GridX = ds.createAttribute("GridXTotal", H5::PredType::NATIVE_INT, att_dspace);
    H5::Attribute att_GridY = ds.createAttribute("GridY", H5::PredType::NATIVE_INT, att_dspace);
    H5::Attribute att_GridZ = ds.createAttribute("GridZ", H5::PredType::NATIVE_INT, att_dspace);
    H5::Attribute att_cellsize = ds.createAttribute("cellsize", H5::PredType::NATIVE_DOUBLE, att_dspace);
    H5::Attribute att_IndenterSubdivisions = ds.createAttribute("IndenterSubdivisions", H5::PredType::NATIVE_INT, att_dspace);

    att_nPtsTotal.write(H5::PredType::NATIVE_INT, &nPtsTotal);
    att_GridX.write(H5::PredType::NATIVE_INT, &GridXTotal);
    att_GridY.write(H5::PredType::NATIVE_INT, &GridY);
    att_GridZ.write(H5::PredType::NATIVE_INT, &GridZ);
    att_cellsize.write(H5::PredType::NATIVE_DOUBLE, &cellsize);
    att_IndenterSubdivisions.write(H5::PredType::NATIVE_INT, &IndenterSubdivisions);

    // time
    H5::Attribute att_InitialTimeStep = ds.createAttribute("InitialTimeStep", H5::PredType::NATIVE_DOUBLE, att_dspace);
    H5::Attribute att_SimulationEndTime = ds.createAttribute("SimulationEndTime", H5::PredType::NATIVE_DOUBLE, att_dspace);
    H5::Attribute att_SimulationTime = ds.createAttribute("SimulationTime", H5::PredType::NATIVE_DOUBLE, att_dspace);
    H5::Attribute att_UpdateEveryNthStep = ds.createAttribute("UpdateEveryNthStep", H5::PredType::NATIVE_INT, att_dspace);
    H5::Attribute att_SnapshotPeriod = ds.createAttribute("SnapshotPeriod", H5::PredType::NATIVE_INT, att_dspace);
    H5::Attribute att_SimulationStep = ds.createAttribute("SimulationStep", H5::PredType::NATIVE_INT, att_dspace);
    att_InitialTimeStep.write(H5::PredType::NATIVE_DOUBLE, &InitialTimeStep);
    att_SimulationEndTime.write(H5::PredType::NATIVE_DOUBLE, &SimulationEndTime);
    att_SimulationTime.write(H5::PredType::NATIVE_DOUBLE, &SimulationTime);
    att_UpdateEveryNthStep.write(H5::PredType::NATIVE_INT, &UpdateEveryNthStep);
    att_SnapshotPeriod.write(H5::PredType::NATIVE_INT, &SnapshotPeriod);
    att_SimulationStep.write(H5::PredType::NATIVE_INT, &SimulationStep);

    // physical parameters
    H5::Attribute att_Gravity = ds.createAttribute("Gravity", H5::PredType::NATIVE_DOUBLE, att_dspace);
    H5::Attribute att_Density = ds.createAttribute("Density", H5::PredType::NATIVE_DOUBLE, att_dspace);
    H5::Attribute att_PoissonsRatio = ds.createAttribute("PoissonsRatio", H5::PredType::NATIVE_DOUBLE, att_dspace);
    H5::Attribute att_YoungsModulus = ds.createAttribute("YoungsModulus", H5::PredType::NATIVE_DOUBLE, att_dspace);
    H5::Attribute att_IceCompressiveStrength = ds.createAttribute("IceCompressiveStrength", H5::PredType::NATIVE_DOUBLE, att_dspace);
    H5::Attribute att_IceTensileStrength = ds.createAttribute("IceTensileStrength", H5::PredType::NATIVE_DOUBLE, att_dspace);
    H5::Attribute att_IceTensileStrength2 = ds.createAttribute("IceTensileStrength2", H5::PredType::NATIVE_DOUBLE, att_dspace);
    H5::Attribute att_IceShearStrength = ds.createAttribute("IceShearStrength", H5::PredType::NATIVE_DOUBLE, att_dspace);
    H5::Attribute att_DP_tan_phi = ds.createAttribute("DP_tan_phi", H5::PredType::NATIVE_DOUBLE, att_dspace);
    H5::Attribute att_DP_threshold_p = ds.createAttribute("DP_threshold_p", H5::PredType::NATIVE_DOUBLE, att_dspace);
    H5::Attribute att_GrainVariability = ds.createAttribute("GrainVariability", H5::PredType::NATIVE_DOUBLE, att_dspace);

    att_Gravity.write(H5::PredType::NATIVE_DOUBLE, &Gravity);
    att_Density.write(H5::PredType::NATIVE_DOUBLE, &Density);
    att_PoissonsRatio.write(H5::PredType::NATIVE_DOUBLE, &PoissonsRatio);
    att_YoungsModulus.write(H5::PredType::NATIVE_DOUBLE, &YoungsModulus);
    att_IceCompressiveStrength.write(H5::PredType::NATIVE_DOUBLE, &IceCompressiveStrength);
    att_IceTensileStrength.write(H5::PredType::NATIVE_DOUBLE, &IceTensileStrength);
    att_IceTensileStrength2.write(H5::PredType::NATIVE_DOUBLE, &IceTensileStrength2);
    att_IceShearStrength.write(H5::PredType::NATIVE_DOUBLE, &IceShearStrength);
    att_DP_tan_phi.write(H5::PredType::NATIVE_DOUBLE, &DP_tan_phi);
    att_GrainVariability.write(H5::PredType::NATIVE_DOUBLE, &GrainVariability);

    // indenter: location, diameter, and velocity
    H5::Attribute att_indenter_x = ds.createAttribute("indenter_x", H5::PredType::NATIVE_DOUBLE, att_dspace);
    H5::Attribute att_indenter_y = ds.createAttribute("indenter_y", H5::PredType::NATIVE_DOUBLE, att_dspace);
    H5::Attribute att_indenter_x_initial = ds.createAttribute("indenter_x_initial", H5::PredType::NATIVE_DOUBLE, att_dspace);
    H5::Attribute att_indenter_y_initial = ds.createAttribute("indenter_y_initial", H5::PredType::NATIVE_DOUBLE, att_dspace);
    H5::Attribute att_IndDiameter = ds.createAttribute("IndDiameter", H5::PredType::NATIVE_DOUBLE, att_dspace);
    H5::Attribute att_IndVelocity = ds.createAttribute("IndVelocity", H5::PredType::NATIVE_DOUBLE, att_dspace);

    att_indenter_x.write(H5::PredType::NATIVE_DOUBLE, &indenter_x);
    att_indenter_y.write(H5::PredType::NATIVE_DOUBLE, &indenter_y);
    att_indenter_x_initial.write(H5::PredType::NATIVE_DOUBLE, &indenter_x_initial);
    att_indenter_y_initial.write(H5::PredType::NATIVE_DOUBLE, &indenter_y_initial);
    att_IndDiameter.write(H5::PredType::NATIVE_DOUBLE, &IndDiameter);
    att_IndVelocity.write(H5::PredType::NATIVE_DOUBLE, &IndVelocity);


    // parameters of the points / total volume of the solid
    H5::Attribute att_ParticleVolume = ds.createAttribute("ParticleVolume", H5::PredType::NATIVE_DOUBLE, att_dspace);
    H5::Attribute att_ParticleMass = ds.createAttribute("ParticleMass", H5::PredType::NATIVE_DOUBLE, att_dspace);
    H5::Attribute att_Volume = ds.createAttribute("Volume", H5::PredType::NATIVE_DOUBLE, att_dspace);

    att_ParticleVolume.write(H5::PredType::NATIVE_DOUBLE, &ParticleVolume);
    att_ParticleMass.write(H5::PredType::NATIVE_DOUBLE, &ParticleMass);
    att_Volume.write(H5::PredType::NATIVE_DOUBLE, &Volume);


    // multi-GPU parameters
    H5::Attribute att_nPartitions = ds.createAttribute("nPartitions", H5::PredType::NATIVE_INT, att_dspace);
    H5::Attribute att_GridHaloSize = ds.createAttribute("GridHaloSize", H5::PredType::NATIVE_INT, att_dspace);
    H5::Attribute att_PointTransferFrequency = ds.createAttribute("PointTransferFrequency", H5::PredType::NATIVE_INT, att_dspace);
    H5::Attribute att_ExtraSpaceForIncomingPoints = ds.createAttribute("ExtraSpaceForIncomingPoints", H5::PredType::NATIVE_DOUBLE, att_dspace);
    H5::Attribute att_PointsTransferBufferFraction = ds.createAttribute("PointsTransferBufferFraction", H5::PredType::NATIVE_DOUBLE, att_dspace);
    H5::Attribute att_RebalanceThresholdFreeSpaceRemaining = ds.createAttribute("RebalanceThresholdFreeSpaceRemaining", H5::PredType::NATIVE_DOUBLE, att_dspace);
    H5::Attribute att_RebalanceThresholdDisabledPercentage = ds.createAttribute("RebalanceThresholdDisabledPercentage", H5::PredType::NATIVE_DOUBLE, att_dspace);

    att_nPartitions.write(H5::PredType::NATIVE_INT, &nPartitions);
    att_GridHaloSize.write(H5::PredType::NATIVE_INT, &GridHaloSize);
    att_PointTransferFrequency.write(H5::PredType::NATIVE_INT, &PointTransferFrequency);
    att_ExtraSpaceForIncomingPoints.write(H5::PredType::NATIVE_DOUBLE, &ExtraSpaceForIncomingPoints);
    att_PointsTransferBufferFraction.write(H5::PredType::NATIVE_DOUBLE, &PointsTransferBufferFraction);
    att_RebalanceThresholdFreeSpaceRemaining.write(H5::PredType::NATIVE_DOUBLE, &RebalanceThresholdFreeSpaceRemaining);
    att_RebalanceThresholdDisabledPercentage.write(H5::PredType::NATIVE_DOUBLE, &RebalanceThresholdDisabledPercentage);
}


void SimParams3D::ReadParametersFromAttributes(H5::DataSet &ds)
{
    Reset();

    H5::Attribute att_SetupType = ds.openAttribute("SetupType");
    att_SetupType.read(H5::PredType::NATIVE_INT, &SetupType);

    // points, grid, and indenter
    H5::Attribute att_nPtsTotal = ds.openAttribute("nPtsTotal");
    H5::Attribute att_GridX = ds.openAttribute("GridXTotal");
    H5::Attribute att_GridY = ds.openAttribute("GridY");
    H5::Attribute att_GridZ = ds.openAttribute("GridZ");
    H5::Attribute att_cellsize = ds.openAttribute("cellsize");
    H5::Attribute att_IndenterSubdivisions = ds.openAttribute("IndenterSubdivisions");

    att_nPtsTotal.read(H5::PredType::NATIVE_INT, &nPtsTotal);
    att_GridX.read(H5::PredType::NATIVE_INT, &GridXTotal);
    att_GridY.read(H5::PredType::NATIVE_INT, &GridY);
    att_GridZ.read(H5::PredType::NATIVE_INT, &GridZ);
    att_cellsize.read(H5::PredType::NATIVE_DOUBLE, &cellsize);
    att_IndenterSubdivisions.read(H5::PredType::NATIVE_INT, &IndenterSubdivisions);

    DomainDimensionX = GridXTotal * cellsize;

    // time
    H5::Attribute att_InitialTimeStep = ds.openAttribute("InitialTimeStep");
    H5::Attribute att_SimulationEndTime = ds.openAttribute("SimulationEndTime");
    H5::Attribute att_SimulationTime = ds.openAttribute("SimulationTime");
    H5::Attribute att_UpdateEveryNthStep = ds.openAttribute("UpdateEveryNthStep");
    H5::Attribute att_SnapshotPeriod = ds.openAttribute("SnapshotPeriod");
    H5::Attribute att_SimulationStep = ds.openAttribute("SimulationStep");
    att_InitialTimeStep.read(H5::PredType::NATIVE_DOUBLE, &InitialTimeStep);
    att_SimulationEndTime.read(H5::PredType::NATIVE_DOUBLE, &SimulationEndTime);
    att_SimulationTime.read(H5::PredType::NATIVE_DOUBLE, &SimulationTime);
    att_UpdateEveryNthStep.read(H5::PredType::NATIVE_INT, &UpdateEveryNthStep);
    att_SnapshotPeriod.read(H5::PredType::NATIVE_INT, &SnapshotPeriod);
    att_SimulationStep.read(H5::PredType::NATIVE_INT, &SimulationStep);

    // physical parameters
    H5::Attribute att_Gravity = ds.openAttribute("Gravity");
    H5::Attribute att_Density = ds.openAttribute("Density");
    H5::Attribute att_PoissonsRatio = ds.openAttribute("PoissonsRatio");
    H5::Attribute att_YoungsModulus = ds.openAttribute("YoungsModulus");
    H5::Attribute att_IceCompressiveStrength = ds.openAttribute("IceCompressiveStrength");
    H5::Attribute att_IceTensileStrength = ds.openAttribute("IceTensileStrength");
    H5::Attribute att_IceTensileStrength2 = ds.openAttribute("IceTensileStrength2");
    H5::Attribute att_IceShearStrength = ds.openAttribute("IceShearStrength");
    H5::Attribute att_DP_tan_phi = ds.openAttribute("DP_tan_phi");
    H5::Attribute att_DP_threshold_p = ds.openAttribute("DP_threshold_p");
    H5::Attribute att_GrainVariability = ds.openAttribute("GrainVariability");

    att_Gravity.read(H5::PredType::NATIVE_DOUBLE, &Gravity);
    att_Density.read(H5::PredType::NATIVE_DOUBLE, &Density);
    att_PoissonsRatio.read(H5::PredType::NATIVE_DOUBLE, &PoissonsRatio);
    att_YoungsModulus.read(H5::PredType::NATIVE_DOUBLE, &YoungsModulus);
    att_IceCompressiveStrength.read(H5::PredType::NATIVE_DOUBLE, &IceCompressiveStrength);
    att_IceTensileStrength.read(H5::PredType::NATIVE_DOUBLE, &IceTensileStrength);
    att_IceTensileStrength2.read(H5::PredType::NATIVE_DOUBLE, &IceTensileStrength2);
    att_IceShearStrength.read(H5::PredType::NATIVE_DOUBLE, &IceShearStrength);
    att_DP_tan_phi.read(H5::PredType::NATIVE_DOUBLE, &DP_tan_phi);
    att_GrainVariability.read(H5::PredType::NATIVE_DOUBLE, &GrainVariability);


    // indenter: location, diameter, and velocity
    H5::Attribute att_indenter_x = ds.openAttribute("indenter_x");
    H5::Attribute att_indenter_y = ds.openAttribute("indenter_y");
    H5::Attribute att_indenter_x_initial = ds.openAttribute("indenter_x_initial");
    H5::Attribute att_indenter_y_initial = ds.openAttribute("indenter_y_initial");
    H5::Attribute att_IndDiameter = ds.openAttribute("IndDiameter");
    H5::Attribute att_IndVelocity = ds.openAttribute("IndVelocity");

    att_indenter_x.read(H5::PredType::NATIVE_DOUBLE, &indenter_x);
    att_indenter_y.read(H5::PredType::NATIVE_DOUBLE, &indenter_y);
    att_indenter_x_initial.read(H5::PredType::NATIVE_DOUBLE, &indenter_x_initial);
    att_indenter_y_initial.read(H5::PredType::NATIVE_DOUBLE, &indenter_y_initial);
    att_IndDiameter.read(H5::PredType::NATIVE_DOUBLE, &IndDiameter);
    att_IndVelocity.read(H5::PredType::NATIVE_DOUBLE, &IndVelocity);


    // parameters of the points / total volume of the solid
    H5::Attribute att_ParticleVolume = ds.openAttribute("ParticleVolume");
    H5::Attribute att_ParticleMass = ds.openAttribute("ParticleMass");
    H5::Attribute att_Volume = ds.openAttribute("Volume");

    att_ParticleVolume.read(H5::PredType::NATIVE_DOUBLE, &ParticleVolume);
    att_ParticleMass.read(H5::PredType::NATIVE_DOUBLE, &ParticleMass);
    att_Volume.read(H5::PredType::NATIVE_DOUBLE, &Volume);


    // multi-GPU parameters
    H5::Attribute att_nPartitions = ds.openAttribute("nPartitions");
    H5::Attribute att_GridHaloSize = ds.openAttribute("GridHaloSize");
    H5::Attribute att_PointTransferFrequency = ds.openAttribute("PointTransferFrequency");
    H5::Attribute att_ExtraSpaceForIncomingPoints = ds.openAttribute("ExtraSpaceForIncomingPoints");
    H5::Attribute att_PointsTransferBufferFraction = ds.openAttribute("PointsTransferBufferFraction");
    H5::Attribute att_RebalanceThresholdFreeSpaceRemaining = ds.openAttribute("RebalanceThresholdFreeSpaceRemaining");
    H5::Attribute att_RebalanceThresholdDisabledPercentage = ds.openAttribute("RebalanceThresholdDisabledPercentage");

    att_nPartitions.read(H5::PredType::NATIVE_INT, &nPartitions);
    att_GridHaloSize.read(H5::PredType::NATIVE_INT, &GridHaloSize);
    att_PointTransferFrequency.read(H5::PredType::NATIVE_INT, &PointTransferFrequency);
    att_ExtraSpaceForIncomingPoints.read(H5::PredType::NATIVE_DOUBLE, &ExtraSpaceForIncomingPoints);
    att_PointsTransferBufferFraction.read(H5::PredType::NATIVE_DOUBLE, &PointsTransferBufferFraction);
    att_RebalanceThresholdFreeSpaceRemaining.read(H5::PredType::NATIVE_DOUBLE, &RebalanceThresholdFreeSpaceRemaining);
    att_RebalanceThresholdDisabledPercentage.read(H5::PredType::NATIVE_DOUBLE, &RebalanceThresholdDisabledPercentage);

    ComputeCamClayParams();
    ComputeHelperVariables();
}
