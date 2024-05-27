#include <iostream>
#include <functional>
#include <string>
#include <filesystem>
#include <chrono>
#include <mutex>

#include <cxxopts.hpp>
#include <spdlog/spdlog.h>
#include <omp.h>
#include <H5Cpp.h>

#include "visualpoint.h"
#include "converter.h"


constexpr int block_size = 100;


void process_subset(int frame_start, int count, std::string directory, bool bgeo, bool paraview);

// -t 5 -p -d /media/s2/Archive1/MPM_2023/simulation3/_snapshots_animation

int main(int argc, char** argv)
{
#pragma omp parallel
    { std::cout << omp_get_thread_num(); }
    std::cout << std::endl;

    // parse options
    cxxopts::Options options("Converter", "A tool to convert .h5 frames to Paraview and/or Bgeo format");

    options.add_options()
        ("d,directory", "Directory with frames in HDF5 format", cxxopts::value<std::string>())
        ("s,startframe", "Start frame frame to convert (must be full frame)", cxxopts::value<int>()->default_value("0"))
        ("e,endframe", "End frame frame to convert", cxxopts::value<int>()->default_value("2399"))
        ("t,threads", "Number of threads to run in parallel", cxxopts::value<int>()->default_value("1"))
        ("b,bgeo", "Export as BGEO", cxxopts::value<bool>())
        ("p,paraview", "Export for Paraview", cxxopts::value<bool>())
        ("i,intact", "Export points for Paraview, only intact material", cxxopts::value<bool>())
        ("m,damaged", "Export for Paraview damaged material only", cxxopts::value<bool>())
        ("a,all", "Sets options 'paraview', 'intact', and 'damaged'", cxxopts::value<bool>())
        ;

    options.parse_positional({"directory"});
    auto option_parse_result = options.parse(argc, argv);

    std::string frames_directory = option_parse_result["directory"].as<std::string>();

    bool export_bgeo = option_parse_result.count("bgeo");
    bool export_paraview = option_parse_result.count("paraview");
    bool export_intact = option_parse_result.count("intact");
    bool export_damaged = option_parse_result.count("damaged");
    bool export_all = option_parse_result.count("all");
    if(export_all) export_paraview = export_intact = export_damaged = true;
    int endframe = option_parse_result["endframe"].as<int>();
    int startframe = option_parse_result["startframe"].as<int>();
    int count_threads = option_parse_result["threads"].as<int>();
    spdlog::info("startframe {}; endframe {}; threads {}; paraview {}; bgeo {}", startframe, endframe,
                 count_threads, export_paraview, export_bgeo);

    std::filesystem::path od(Converter::directory_output);
    std::filesystem::path od1(std::string(Converter::directory_output) + "/" + std::string(Converter::directory_bgeo));
    std::filesystem::path od2(std::string(Converter::directory_output) + "/" + std::string(Converter::directory_points));
    std::filesystem::path od2b(std::string(Converter::directory_output) + "/" + std::string(Converter::directory_points_intact));
    std::filesystem::path od2c(std::string(Converter::directory_output) + "/" + std::string(Converter::directory_points_damaged));
    std::filesystem::path od3(std::string(Converter::directory_output) + "/" + std::string(Converter::directory_indenter));
    std::filesystem::path od3b(std::string(Converter::directory_output) + "/" + std::string(Converter::directory_indenter_hdf5));
    std::filesystem::path od4(std::string(Converter::directory_output) + "/" + std::string(Converter::directory_sensor));
    if(!std::filesystem::is_directory(od) || !std::filesystem::exists(od)) std::filesystem::create_directory(od);
    if(!std::filesystem::is_directory(od1) || !std::filesystem::exists(od1)) std::filesystem::create_directory(od1);
    if(!std::filesystem::is_directory(od2) || !std::filesystem::exists(od2)) std::filesystem::create_directory(od2);
    if(!std::filesystem::is_directory(od2b) || !std::filesystem::exists(od2b)) std::filesystem::create_directory(od2b);
    if(!std::filesystem::is_directory(od2c) || !std::filesystem::exists(od2c)) std::filesystem::create_directory(od2c);
    if(!std::filesystem::is_directory(od3) || !std::filesystem::exists(od3)) std::filesystem::create_directory(od3);
    if(!std::filesystem::is_directory(od3b) || !std::filesystem::exists(od3b)) std::filesystem::create_directory(od3b);
    if(!std::filesystem::is_directory(od4) || !std::filesystem::exists(od4)) std::filesystem::create_directory(od4);

    VisualPoint::InitializeStatic();
    std::mutex accessing_indenter_force_file;
    Converter::accessing_indenter_force_file = &accessing_indenter_force_file;

    H5::H5File file(std::string(Converter::directory_output) + "/indenter.h5", H5F_ACC_TRUNC);
    hsize_t dims_indenter[2] = {(hsize_t)endframe+1,5};
    H5::DataSpace dataspace_indenter(2, dims_indenter);
    H5::DataSet dataset_indenter = file.createDataSet("IndenterTotals", H5::PredType::NATIVE_DOUBLE, dataspace_indenter);
    Converter::dataset_indenter_totals = &dataset_indenter;
    Converter::frames_total = endframe+1;

    // open the first frame and get indenter dimensions
    char fileName[20];
    snprintf(fileName, sizeof(fileName), "v%05d.h5", startframe);
    std::string fullFilePath = frames_directory + "/" + fileName;

    H5::H5File file2(fullFilePath, H5F_ACC_RDONLY);
    H5::DataSet dataset_indenter2 = file2.openDataSet("Indenter");
    H5::Attribute att_GridZ = dataset_indenter2.openAttribute("GridZ");
    H5::Attribute att_IndenterSubdividions = dataset_indenter2.openAttribute("IndenterSubdivisions");
    int gridz, indentersubdivisions;
    att_GridZ.read(H5::PredType::NATIVE_INT, &gridz);
    att_IndenterSubdividions.read(H5::PredType::NATIVE_INT, &indentersubdivisions);
    file2.close();

    // create a summarized one-file indenter pressure table
    hsize_t tekscan[3] = {(hsize_t)endframe+1,(hsize_t)indentersubdivisions, (hsize_t)gridz};
    hsize_t chunk[3] = {(hsize_t)10, (hsize_t)indentersubdivisions/10, (hsize_t)gridz};
    H5::DataSpace dataspace_tekscan(3, tekscan);
    H5::DSetCreatPropList proplist2;
    proplist2.setChunk(3, chunk);
    proplist2.setDeflate(7);
    H5::DataSet dataset_tekscan = file.createDataSet("IndenterTekscan", H5::PredType::NATIVE_DOUBLE, dataspace_tekscan, proplist2);
    Converter::dataset_tekscan = &dataset_tekscan;


    omp_set_num_threads(count_threads);

#pragma omp parallel for schedule(dynamic, 1)
    for(int i=startframe; i<=endframe; i+=block_size)
    {
        int remaining = std::min(endframe-i+1, block_size);
        spdlog::info("thread {}; processing frames {} to {}", omp_get_thread_num(), i, i+remaining);
        Converter c;
        c.process_subset(i, remaining, frames_directory, export_bgeo, export_paraview, export_intact, export_damaged);
    }

    file.close();
    spdlog::info("done");

    return 0;
}


