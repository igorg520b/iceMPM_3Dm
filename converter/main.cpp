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
        ("f,frames", "Number of frames to convert", cxxopts::value<int>()->default_value("2400"))
        ("t,threads", "Number of threads to run in parallel", cxxopts::value<int>()->default_value("1"))
        ("b,bgeo", "Export as BGEO", cxxopts::value<bool>())
        ("p,paraview", "Export for Paraview", cxxopts::value<bool>())
        ("i,intact", "Export points for Paraview, only intact material", cxxopts::value<bool>())
        ;

//    options.parse_positional({"directory"});
    auto option_parse_result = options.parse(argc, argv);

    std::string frames_directory = option_parse_result["directory"].as<std::string>();

    bool export_bgeo = option_parse_result.count("bgeo");
    bool export_paraview = option_parse_result.count("paraview");
    bool export_intact = option_parse_result.count("intact");
    int frames = option_parse_result["frames"].as<int>();
    int count_threads = option_parse_result["threads"].as<int>();
    spdlog::info("frames {}; threads {}; paraview {}; bgeo {}", frames, count_threads, export_paraview, export_bgeo);

    std::filesystem::path od(Converter::directory_output);
    std::filesystem::path od1(std::string(Converter::directory_output) + "/" + std::string(Converter::directory_bgeo));
    std::filesystem::path od2(std::string(Converter::directory_output) + "/" + std::string(Converter::directory_points));
    std::filesystem::path od2b(std::string(Converter::directory_output) + "/" + std::string(Converter::directory_points_intact));
    std::filesystem::path od3(std::string(Converter::directory_output) + "/" + std::string(Converter::directory_indenter));
    std::filesystem::path od4(std::string(Converter::directory_output) + "/" + std::string(Converter::directory_sensor));
    if(!std::filesystem::is_directory(od) || !std::filesystem::exists(od)) std::filesystem::create_directory(od);
    if(!std::filesystem::is_directory(od1) || !std::filesystem::exists(od1)) std::filesystem::create_directory(od1);
    if(!std::filesystem::is_directory(od2) || !std::filesystem::exists(od2)) std::filesystem::create_directory(od2);
    if(!std::filesystem::is_directory(od2b) || !std::filesystem::exists(od2b)) std::filesystem::create_directory(od2b);
    if(!std::filesystem::is_directory(od3) || !std::filesystem::exists(od3)) std::filesystem::create_directory(od3);
    if(!std::filesystem::is_directory(od4) || !std::filesystem::exists(od4)) std::filesystem::create_directory(od4);

    VisualPoint::InitializeStatic();
    std::mutex accessing_indenter_force_file;
    Converter::accessing_indenter_force_file = &accessing_indenter_force_file;

    H5::H5File file(std::string(Converter::directory_output) + "/indenter.h5", H5F_ACC_TRUNC);
    hsize_t dims_indenter[2] = {(hsize_t)frames,5};
    H5::DataSpace dataspace_indenter(2, dims_indenter);
    H5::DataSet dataset_indenter = file.createDataSet("IndenterTotals", H5::PredType::NATIVE_DOUBLE, dataspace_indenter);
    Converter::dataset_indenter_totals = &dataset_indenter;
    Converter::frames_total = frames;

    omp_set_num_threads(count_threads);

#pragma omp parallel for schedule(dynamic, 1)
    for(int i=0; i<frames; i+=block_size)
    {
        int remaining = std::min(frames-i, block_size);
        spdlog::info("thread {}; processing frames {} to {}", omp_get_thread_num(), i, i+remaining);
        Converter c;
        c.process_subset(i, remaining, frames_directory, export_bgeo, export_paraview, export_intact);
    }

    file.close();
    spdlog::info("done");

    return 0;
}


