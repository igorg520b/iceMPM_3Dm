#include <iostream>
#include <functional>
#include <string>
#include <filesystem>
#include <chrono>

#include <cxxopts.hpp>
#include <spdlog/spdlog.h>
#include <omp.h>
#include <H5Cpp.h>

#include "visualpoint.h"


constexpr int block_size = 100;
constexpr std::string_view directory_output = "output";
constexpr std::string_view directory_bgeo = "output_bgeo";
constexpr std::string_view directory_points = "points";
constexpr std::string_view directory_indenter = "indenter";
constexpr std::string_view directory_sensor = "sensor";


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
        ("b,bgeo", "Export as BGEO")
        ("p,paraview", "Export for Paraview")
        ;

//    options.parse_positional({"directory"});
    auto option_parse_result = options.parse(argc, argv);

    std::string frames_directory = option_parse_result["directory"].as<std::string>();

    bool export_bgeo = option_parse_result.count("bgeo");
    bool export_paraview = option_parse_result.count("paraview");
    int frames = option_parse_result["frames"].as<int>();
    int count_threads = option_parse_result["threads"].as<int>();
    spdlog::info("frames {}; threads {}", frames, count_threads);

    std::filesystem::path od(directory_output);
    std::filesystem::path od1(std::string(directory_output) + "/" + std::string(directory_bgeo));
    std::filesystem::path od2(std::string(directory_output) + "/" + std::string(directory_points));
    std::filesystem::path od3(std::string(directory_output) + "/" + std::string(directory_indenter));
    std::filesystem::path od4(std::string(directory_output) + "/" + std::string(directory_sensor));
    if(!std::filesystem::is_directory(od) || !std::filesystem::exists(od)) std::filesystem::create_directory(od);
    if(!std::filesystem::is_directory(od1) || !std::filesystem::exists(od1)) std::filesystem::create_directory(od1);
    if(!std::filesystem::is_directory(od2) || !std::filesystem::exists(od2)) std::filesystem::create_directory(od2);
    if(!std::filesystem::is_directory(od3) || !std::filesystem::exists(od3)) std::filesystem::create_directory(od3);
    if(!std::filesystem::is_directory(od4) || !std::filesystem::exists(od4)) std::filesystem::create_directory(od4);

    VisualPoint::InitializeStatic();

    omp_set_num_threads(count_threads);

#pragma omp parallel for schedule(dynamic, 1)
    for(int i=0; i<frames; i+=block_size)
    {
        int remaining = std::min(frames-i, block_size);
        spdlog::info("thread {}; processing frames {} to {}", omp_get_thread_num(), i, i+remaining);
        process_subset(i, remaining, frames_directory, export_bgeo, export_paraview);
    }

    spdlog::info("done");

    return 0;
}



void process_subset(int frame_start, int count, std::string directory, bool bgeo, bool paraview)
{
    spdlog::info("process_subset; frame_start {}; count {}", frame_start, count);

    std::vector<VisualPoint> v;
    std::vector<std::pair<int, std::array<float,6>>> update_pos_vel;
    std::vector<std::pair<int, float>> update_Jp;
    std::vector<std::pair<int, uint8_t>> update_status;
    std::vector<int> last_pos_refresh_frame;

    char fileName[20];

    for(int frame=frame_start; frame<frame_start+count; frame++)
    {
        snprintf(fileName, sizeof(fileName), "v%05d.h5", frame);
        std::string fullFilePath = directory + "/" + fileName;
        spdlog::info("reading frame {} from file {}", frame, fullFilePath);

        H5::H5File file(fullFilePath, H5F_ACC_RDONLY);

        H5::DataSet dataset_indenter = file.openDataSet("Indenter");
        H5::Attribute att_partial_frame = dataset_indenter.openAttribute("partial_frame");
        uint8_t prev_frame;
        att_partial_frame.read(H5::PredType::NATIVE_UINT8, &prev_frame);
        if(frame == frame_start && prev_frame) throw std::runtime_error("partial frame instead of a full frame");

        if(!prev_frame)
        {
            // load full frame
            H5::DataSet dataset_full = file.openDataSet("VisualPoints");
            hsize_t nPoints;
            dataset_full.getSpace().getSimpleExtentDims(&nPoints, NULL);

            if(frame == frame_start)
            {
                v.resize(nPoints);
                last_pos_refresh_frame.resize(nPoints);
            }
            dataset_full.read(v.data(), VisualPoint::ctVisualPoint);
            std::fill(last_pos_refresh_frame.begin(), last_pos_refresh_frame.end(),frame);
        }
        else
        {
            update_pos_vel.clear();
            update_Jp.clear();
            update_status.clear();

            H5::DataSet ds_pos, ds_status, ds_jp;

            try {
                ds_pos = file.openDataSet("UpdatePos");
                hsize_t n;
                ds_pos.getSpace().getSimpleExtentDims(&n, NULL);
                update_pos_vel.resize(n);
                ds_pos.read(update_pos_vel.data(), VisualPoint::ctUpdPV);
            }
            catch(...) {}


            // load partial frame
            /*
    H5::DataSet dataset;
    try {
        dataset = file.openDataSet(dataset_name);
        std::cout << "Dataset exists: " << dataset_name << std::endl;
    } catch(const H5::FileIException&) {
        std::cout << "Dataset does not exist: " << dataset_name << std::endl;
    } catch(const H5::DataSetIException&) {
        std::cout << "Dataset does not exist: " << dataset_name << std::endl;
    }
*/
        }

        file.close();
    }
}




