#include <iostream>
#include <functional>
#include <string>
#include <filesystem>
#include <atomic>
#include <thread>
#include <chrono>
#include <mutex>

#include <cxxopts.hpp>
#include <spdlog/spdlog.h>

#include "model_3d.h"
#include "snapshotmanager.h"



int main(int argc, char** argv)
{
    Model3D model;
    SnapshotManager snapshot;
    snapshot.model = &model;

    std::string snapshot_directory = "_snapshots";
    std::string animation_frame_directory = "_snapshots_animation";
    std::thread snapshot_thread;

    // parse options
    cxxopts::Options options("Ice MPM", "CLI version of MPM simulation");

    options.add_options()
        ("file", "Configuration file", cxxopts::value<std::string>())
        ("s,snapshot", "Only write the starting snapshot", cxxopts::value<bool>()->default_value("false"))
        ("r,resume", "Resume from a full snapshot (.h5) file", cxxopts::value<std::string>())
        ("p,partitions", "Number of partitions (if different from snapshot)", cxxopts::value<int>()->default_value("-1"))
        ("period", "Snapshot record period (in number of frames)", cxxopts::value<int>())
        ("dpthreshold", "Override DP threshold value", cxxopts::value<double>())
        ("phi", "Override Phi value (in degrees)", cxxopts::value<double>())
        ;
    options.parse_positional({"file"});

    auto option_parse_result = options.parse(argc, argv);

    if(option_parse_result.count("resume"))
    {
        // resume from snapshot
        std::string snapshot_file = option_parse_result["resume"].as<std::string>();
        int partitions = option_parse_result["partitions"].as<int>();
        spdlog::info("resuming snapshot {}",snapshot_file);
        snapshot.ReadSnapshot(snapshot_file, partitions);
        if(option_parse_result.count("period")) model.prms.SnapshotPeriod = option_parse_result["period"].as<int>();
    }
    else if(option_parse_result.count("file"))
    {
        std::string params_file = option_parse_result["file"].as<std::string>();
        std::string pointCloudFile = model.prms.ParseFile(params_file);
        snapshot.LoadRawPoints(pointCloudFile);
    }

    if(option_parse_result.count("dpthreshold")) model.prms.DP_threshold_p = option_parse_result["dpthreshold"].as<double>();
    if(option_parse_result.count("phi")) model.prms.SetPhi(option_parse_result["phi"].as<double>());

    if(option_parse_result.count("snapshot"))
    {
        // only generate the starting snapshot
        // write a snapshot and return
        snapshot.SaveSnapshot(snapshot_directory, model.prms.AnimationFrameNumber(), true);
        return 0;
    }
    else
    {
        // save frame zero
        snapshot.SaveFrame(animation_frame_directory, model.prms.AnimationFrameNumber());
    }

    // ensure that the folder exists
//    std::filesystem::path outputFolder(snapshot_directory);
//    std::filesystem::create_directory(outputFolder);

    // start the simulation thread
    bool result;
    do
    {
        result = model.Step();

        if(snapshot_thread.joinable()) snapshot_thread.join(); // this should not happen
        snapshot_thread = std::thread([&](){
            int snapshot_number = model.prms.AnimationFrameNumber();
            spdlog::info("step {} finished; saving data", snapshot_number);

            if(snapshot_number % model.prms.SnapshotPeriod == 0)
                snapshot.SaveSnapshot(snapshot_directory, snapshot_number, true);

            if(snapshot_number % 100 == 0) snapshot.previous_frame_exists = false;
            snapshot.SaveFrame(animation_frame_directory, snapshot_number);

            model.UnlockCycleMutex();
            spdlog::info("done saving frame {}", snapshot_number);
        });
    } while(result);

    model.gpu.synchronize();
    snapshot_thread.join();

    std::cout << "cm done\n";

    return 0;
}
