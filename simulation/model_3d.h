#ifndef MODEL_H
#define MODEL_H

#include <vector>
#include <algorithm>
#include <chrono>
#include <unordered_set>
#include <utility>
#include <cmath>
#include <random>
#include <mutex>
#include <iostream>
#include <string>
#include <fstream>

#include "parameters_sim_3d.h"
#include "point_3d.h"
#include "gpu_implementation6.h"

#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/LU>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/logger.h>


class Model3D
{
public:
    Model3D();
    ~Model3D() {};
    void Reset();

    void Prepare();        // invoked once, at simulation start
    bool Step();           // either invoked by Worker or via GUI
    void RequestAbort() {abortRequested = true;}   // asynchronous stop

    void UnlockCycleMutex();

    SimParams3D prms;
    GPU_Implementation6 gpu;
    int max_points_transferred, max_pt_deviation;
    bool SyncTopologyRequired;  // only for GUI visualization

    std::mutex processing_current_cycle_data; // locked until the current cycle results' are copied to host and processed
    std::mutex accessing_point_data;

private:
    bool abortRequested;
    std::shared_ptr<spdlog::logger> log_timing;
    std::shared_ptr<spdlog::logger> log_indenter_force;
};

#endif
