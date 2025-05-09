#include "model_3d.h"
#include <spdlog/spdlog.h>
#include <algorithm>


Model3D::Model3D()
{
    log_timing = spdlog::basic_logger_mt("timings", "logs/timings.csv", true);
    log_indenter_force = spdlog::basic_logger_mt("indenter_force", "logs/indenter_force.csv", true);

    auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("logs/multisink.txt", true);
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    auto lg = std::make_shared<spdlog::logger>("multi_sink", spdlog::sinks_init_list({console_sink, file_sink}));
    spdlog::set_default_logger(lg);
    spdlog::set_pattern("%v");

    prms.Reset();
    gpu.model = this;
    GPU_Partition_3D::prms = &this->prms;
    gpu.enable_peer_access();
    spdlog::info("Model constructor");
}

bool Model3D::Step()
{
    double simulation_time = prms.SimulationTime;
    std::cout << '\n';
    spdlog::info("step {} ({}) started; sim_time {:>6.3}; host pts {}; cap {}; max_tr {}",
                 prms.SimulationStep, prms.AnimationFrameNumber(), simulation_time,
                 gpu.hssoa.size, gpu.hssoa.capacity, max_points_transferred);

    int count_unupdated_steps = 0;
    gpu.reset_indenter_force_accumulator();

    do
    {
        simulation_time += prms.InitialTimeStep;
        prms.indenter_x = prms.indenter_x_initial + simulation_time*prms.IndVelocity;

        gpu.reset_grid();
        gpu.p2g();
        gpu.receive_halos();
        gpu.update_nodes();
        const bool isZeroStep = (prms.SimulationStep+count_unupdated_steps) % prms.UpdateEveryNthStep == 0;
        const bool enablePointTransfer = (prms.SimulationStep+count_unupdated_steps) % (prms.UpdateEveryNthStep/prms.PointTransferFrequency) == 0;
        gpu.g2p(isZeroStep, enablePointTransfer);
        if(enablePointTransfer) gpu.receive_points();
        gpu.record_timings(enablePointTransfer);

        count_unupdated_steps++;
    } while((prms.SimulationStep+count_unupdated_steps) % prms.UpdateEveryNthStep != 0);

    processing_current_cycle_data.lock();   // if locked, previous results are not yet processed by the host
    accessing_point_data.lock();

    prms.SimulationTime = simulation_time;
    prms.SimulationStep += count_unupdated_steps;

    gpu.transfer_from_device();
    max_pt_deviation = 0;
    for(GPU_Partition_3D &p : gpu.partitions)
    {
        max_points_transferred = std::max((int)max_points_transferred, (int)p.max_pts_sent);
        max_pt_deviation = std::max(max_pt_deviation, p.max_pt_deviation);
    }
    spdlog::info("finished {} ({}); host pts {}; cap-pts {}; max_tr {}; max_dev {}; ptf {}; t {:>6.3}; f {:>7.3}", prms.SimulationStep,
                 prms.AnimationFrameNumber(), gpu.hssoa.size, gpu.hssoa.capacity-gpu.hssoa.size, max_points_transferred,
                    max_pt_deviation, prms.PointTransferFrequency,prms.SimulationTime,gpu.indenter_force.norm());

    spdlog::get("indenter_force")->info("{},{},{},{}",prms.SimulationTime,gpu.indenter_force.norm(),gpu.indenter_force.x(), gpu.indenter_force.y());

    if(max_pt_deviation > prms.GridHaloSize/2) prms.PointTransferFrequency++; // transfer points more often if any risk

    spdlog::info("{:^3s} {:^8s} {:^8s} {:^7s} {:^3s} {:^3s} | {:^6s} {:^5s} {:^6s} | {:^5s} {:^6s} {:^5s} {:^5s} {:^5s} | {:^6s}",
                 "P-D",    "pts",  "free", "dis","msn", "mdv", "p2g",  "s2",  "S12",   "u",  "g2p", "psnt", "prcv","S36", "tot");
    bool rebalance = false;
    for(GPU_Partition_3D &p : gpu.partitions)
    {
        p.normalize_timings(count_unupdated_steps);
        double freeSpacePercentage = (double)(p.nPtsPitch-p.nPts_partition)/p.nPts_partition;
        double disabledPercentage = (double)p.nPts_disabled/p.nPtsPitch;
        if(freeSpacePercentage < prms.RebalanceThresholdFreeSpaceRemaining) rebalance = true;
        if(disabledPercentage > prms.RebalanceThresholdDisabledPercentage) rebalance = true;

        spdlog::info("{:>1}-{:>1} {:>8} {:>8} {:>7} {:>3} {:>3} | {:>6.0f} {:>5.0f} {:>6.0f} | {:>5.0f} {:>6.0f} {:>5.0f} {:5.0f} {:5.0f} | {:>6.0f}",
                     p.PartitionID, p.Device, p.nPts_partition, (p.nPtsPitch-p.nPts_partition), p.nPts_disabled, p.max_pts_sent, p.max_pt_deviation,
                     p.timing_10_P2GAndHalo, p.timing_20_acceptHalo, (p.timing_10_P2GAndHalo + p.timing_20_acceptHalo),
                     p.timing_30_updateGrid, p.timing_40_G2P, p.timing_60_ptsSent, p.timing_70_ptsAccepted,
                     (p.timing_30_updateGrid + p.timing_40_G2P + p.timing_60_ptsSent + p.timing_70_ptsAccepted),
                     p.timing_stepTotal);
        spdlog::get("timings")->info("{}, {:>2},{:>8},{:>8},{:>7},{:>3},{:>3},{:>5.1f},{:>5.1f},{:>5.1f},{:>5.1f},{:>5.1f},{:5.1f},{:>6.1f}",
                                     prms.AnimationFrameNumber(),
                                     p.PartitionID, p.nPts_partition, (p.nPtsPitch-p.nPts_partition), p.nPts_disabled, p.max_pts_sent, p.max_pt_deviation,
                                     p.timing_10_P2GAndHalo, p.timing_20_acceptHalo,
                                     p.timing_30_updateGrid, p.timing_40_G2P, p.timing_60_ptsSent, p.timing_70_ptsAccepted,p.timing_stepTotal);

    }

    // rebalance
    if(rebalance)
    {
        spdlog::info("squeezing and sorting HSSOA");
        gpu.hssoa.RemoveDisabledAndSort(prms.cellsize_inv, prms.GridY, prms.GridZ);
        gpu.split_hssoa_into_partitions();
        gpu.transfer_ponts_to_device();
        SyncTopologyRequired = true;
        spdlog::info("rebalancing done");
    }
    accessing_point_data.unlock();

    return (prms.SimulationTime < prms.SimulationEndTime);
}


void Model3D::UnlockCycleMutex()
{
    // current data was handled by host - allow next cycle to proceed
    processing_current_cycle_data.unlock();
}


void Model3D::Reset()
{
    spdlog::info("model reset");

    max_points_transferred = 0;
    prms.SimulationStep = 0;
    prms.SimulationTime = 0;
    SyncTopologyRequired = true;
}

void Model3D::Prepare()
{
    spdlog::info("model prepare");
    abortRequested = false;
    gpu.update_constants();
}

