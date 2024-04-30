#ifndef SNAPSHOTMANAGER_H
#define SNAPSHOTMANAGER_H

#include <array>
#include <vector>
#include <string>
#include <utility>

#include <H5Cpp.h>
#include <Eigen/Core>

class Model3D;


class SnapshotManager
{
public:
    SnapshotManager();
    Model3D *model;

    void LoadRawPoints(std::string fileName);
    void SaveSnapshot(std::string outputDirectory, bool compress = false);
    void ReadSnapshot(std::string fileName, int partitions);

    // saving animation frames
    bool previous_frame_exists = false;
    struct VisualPoint
    {
        float pos[3], vel[3];
        float Jp_inv, p, q;
        uint8_t status;
    };

    std::vector<VisualPoint> visual_state;
    std::vector<int> last_pos_refresh_frame;
    std::vector<std::pair<int, std::array<float,6>>> update_pos_vel;
    std::vector<std::pair<int, std::array<float,3>>> update_Jp_p_q;
    std::vector<std::pair<int, uint8_t>> update_status;

    H5::CompType ctUpdPV, ctUpdJpPQ, ctUpdS, ctVisualPoint;

    constexpr static float threshold_pos = 2e-3;
    constexpr static float threshold_Jp = 2e-2;
    constexpr static float threshold_pq = 1e5;

    void SaveFrame(std::string outputDirectory);

};

#endif
