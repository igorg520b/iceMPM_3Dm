#ifndef SNAPSHOTMANAGER_H
#define SNAPSHOTMANAGER_H

#include <array>
#include <vector>
#include <string>
#include <utility>

#include <H5Cpp.h>
#include <Eigen/Core>

#include <converter/visualpoint.h>

class Model3D;


class SnapshotManager
{
public:
    SnapshotManager();
    Model3D *model;

    void LoadRawPoints(std::string fileName);
    void SaveSnapshot(std::string outputDirectory, const int frame, bool compress = false);
    void ReadSnapshot(std::string fileName, int partitions);

    // saving animation frames
    bool previous_frame_exists = false;

    std::vector<VisualPoint> visual_state;
    std::vector<int> last_pos_refresh_frame;
    std::vector<std::pair<int, std::array<float,6>>> update_pos_vel;
    std::vector<std::pair<int, float>> update_Jp;
    std::vector<std::pair<int, uint8_t>> update_status;

    constexpr static float threshold_Jp = 1e-2;

    void SaveFrame(std::string outputDirectory, const int frame);

};

#endif
