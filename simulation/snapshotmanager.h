#ifndef SNAPSHOTMANAGER_H
#define SNAPSHOTMANAGER_H

#include <array>
#include <vector>
#include <string>

#include <H5Cpp.h>


class SnapshotManager
{
public:
    Model3D *model;

    void SaveSnapshot(std::string directory);
    void ReadSnapshot(std::string fileName); // return file number
    void LoadRawPoints(std::string fileName);
    void SaveParametersAsAttributes(H5::DataSet &dataset);
    void SavePQ(std::string directory);

    const std::string directory_snapshots = "snapshots";
    const std::string directory_pq = "pq";
};

#endif
