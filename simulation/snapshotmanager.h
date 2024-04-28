#ifndef SNAPSHOTMANAGER_H
#define SNAPSHOTMANAGER_H

#include <array>
#include <vector>
#include <string>

#include <H5Cpp.h>

class Model3D;


class SnapshotManager
{
public:
    Model3D *model;

    void LoadRawPoints(std::string fileName);
    void SaveSnapshot(std::string outputDirectory, bool compress = false);



//    void ReadSnapshot(std::string fileName); // return file number
//    void SavePQ(std::string directory);

    const std::string directory_snapshots = "snapshots";
    const std::string directory_pq = "pq";
};

#endif
