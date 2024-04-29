# iceMPM 3D multi-GPU
Multi-GPU implementation of the Material Point Method for modeling ice

![Screenshot of the GUI version](/screenshot.png)


## Required libraries for CLI version

> apt update && apt install libeigen3-dev libspdlog-dev libcxxopts-dev rapidjson-dev libhdf5-dev cmake mc -y

## Additional libraries for GUI version

> apt install libvtk9-dev qtbase5-dev qtbase5-dev-tools qtchooser

Edit CMakeLists.txt to select the compute capability of the NVIDA GPU (default is 8.0).

## To generate a point cloud in HDF5 format: 

> gen3d -s block -n 2000000 -o 3d_2m.h5 -m /home/s2/Projects-CUDA/iceMPM_multi_3D/generator/msh_3d/1k.msh -c 1.5

## Input files

A simulation can be started with a JSON configuration file via:

> cm3 startfile.json

A simulation can be "resumed" from a full snapshot via:

> cm3 --resume snapshot_file.h5 --partitions 4

To create an initial snapshot (.h5) from a JSON file:

> cm3 startfile.json --snapshot
