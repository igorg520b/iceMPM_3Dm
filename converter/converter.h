#ifndef CONVERTER_H
#define CONVERTER_H

#include <utility>
#include <vector>
#include <array>
#include <cstdio>
#include <mutex>

#include <H5Cpp.h>
#include <spdlog/spdlog.h>
#include <Eigen/Core>
#include <Partio.h>

#include "visualpoint.h"

#include <vtkCellArray.h>
#include <vtkNew.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkXMLPolyDataWriter.h>
#include <vtkFloatArray.h>
#include <vtkIntArray.h>
#include <vtkPointData.h>
#include <vtkUnstructuredGrid.h>
#include <vtkXMLUnstructuredGridWriter.h>
#include <vtkCylinderSource.h>
#include <vtkPolyDataMapper.h>
#include <vtkTransform.h>
#include <vtkTransformFilter.h>
#include <vtkAppendFilter.h>
#include <vtkStructuredGrid.h>
#include <vtkXMLStructuredGridWriter.h>
#include <vtkCellData.h>

class Converter
{
public:
    Converter();

    constexpr static std::string_view directory_output = "_output";
    constexpr static std::string_view directory_bgeo = "output_bgeo";
    constexpr static std::string_view directory_points = "points";
    constexpr static std::string_view directory_points_intact = "points_intact";
    constexpr static std::string_view directory_points_damaged = "points_damaged";
    constexpr static std::string_view directory_indenter = "indenter";
    constexpr static std::string_view directory_indenter_hdf5 = "indenter_hdf5";
    constexpr static std::string_view directory_sensor = "sensor";

    static std::mutex *accessing_indenter_force_file;
    static H5::DataSet *dataset_indenter_totals;
    static H5::DataSet *dataset_tekscan;
    static int frames_total;

    void process_subset(const int frame_start, int count, std::string directory, bool bgeo,
                        bool paraview, bool paraview_intact, bool paraview_damaged);

private:
    std::vector<VisualPoint> v;
    std::vector<std::pair<int, std::array<float,6>>> update_pos_vel;
    std::vector<std::pair<int, float>> update_Jp;
    std::vector<std::pair<int, uint8_t>> update_status;
    std::vector<int> last_pos_refresh_frame;
    std::vector<double> indenter_data;
    std::vector<double> indenter_data_norm;
    bool need_to_read_points; // no need if only indenter force is exported

    int IndenterSubdivisions, GridZ, UpdateEveryNthStep;
    char fileName[20];
    double dt, IndDiameter, cellsize;
    double indenter_x, indenter_y;
    double SimulationTime;

    void read_full_frame(H5::H5File &file, H5::DataSet &dataset_indenter);
    void read_partial_frame(H5::H5File &file);
    void read_file(std::string fileName);

    void save_points();
    void save_points_intact();
    void save_points_damaged();
    void save_indenter();
    void save_tekscan();
    void save_bgeo();
    void save_indenter_total();


    int frame, frame_start;

    vtkNew<vtkCylinderSource> cylinder;
    vtkNew<vtkTransform> transform;
    vtkNew<vtkTransformFilter> transformFilter;
    vtkNew<vtkAppendFilter> appendFilter;
    vtkNew<vtkUnstructuredGrid> unstructuredGrid;
    vtkNew<vtkXMLUnstructuredGridWriter> writer2;

    vtkNew<vtkPoints> grid_points;
    vtkNew<vtkStructuredGrid> structuredGrid;
    vtkNew<vtkXMLStructuredGridWriter> writer3;
    vtkNew<vtkFloatArray> values;

    // points
    vtkNew<vtkPoints> points;
    vtkNew<vtkFloatArray> values_Jp;
    vtkNew<vtkIntArray> values_status;
    vtkNew<vtkIntArray> values_partitions;
    vtkNew<vtkPolyData> polydata;
    vtkNew<vtkXMLPolyDataWriter> writer1;
};

#endif // CONVERTER_H
