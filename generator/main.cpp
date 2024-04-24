#include <cxxopts.hpp>

#include <gmsh.h>
#include <omp.h>
#include <spdlog/spdlog.h>

#include "grainprocessor.h"

// 80 million (grid resolution ~512)
// -s block -n 80000000 -o 3d_80m.h5 -m /home/s2/Projects-CUDA/iceMPM_multi_3D/generator/msh_3d/1k.msh -c 1.5

// 10 million (grid resolution ~256)
// -s block -n 10000000 -o 3d_10m.h5 -m /home/s2/Projects-CUDA/iceMPM_multi_3D/generator/msh_3d/1k.msh -c 1.5

// 1.3 million (grid ~128)
// -s block -n 1300000 -o 3d_1.3m.h5 -m /home/s2/Projects-CUDA/iceMPM_multi_3D/generator/msh_3d/1k.msh -c 1.5

// 200k (grid ~64)
// -s block -n 200000 -o 3d_200k.h5 -m /home/s2/Projects-CUDA/iceMPM_multi_3D/generator/msh_3d/1k.msh -c 1.5


int main(int argc, char *argv[])
{
    gmsh::initialize();
    spdlog::info("testing threads {}", omp_get_max_threads());
#pragma omp parallel
    {     spdlog::info("{}", omp_get_thread_num()); }
    std::cout << std::endl;


    // parse options
    cxxopts::Options options("grain identifier", "Generate raw input file for MPM simulation");

    options.add_options()
        // point generation
        ("s,shape", "Shape to generate (cone, block)", cxxopts::value<std::string>()->default_value("cone"))
        ("n,numberofpoints", "Make a set of N points for the simulation starting input", cxxopts::value<int>()->default_value("10000000"))
        ("o,output", "Output file name", cxxopts::value<std::string>()->default_value("raw_10m.h5"))
        ("m,msh", "Input file with grains", cxxopts::value<std::string>())
        ("c,scale", "Scale for grain mapping", cxxopts::value<float>()->default_value("3"))

        // for block
        ("x,bx", "Length of the block", cxxopts::value<float>()->default_value("2.5"))
        ("y,by", "Height of the block", cxxopts::value<float>()->default_value("1.0"))
        ("z,bz", "Width of the block", cxxopts::value<float>()->default_value("1.5"))

        // for cone
        ("diameter", "Diameter of the cone", cxxopts::value<float>()->default_value("0.2688"))
        ("top", "Diameter at the top of the cone", cxxopts::value<float>()->default_value("0.0254"))
        ("angle", "Taper angle of the cone", cxxopts::value<float>()->default_value("21"))
        ("height", "Total height of the sample", cxxopts::value<float>()->default_value("0.1"))
        ;

    auto option_parse_result = options.parse(argc, argv);


    // generate points input file
    std::string shape = option_parse_result["shape"].as<std::string>();
    int n = option_parse_result["numberofpoints"].as<int>();
    std::string output_file = option_parse_result["output"].as<std::string>();
    std::string msh_file = option_parse_result["msh"].as<std::string>();
    float scale = option_parse_result["scale"].as<float>();

    GrainProcessor gp;

    if(shape == "cone")
    {
        gp.LoadMSH(msh_file);
        float diameter = option_parse_result["diameter"].as<float>();
        float top = option_parse_result["top"].as<float>();
        float angle = option_parse_result["angle"].as<float>();
        float height = option_parse_result["height"].as<float>();
        gp.generate_cone(diameter, top, angle, height, n);
        gp.IdentifyGrains(scale);
        gp.Write_HDF5(output_file);
    }
    else if(shape == "block")
    {
        gp.LoadMSH(msh_file);
        float bx = option_parse_result["bx"].as<float>();
        float by = option_parse_result["by"].as<float>();
        float bz = option_parse_result["bz"].as<float>();
        gp.generate_block(bx, by, bz, n);
        gp.IdentifyGrains(scale);
        gp.Write_HDF5(output_file);
    }
    else throw std::runtime_error("incorrect shape");
}
