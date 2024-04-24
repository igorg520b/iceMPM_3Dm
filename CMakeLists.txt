cmake_minimum_required(VERSION 3.22.1)

project(RHITA_MPM_multi_3D LANGUAGES CXX C CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored -use_fast_maths")

set(CMAKE_INCLUDE_CURRENT_DIR ON)

option(BUILD_GUI_VERSION "Build the Qt version" OFF)
option(BUILD_GENERATOR "Build generator" OFF)
option(BUILD_CLI_VERSION "Build CLI version" OFF)

if(BUILD_GUI_VERSION)
    set(CMAKE_AUTOUIC ON)
    set(CMAKE_AUTOMOC ON)
    set(CMAKE_AUTORCC ON)
endif()

set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

find_package(Eigen3 REQUIRED NO_MODULE HINTS "C:\\libraries\\eigen-build")
if(WIN32)
    add_compile_definitions(H5_BUILT_AS_DYNAMIC_LIB)
else()
    find_package(HDF5 REQUIRED COMPONENTS CXX)
    find_package(spdlog REQUIRED)
endif()


# CLI version
if(BUILD_CLI_VERSION)

add_executable(cm3
    cli/main.cpp
    
    # simulation
    simulation/parameters_sim_3d.h
    simulation/parameters_sim_3d.cpp
    simulation/model_3d.cpp
    simulation/model_3d.h
    simulation/point_3d.h
    simulation/point_3d.cpp
    simulation/host_side_soa.h
    simulation/host_side_soa.cpp

    # cuda
    simulation/helper_math.cuh
    simulation/gpu_implementation6.h
    simulation/gpu_implementation6.cpp
    simulation/gpu_partition_3d.h
    simulation/gpu_partition_3d.cu

    # serializing
    simulation/snapshotmanager.h
    simulation/snapshotmanager.cpp
)


set_target_properties(cm3 PROPERTIES CUDA_ARCHITECTURES "80")
set_target_properties(cm3 PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_include_directories(cm3 PRIVATE cli simulation)

# Eigen
target_link_libraries(cm3 PRIVATE Eigen3::Eigen)

# HDF5
if(WIN32)
    target_link_directories(cm3 PRIVATE "C:\\libraries\\hdf5-build\\bin\\Release")
    target_link_libraries(cm3 PRIVATE hdf5_cpp hdf5 )
else()
    target_link_libraries(cm3 PRIVATE ${HDF5_LIBRARIES})
    target_include_directories(cm3 PRIVATE ${HDF5_INCLUDE_DIRS})
endif()

# spdlog
if (WIN32)
    target_link_libraries(cm3 PRIVATE spdlog)
    target_include_directories(cm3 PRIVATE "C:\\libraries\\spdlog\\include")
    target_link_directories(cm3 PRIVATE "C:\\libraries\\spdlog-build\\Release")
else()
    target_link_libraries(cm3 PRIVATE spdlog::spdlog)
endif()

# OpenMP
    if(WIN32)
    target_compile_options(cm3 PRIVATE /openmp)
else()
    target_compile_options(cm3 PRIVATE -fopenmp -Wno-stringop-overflow -Wno-invalid-offsetof)
    target_link_libraries(cm3 PRIVATE gomp)
endif()

endif()


# GUI VERSION
if(BUILD_GUI_VERSION)

find_package(VTK 9.1 HINTS "C:\\libraries\\vtk-build\\lib\\cmake\\vtk-9.1" REQUIRED COMPONENTS
  CommonColor
  CommonCore
  FiltersCore
  FiltersSources
  FiltersGeometry
  FiltersGeneral
  FiltersModeling
  InteractionStyle
  RenderingContextOpenGL2
  RenderingCore
  RenderingFreeType
  RenderingGL2PSOpenGL2
  RenderingOpenGL2
  IOGeometry
  RenderingAnnotation
  RenderingLabel
  IOImage
  IOXML
  ViewsQt
  CommonDataModel
  CommonTransforms
)

    find_package(Qt5 COMPONENTS Widgets REQUIRED)
    add_compile_definitions(QT_NO_KEYWORDS)

    add_executable(iceMPM_GUIm_3D
	# gui
        gui/main.cpp

        gui/windows/mainwindow.cpp
        gui/windows/mainwindow.h
        gui/windows/mainwindow.ui
        icons.qrc

        gui/vtk/vtk_representation.cpp
        gui/vtk/vtk_representation.h

        gui/backgroundworker.cpp
        gui/backgroundworker.h
        gui/parameters_wrapper.h

    )

    set_target_properties(iceMPM_GUIm_3D PROPERTIES CUDA_ARCHITECTURES "80")
    set_target_properties(iceMPM_GUIm_3D PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)

    target_include_directories(iceMPM_GUIm_3D PRIVATE gui gui/windows gui/vtk simulation)
    target_link_libraries(iceMPM_GUIm_3D PRIVATE Qt5::Widgets Qt5::Core)

    # Eigen
    find_package(Eigen3 REQUIRED NO_MODULE HINTS "C:\\libraries\\eigen-build")
    target_link_libraries(iceMPM_GUIm_3D PRIVATE Eigen3::Eigen)

    # VTK
    target_link_libraries(iceMPM_GUIm_3D PRIVATE ${VTK_LIBRARIES})

    #PBrowser
    if(WIN32)
        target_link_directories(iceMPM_GUIm_3D PRIVATE "C:\\libraries\\pbrowser-build")
        target_include_directories(iceMPM_GUIm_3D PRIVATE "C:\\libraries\\PBrowser")
        target_include_directories(iceMPM_GUIm_3D PRIVATE "C:\\libraries\\PBrowser\\propertybrowser")
        target_include_directories(iceMPM_GUIm_3D PRIVATE "C:\\libraries\\PBrowser\\objectpropertybrowser")
    else()
        target_link_directories(iceMPM_GUIm_3D PRIVATE "/home/s2/Projects-Libraries/build-PBrowser-Desktop_Qt_5_15_2_GCC_64bit-Release")
        target_include_directories(iceMPM_GUIm_3D PRIVATE /home/s2/Projects-Libraries/PBrowser/objectpropertybrowser)
        target_include_directories(iceMPM_GUIm_3D PRIVATE /home/s2/Projects-Libraries/PBrowser/propertybrowser)
        target_include_directories(iceMPM_GUIm_3D PRIVATE /home/s2/Projects-Libraries/PBrowser)
    endif()
    target_link_libraries(iceMPM_GUIm_3D PRIVATE PBrowser)

    # HDF5
    if(WIN32)
        target_link_directories(iceMPM_GUIm_3D PRIVATE "C:\\libraries\\hdf5-build\\bin\\Release")
        target_link_libraries(iceMPM_GUIm_3D PRIVATE hdf5_cpp hdf5 )
    else()
        target_link_libraries(iceMPM_GUIm_3D PRIVATE ${HDF5_LIBRARIES})
        target_include_directories(iceMPM_GUIm_3D PRIVATE ${HDF5_INCLUDE_DIRS})
    endif()

    # spdlog
    if (WIN32)
        target_link_libraries(iceMPM_GUIm_3D PRIVATE spdlog)
        target_include_directories(iceMPM_GUIm_3D PRIVATE "C:\\libraries\\spdlog\\include")
        target_link_directories(iceMPM_GUIm_3D PRIVATE "C:\\libraries\\spdlog-build\\Release")
    else()
        target_link_libraries(iceMPM_GUIm_3D PRIVATE spdlog::spdlog)
    endif()


    # OpenMP
        if(WIN32)
        target_compile_options(iceMPM_GUIm_3D PRIVATE /openmp)
    else()
        target_compile_options(iceMPM_GUIm_3D PRIVATE -fopenmp -Wno-stringop-overflow -Wno-invalid-offsetof)
        target_link_libraries(iceMPM_GUIm_3D PRIVATE gomp)
    endif()

    vtk_module_autoinit(
      TARGETS iceMPM_GUIm_3D
      MODULES ${VTK_LIBRARIES}
    )

endif()



if(BUILD_GENERATOR)

    add_executable(gen3d
        generator/main.cpp
        generator/grainprocessor.h
        generator/grainprocessor.cpp
        generator/bvh/bvhn.cpp
        generator/bvh/bvhn.h
        generator/bvh/kdop24.cpp
        generator/bvh/kdop24.h
        generator/bvh/SimplePool.h
    )

target_include_directories(gen3d PRIVATE generator generator/bvh generator/poisson_disk_sampling)
target_include_directories(gen3d PRIVATE "/home/s2/Projects-other/gmsh/api")

target_link_libraries(gen3d PRIVATE gmsh)

# Eigen
target_link_libraries(gen3d PRIVATE Eigen3::Eigen)

# HDF5
if(WIN32)
    target_link_directories(gen3d PRIVATE "C:\\libraries\\hdf5-build\\bin\\Release")
    target_link_libraries(gen3d PRIVATE hdf5_cpp hdf5 )
else()
    target_link_libraries(gen3d PRIVATE ${HDF5_LIBRARIES})
    target_include_directories(gen3d PRIVATE ${HDF5_INCLUDE_DIRS})
endif()

# spdlog
if (WIN32)
    target_link_libraries(gen3d PRIVATE spdlog)
    target_include_directories(gen3d PRIVATE "C:\\libraries\\spdlog\\include")
    target_link_directories(gen3d PRIVATE "C:\\libraries\\spdlog-build\\Release")
else()
    target_link_libraries(gen3d PRIVATE spdlog::spdlog)
endif()

# OpenMP
    if(WIN32)
    target_compile_options(gen3d PRIVATE /openmp)
else()
    target_compile_options(gen3d PRIVATE -fopenmp -Wno-stringop-overflow -Wno-invalid-offsetof)
    target_link_libraries(gen3d PRIVATE gomp)
endif()

endif()
