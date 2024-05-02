#include "visualpoint.h"
#include <array>
#include <utility>
#include <vector>

H5::CompType VisualPoint::ctUpdPV;
H5::CompType VisualPoint::ctUpdJp;
H5::CompType VisualPoint::ctUpdS;
H5::CompType VisualPoint::ctVisualPoint;


void VisualPoint::InitializeStatic()
{
    ctUpdPV = H5::CompType(sizeof(std::pair<int, std::array<float,6>>));
    ctUpdJp = H5::CompType(sizeof(std::pair<int, float>));
    ctUpdS = H5::CompType(sizeof(std::pair<int, uint8_t>));
    ctVisualPoint = H5::CompType(sizeof(VisualPoint));

    ctUpdPV.insertMember("idx", 0, H5::PredType::NATIVE_INT);
    ctUpdPV.insertMember("px", sizeof(int)+sizeof(float)*0, H5::PredType::NATIVE_FLOAT);
    ctUpdPV.insertMember("py", sizeof(int)+sizeof(float)*1, H5::PredType::NATIVE_FLOAT);
    ctUpdPV.insertMember("pz", sizeof(int)+sizeof(float)*2, H5::PredType::NATIVE_FLOAT);
    ctUpdPV.insertMember("vx", sizeof(int)+sizeof(float)*3, H5::PredType::NATIVE_FLOAT);
    ctUpdPV.insertMember("vy", sizeof(int)+sizeof(float)*4, H5::PredType::NATIVE_FLOAT);
    ctUpdPV.insertMember("vz", sizeof(int)+sizeof(float)*5, H5::PredType::NATIVE_FLOAT);

    ctUpdJp.insertMember("idx", 0, H5::PredType::NATIVE_INT);
    ctUpdJp.insertMember("Jp_inv", sizeof(int), H5::PredType::NATIVE_FLOAT);

    ctUpdS.insertMember("idx", 0, H5::PredType::NATIVE_INT);
    ctUpdS.insertMember("status", sizeof(int), H5::PredType::NATIVE_UINT8);

    ctVisualPoint.insertMember("px", 0, H5::PredType::NATIVE_FLOAT);
    ctVisualPoint.insertMember("py", sizeof(float), H5::PredType::NATIVE_FLOAT);
    ctVisualPoint.insertMember("pz", sizeof(float)*2, H5::PredType::NATIVE_FLOAT);
    ctVisualPoint.insertMember("vx", sizeof(float)*3, H5::PredType::NATIVE_FLOAT);
    ctVisualPoint.insertMember("vy", sizeof(float)*4, H5::PredType::NATIVE_FLOAT);
    ctVisualPoint.insertMember("vz", sizeof(float)*5, H5::PredType::NATIVE_FLOAT);
    ctVisualPoint.insertMember("Jp_inv", HOFFSET(VisualPoint, Jp_inv), H5::PredType::NATIVE_FLOAT);
    ctVisualPoint.insertMember("status", HOFFSET(VisualPoint, status), H5::PredType::NATIVE_UINT8);
}


