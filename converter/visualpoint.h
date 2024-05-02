#ifndef VISUALPOINT_H
#define VISUALPOINT_H

#include <H5Cpp.h>


struct VisualPoint
{
    float pos[3], vel[3];
    float Jp_inv;
    uint8_t status;

    static H5::CompType ctUpdPV, ctUpdJp, ctUpdS, ctVisualPoint;
    static void InitializeStatic();
};

#endif // VISUALPOINT_H
