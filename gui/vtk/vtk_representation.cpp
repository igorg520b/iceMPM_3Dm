#include "vtk_representation.h"
#include "model_3d.h"
#include "parameters_sim_3d.h"
//#include <omp.h>
#include <algorithm>
#include <iostream>
#include <spdlog/spdlog.h>


VisualRepresentation::VisualRepresentation()
{
    int nLut = sizeof lutArrayTemperatureAdj / sizeof lutArrayTemperatureAdj[0];
//    hueLut->SetNumberOfTableValues(nLut);
//    for ( int i=0; i<nLut; i++)
//        hueLut->SetTableValue(i, lutArrayTemperatureAdj[i][0],
//                              lutArrayTemperatureAdj[i][1],
//                              lutArrayTemperatureAdj[i][2], 1.0);

    nLut = sizeof lutArrayPastel / sizeof lutArrayPastel[0];
    hueLut_pastel->SetNumberOfTableValues(nLut);
    for ( int i=0; i<nLut; i++)
        hueLut_pastel->SetTableValue(i, lutArrayPastel[i][0],
                              lutArrayPastel[i][1],
                              lutArrayPastel[i][2], 1.0);
    hueLut_pastel->SetTableRange(0,nLut-1);

    nLut = sizeof lutArrayMPMColors / sizeof lutArrayMPMColors[0];
    lutMPM->SetNumberOfTableValues(nLut);
    for ( int i=0; i<nLut; i++)
        lutMPM->SetTableValue(i, lutArrayMPMColors[i][0],
                              lutArrayMPMColors[i][1],
                              lutArrayMPMColors[i][2], 1.0);

    hueLut_four->SetNumberOfColors(5);
    hueLut_four->SetTableValue(0, 0.3, 0.3, 0.3);
    hueLut_four->SetTableValue(1, 1.0, 0, 0);
    hueLut_four->SetTableValue(2, 0, 1.0, 0);
    hueLut_four->SetTableValue(3, 0, 0, 1.0);
    hueLut_four->SetTableValue(4, 0, 0.5, 0.5);
    hueLut_four->SetTableRange(0,4);


    // indenter
    indenterSource->SetResolution(33);
    indenterSource->Update();

    transform->RotateX(90);
    transformFilter->SetTransform(transform);
    transformFilter->SetInputConnection(indenterSource->GetOutputPort());
    transformFilter->Update();

    indenterMapper->SetInputConnection(transformFilter->GetOutputPort());

    actor_indenter->SetMapper(indenterMapper);
    actor_indenter->GetProperty()->EdgeVisibilityOn();
    actor_indenter->GetProperty()->VertexVisibilityOff();
    actor_indenter->GetProperty()->SetColor(0.3,0.1,0.1);
    actor_indenter->GetProperty()->SetOpacity(0.5);
    actor_indenter->GetProperty()->SetEdgeColor(90.0/255.0, 90.0/255.0, 97.0/255.0);
    actor_indenter->PickableOff();
    actor_indenter->GetProperty()->SetLineWidth(3);


    // points
    points_polydata->SetPoints(points);
    points_polydata->GetPointData()->AddArray(visualized_values);
    visualized_values->SetName("visualized_values");
    points_polydata->GetPointData()->SetActiveScalars("visualized_values");

    points_filter->SetInputData(points_polydata);
    points_filter->Update();

    points_mapper->SetInputData(points_filter->GetOutput());
    points_mapper->UseLookupTableScalarRangeOn();
    points_mapper->SetLookupTable(lutMPM);

    actor_points->SetMapper(points_mapper);
    actor_points->GetProperty()->SetPointSize(2);
    actor_points->GetProperty()->SetVertexColor(1,0,0);
    actor_points->GetProperty()->SetColor(0,0,0);
    actor_points->GetProperty()->LightingOff();
    actor_points->GetProperty()->ShadingOff();
    actor_points->GetProperty()->SetInterpolationToFlat();
    actor_points->PickableOff();

    grid_mapper->SetInputData(structuredGrid);
//    grid_mapper->SetLookupTable(hueLut);

    actor_grid->SetMapper(grid_mapper);
    actor_grid->GetProperty()->SetEdgeVisibility(true);
    actor_grid->GetProperty()->SetEdgeColor(0.8,0.8,0.8);
    actor_grid->GetProperty()->LightingOff();
    actor_grid->GetProperty()->ShadingOff();
    actor_grid->GetProperty()->SetInterpolationToFlat();
    actor_grid->PickableOff();
    actor_grid->GetProperty()->SetColor(0.98,0.98,0.98);
//    actor_grid->GetProperty()->SetRepresentationToWireframe();

    // partitions grid
    partitions_grid_mapper->SetInputData(partitionsGrid);
    actor_partitions->SetMapper(partitions_grid_mapper);
    actor_partitions->GetProperty()->SetEdgeVisibility(true);
    actor_partitions->GetProperty()->SetEdgeColor(0.8,0.8,0.8);
    actor_partitions->GetProperty()->LightingOff();
    actor_partitions->GetProperty()->ShadingOff();
    actor_partitions->GetProperty()->SetInterpolationToFlat();
    actor_partitions->PickableOff();
    actor_partitions->GetProperty()->SetColor(0.4,0.4,0.4);
    actor_partitions->GetProperty()->SetRepresentationToWireframe();
    actor_partitions->GetProperty()->SetLineWidth(2);

    // bounding box
    boundingBox_mapper->SetInputData(boundingBoxGrid);

    actor_boundingBox->SetMapper(boundingBox_mapper);
    actor_boundingBox->GetProperty()->SetEdgeVisibility(true);
    actor_boundingBox->GetProperty()->LightingOff();
    actor_boundingBox->GetProperty()->ShadingOff();
    actor_boundingBox->GetProperty()->SetInterpolationToFlat();
    actor_boundingBox->PickableOff();
    actor_boundingBox->GetProperty()->SetColor(0.1,0.1,0.1);
    actor_boundingBox->GetProperty()->SetRepresentationToWireframe();



    // scalar bar
    scalarBar->SetLookupTable(lutMPM);
    scalarBar->SetMaximumWidthInPixels(130);
    scalarBar->SetBarRatio(0.07);
    scalarBar->SetMaximumHeightInPixels(200);
    scalarBar->GetPositionCoordinate()->SetCoordinateSystemToNormalizedDisplay();
    scalarBar->GetPositionCoordinate()->SetValue(0.01,0.015, 0.0);
    scalarBar->SetLabelFormat("%.1e");
    scalarBar->GetLabelTextProperty()->BoldOff();
    scalarBar->GetLabelTextProperty()->ItalicOff();
    scalarBar->GetLabelTextProperty()->ShadowOff();
    scalarBar->GetLabelTextProperty()->SetColor(0.1,0.1,0.1);

    // text
    vtkTextProperty* txtprop = actorText->GetTextProperty();
    txtprop->SetFontFamilyToArial();
    txtprop->BoldOff();
    txtprop->SetFontSize(14);
    txtprop->ShadowOff();
    txtprop->SetColor(0,0,0);
    actorText->SetDisplayPosition(500, 30);
}



void VisualRepresentation::SynchronizeTopology()
{
    model->accessing_point_data.lock();

    spdlog::info("SynchronizeTopology()");

    points->SetNumberOfPoints(model->prms.nPtsTotal);
    visualized_values->SetNumberOfValues(model->prms.nPtsTotal);

    // indenter
    indenterSource->SetRadius(model->prms.IndDiameter/2.f);
    indenterSource->SetHeight(model->prms.GridZ * model->prms.cellsize);
    indenterSource->Update();


    int gx = model->prms.GridXTotal;
    int gz = model->prms.GridZ;
    double &h = model->prms.cellsize;
    structuredGrid->SetDimensions(gx, 1, gz);

    grid_points->SetNumberOfPoints(gx*gz);
    for(int idx_z=0; idx_z<gz; idx_z++)
        for(int idx_x=0; idx_x<gx; idx_x++)
        {
            float x = idx_x * h;
            float z = idx_z * h;
            double pt_pos[3] {x, 0, z};
            grid_points->SetPoint((vtkIdType)(idx_x+idx_z*gx), pt_pos);
        }
    structuredGrid->SetPoints(grid_points);

    // bounding box
    boundingBox_points->SetNumberOfPoints(8);
    double x = model->prms.GridXTotal * model->prms.cellsize;
    double y = model->prms.GridY * model->prms.cellsize;
    double z = model->prms.GridZ * model->prms.cellsize;
    boundingBox_points->SetPoint(0, 0., 0., 0.);
    boundingBox_points->SetPoint(1, 0., y, 0.);
    boundingBox_points->SetPoint(2, x, 0., 0.);
    boundingBox_points->SetPoint(3, x, y, 0.);
    boundingBox_points->SetPoint(4, 0., 0., z);
    boundingBox_points->SetPoint(5, 0., y, z);
    boundingBox_points->SetPoint(6, x, 0., z);
    boundingBox_points->SetPoint(7, x, y, z);

    boundingBoxGrid->SetDimensions(2,2,2);
    boundingBoxGrid->SetPoints(boundingBox_points);


/*
    int nPartitions = model->gpu.partitions.size();
    partitionsGrid->SetDimensions(nPartitions+1, 2, 1);
    partitions_grid_points->SetNumberOfPoints((nPartitions+1)*2);
    // partitions grid
    double y1 = -0.5*h;
    double y2 = (model->prms.GridY-0.5)*h;
    for(int i=0;i<nPartitions;i++)
    {
        GPU_Partition &p = model->gpu.partitions[i];
        double x =(p.GridX_offset - 0.5)*h;
        double pt_pos1[3] {x,y1,-1.0};
        partitions_grid_points->SetPoint(i, pt_pos1);
        double pt_pos2[3] {x,y2,-1.0};
        partitions_grid_points->SetPoint(i+nPartitions+1, pt_pos2);
    }

    double x =(model->prms.GridXTotal - 0.5)*h;
    double pt_pos1[3] {x,y1,-0.5};
    partitions_grid_points->SetPoint(nPartitions, pt_pos1);
    double pt_pos2[3] {x,y2,-0.5};
    partitions_grid_points->SetPoint(2*nPartitions + 1, pt_pos2);

    partitionsGrid->SetPoints(partitions_grid_points);
*/
    model->accessing_point_data.unlock();
    SynchronizeValues();
}


void VisualRepresentation::SynchronizeValues()
{
    model->accessing_point_data.lock();

    // spdlog::info("SynchronizeValues() npts {}", model->prms.nPtsTotal);
    double indenter_x = model->prms.indenter_x;
    double indenter_y = model->prms.indenter_y;
    double indenter_z = model->prms.GridZ*model->prms.cellsize/2;
    indenterSource->SetCenter(indenter_x, indenter_z, -indenter_y);

    unsigned activePtsCount = 0;
    for(int i=0;i<model->gpu.hssoa.size;i++)
    {
        SOAIterator s = model->gpu.hssoa.begin()+i;
        if(s->getDisabledStatus()) continue;
        Eigen::Vector3d pos = s->getPos();
        points->SetPoint((vtkIdType)activePtsCount, pos[0], pos[1], pos[2]);
        activePtsCount++;
    }
    if(activePtsCount != model->prms.nPtsTotal) throw std::runtime_error("SynchronizeValues() point count mismatch (pos)");
    points->Modified();
    actor_points->GetProperty()->SetPointSize(model->prms.ParticleViewSize);
    points_filter->Update();
    double range = std::pow(10,ranges[VisualizingVariable]);
    double centerVal = 0;


    if(VisualizingVariable == VisOpt::partition)
    {
        scalarBar->VisibilityOn();
        points_mapper->ScalarVisibilityOn();
        points_mapper->SetColorModeToMapScalars();
        points_mapper->UseLookupTableScalarRangeOn();
        points_mapper->SetLookupTable(hueLut_pastel);
        scalarBar->SetLookupTable(hueLut_pastel);

        visualized_values->SetNumberOfValues(model->prms.nPtsTotal);
        activePtsCount = 0;
        for(int i=0;i<model->gpu.hssoa.size;i++)
        {
            SOAIterator s = model->gpu.hssoa.begin()+i;
            if(s->getDisabledStatus()) continue;
            uint8_t partition = s->getPartition();
            bool isCrushed = s->getCrushedStatus();
            if(isCrushed) partition = 41;
            bool isLiquid = s->getLiquidStatus();
            if(isLiquid) partition = 42;
            visualized_values->SetValue((vtkIdType)activePtsCount++, (float)partition);
        }
        if(activePtsCount != model->prms.nPtsTotal) throw std::runtime_error("SynchronizeValues() point count mismatch");
        visualized_values->Modified();
    }
    else if(VisualizingVariable == VisOpt::status)
    {
        scalarBar->VisibilityOn();
        points_mapper->ScalarVisibilityOn();
        points_mapper->SetColorModeToMapScalars();
        points_mapper->UseLookupTableScalarRangeOn();
        points_mapper->SetLookupTable(hueLut_four);
        scalarBar->SetLookupTable(hueLut_four);
        visualized_values->SetNumberOfValues(model->prms.nPtsTotal);
        activePtsCount = 0;
        for(int i=0;i<model->gpu.hssoa.size;i++)
        {
            SOAIterator s = model->gpu.hssoa.begin()+i;
            if(s->getDisabledStatus()) continue;
            bool isCrushed = s->getCrushedStatus();
            visualized_values->SetValue((vtkIdType)activePtsCount++, (float)(isCrushed ? 1 : 0));
        }
        if(activePtsCount != model->prms.nPtsTotal) throw std::runtime_error("SynchronizeValues() point count mismatch");
        visualized_values->Modified();
    }
    else if(VisualizingVariable == VisOpt::Jp_inv)
    {
        scalarBar->VisibilityOn();
        points_mapper->ScalarVisibilityOn();
        points_mapper->SetColorModeToMapScalars();
        points_mapper->UseLookupTableScalarRangeOn();
        points_mapper->SetLookupTable(lutMPM);
        scalarBar->SetLookupTable(lutMPM);
        lutMPM->SetTableRange(centerVal-range, centerVal+range);
        visualized_values->SetNumberOfValues(model->prms.nPtsTotal);
        activePtsCount = 0;
        for(int i=0;i<model->gpu.hssoa.size;i++)
        {
            SOAIterator s = model->gpu.hssoa.begin()+i;
            if(s->getDisabledStatus()) continue;
            double value = s->getValue(SimParams3D::idx_Jp_inv)-1;
            visualized_values->SetValue((vtkIdType)activePtsCount++, (float)value);
        }
        if(activePtsCount != model->prms.nPtsTotal) throw std::runtime_error("SynchronizeValues() point count mismatch");
        visualized_values->Modified();
    }
    else if(VisualizingVariable == VisOpt::grains)
    {
        scalarBar->VisibilityOn();
        points_mapper->ScalarVisibilityOn();
        points_mapper->SetColorModeToMapScalars();
        points_mapper->UseLookupTableScalarRangeOn();
        points_mapper->SetLookupTable(hueLut_pastel);
        scalarBar->SetLookupTable(hueLut_pastel);

        visualized_values->SetNumberOfValues(model->prms.nPtsTotal);
        activePtsCount = 0;
        for(int i=0;i<model->gpu.hssoa.size;i++)
        {
            SOAIterator s = model->gpu.hssoa.begin()+i;
            if(s->getDisabledStatus()) continue;
            uint16_t grain = s->getGrain()%40;
            bool isCrushed = s->getCrushedStatus();
            if(isCrushed) grain = 41;
            bool isLiquid = s->getLiquidStatus();
            if(!isLiquid) grain = 42;

            visualized_values->SetValue((vtkIdType)activePtsCount++, (float)grain);
        }
        if(activePtsCount != model->prms.nPtsTotal) throw std::runtime_error("SynchronizeValues() point count mismatch");
        visualized_values->Modified();
    }
    else
    {
        points_mapper->ScalarVisibilityOff();
        scalarBar->VisibilityOff();
    }
    model->accessing_point_data.unlock();

}


void VisualRepresentation::ChangeVisualizationOption(int option)
{
    VisualizingVariable = (VisOpt)option;
    SynchronizeTopology();
}

