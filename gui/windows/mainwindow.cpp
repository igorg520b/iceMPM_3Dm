#include <QFileDialog>
#include <QList>
#include <QPointF>
#include <QCloseEvent>
#include <QStringList>
#include <algorithm>
#include <cmath>
#include <fstream>
#include "mainwindow.h"
#include "./ui_mainwindow.h"

MainWindow::~MainWindow() {delete ui;}

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    params = new ParamsWrapper(&model.prms);
    snapshot.model = &model;
    representation.model = &model;
    worker = new BackgroundWorker(&model);
    representation.SynchronizeTopology();

    // VTK
    qt_vtk_widget = new QVTKOpenGLNativeWidget();
    qt_vtk_widget->setRenderWindow(renderWindow);

    renderer->SetBackground(1.0,1.0,1.0);
    renderWindow->AddRenderer(renderer);

    // property browser
    pbrowser = new ObjectPropertyBrowser(this);

    // splitter
    splitter = new QSplitter(Qt::Orientation::Horizontal);
    splitter->addWidget(pbrowser);
    splitter->addWidget(qt_vtk_widget);
    splitter->setSizes(QList<int>({100, 500}));
    setCentralWidget(splitter);

    // toolbar - combobox
    comboBox_visualizations = new QComboBox();
    ui->toolBar->addWidget(comboBox_visualizations);

    // double spin box
    qdsbValRange = new QDoubleSpinBox();
    qdsbValRange->setRange(-10, 10);
    qdsbValRange->setValue(-2);
    qdsbValRange->setDecimals(2);
    qdsbValRange->setSingleStep(0.25);
    ui->toolBar->addWidget(qdsbValRange);

    // statusbar
    statusLabel = new QLabel();
    labelElapsedTime = new QLabel();
    labelStepCount = new QLabel();

    QSizePolicy sp;
    const int status_width = 80;
    sp.setHorizontalPolicy(QSizePolicy::Fixed);
    labelStepCount->setSizePolicy(sp);
    labelStepCount->setFixedWidth(status_width*1.5);
    labelElapsedTime->setSizePolicy(sp);
    labelElapsedTime->setFixedWidth(status_width);

    ui->statusbar->addWidget(statusLabel);
    ui->statusbar->addPermanentWidget(labelElapsedTime);
    ui->statusbar->addPermanentWidget(labelStepCount);

// anything that includes the Model
    renderer->AddActor(representation.actor_points);
    renderer->AddActor(representation.actor_indenter);
//    renderer->AddActor(representation.actor_axes);
    renderer->AddActor(representation.actor_grid);
/*    renderer->AddActor(representation.actor_partitions);
    renderer->AddActor(representation.actorText);
    renderer->AddActor(representation.scalarBar);
*/
    renderer->AddActor(representation.actor_boundingBox);
    renderer->AddActor(representation.actor_wavePusher);
    representation.actor_wavePusher->VisibilityOff();

    // populate combobox
    QMetaEnum qme = QMetaEnum::fromType<VisualRepresentation::VisOpt>();
    for(int i=0;i<qme.keyCount();i++) comboBox_visualizations->addItem(qme.key(i));

    connect(comboBox_visualizations, QOverload<int>::of(&QComboBox::currentIndexChanged),
            [&](int index){ comboboxIndexChanged_visualizations(index); });

    restore_settings();

    windowToImageFilter->SetInput(renderWindow);
    windowToImageFilter->SetScale(1); // image quality
    windowToImageFilter->SetInputBufferTypeToRGBA(); //also record the alpha (transparency) channel
    windowToImageFilter->ReadFrontBufferOn(); // read from the back buffer
    writerPNG->SetInputConnection(windowToImageFilter->GetOutputPort());

    connect(ui->action_quit, &QAction::triggered, this, &MainWindow::quit_triggered);
    connect(ui->action_camera_reset, &QAction::triggered, this, &MainWindow::cameraReset_triggered);
    connect(ui->actionOpen, &QAction::triggered, this, &MainWindow::open_snapshot_triggered);
    connect(ui->actionStart_Pause, &QAction::triggered, this, &MainWindow::simulation_start_pause);
    connect(ui->actionLoad_Parameters, &QAction::triggered, this, &MainWindow::load_parameter_triggered);

    connect(qdsbValRange,QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &MainWindow::limits_changed);

    connect(worker, SIGNAL(workerPaused()), SLOT(background_worker_paused()));
    connect(worker, SIGNAL(stepCompleted()), SLOT(simulation_data_ready()));

    representation.SynchronizeTopology();
    pbrowser->setActiveObject(params);
    qDebug() << "MainWindow constructor done";
    updateGUI();
}



void MainWindow::restore_settings()
{
    // read/restore saved settings
    settingsFileName = QDir::currentPath() + "/cm.ini";
    QFileInfo fi(settingsFileName);
    renderer->ResetCamera();
    vtkCamera* camera = renderer->GetActiveCamera();

    if(fi.exists())
    {
        QSettings settings(settingsFileName,QSettings::IniFormat);
        QVariant var;

        renderer->ResetCamera();

        var = settings.value("camData");
        if(!var.isNull())
        {
            double *data = (double*)var.toByteArray().constData();
            camera->SetPosition(data[0],data[1],data[2]);
            camera->SetFocalPoint(data[3],data[4],data[5]);
            camera->SetViewUp(data[6],data[7],data[8]);
            camera->SetViewAngle(data[9]);
            camera->SetClippingRange(1e-3,1e5);
        }

        var = settings.value("visualization_ranges");
        if(!var.isNull())
        {
            QByteArray ba = var.toByteArray();
            memcpy(representation.ranges, ba.constData(), ba.size());
        }

        var = settings.value("take_screenshots");
        if(!var.isNull())
        {
            bool b = var.toBool();
            ui->actionTake_Screenshots->setChecked(b);
        }

        var = settings.value("splitter_size_0");
        if(!var.isNull())
        {
            int sz1 = var.toInt();
            int sz2 = settings.value("splitter_size_1").toInt();
            splitter->setSizes(QList<int>({sz1, sz2}));
        }

        var = settings.value("vis_option");
        if(!var.isNull())
        {
            comboBox_visualizations->setCurrentIndex(var.toInt());
            qdsbValRange->setValue(representation.ranges[var.toInt()]);
        }
    }
    else
    {
        cameraReset_triggered();
    }
}


void MainWindow::cameraReset_triggered()
{
    qDebug() << "MainWindow::on_action_camera_reset_triggered()";
    vtkCamera* camera = renderer->GetActiveCamera();
    renderer->ResetCamera();
    camera->Modified();
    renderWindow->Render();
}

void MainWindow::closeEvent(QCloseEvent* event)
{
    quit_triggered();
    event->accept();
}


void MainWindow::quit_triggered()
{
    qDebug() << "MainWindow::quit_triggered() ";
    worker->Finalize();
    // save settings and stop simulation
    QSettings settings(settingsFileName,QSettings::IniFormat);
    qDebug() << "MainWindow: closing main window; " << settings.fileName();

    double data[10];
    renderer->GetActiveCamera()->GetPosition(&data[0]);
    renderer->GetActiveCamera()->GetFocalPoint(&data[3]);
    renderer->GetActiveCamera()->GetViewUp(&data[6]);
    data[9]=renderer->GetActiveCamera()->GetViewAngle();
    QByteArray arr((char*)data, sizeof(data));
    settings.setValue("camData", arr);

    QByteArray ranges((char*)representation.ranges, sizeof(representation.ranges));
    settings.setValue("visualization_ranges", ranges);

    settings.setValue("vis_option", comboBox_visualizations->currentIndex());

    if(!qLastParameterFile.isEmpty()) settings.setValue("lastParameterFile", qLastParameterFile);

    QList<int> szs = splitter->sizes();
    settings.setValue("splitter_size_0", szs[0]);
    settings.setValue("splitter_size_1", szs[1]);

    settings.setValue("take_screenshots", ui->actionTake_Screenshots->isChecked());
    QApplication::quit();
}



void MainWindow::comboboxIndexChanged_visualizations(int index)
{
    representation.ChangeVisualizationOption(index);
    qdsbValRange->setValue(representation.ranges[index]);
    renderWindow->Render();
}

void MainWindow::limits_changed(double val_)
{
    int idx = (int)representation.VisualizingVariable;
    representation.ranges[idx] = val_;
    representation.SynchronizeValues();
    renderWindow->Render();
}




void MainWindow::open_snapshot_triggered()
{
    QString qFileName = QFileDialog::getOpenFileName(this, "Open Simulation Snapshot", QDir::currentPath(), "HDF5 Files (*.h5)");
    if(qFileName.isNull())return;

    snapshot.ReadSnapshot(qFileName.toStdString(), -1);
    representation.SynchronizeTopology();
    updateGUI();
    pbrowser->setActiveObject(params);
}


void MainWindow::load_parameter_triggered()
{
    QString qFileName = QFileDialog::getOpenFileName(this, "Load Parameters", QDir::currentPath(), "JSON Files (*.json)");
    if(qFileName.isNull())return;
    LoadParameterFile(qFileName);
}

void MainWindow::LoadParameterFile(QString qFileName)
{
    std::string rawPointFile = model.prms.ParseFile(qFileName.toStdString());
    snapshot.LoadRawPoints(rawPointFile);
    this->qLastParameterFile = qFileName;
    this->setWindowTitle(qLastParameterFile);
    model.Reset();
    representation.SynchronizeTopology();
    pbrowser->setActiveObject(params);

    if(model.prms.SetupType == 2)
    {
        representation.actor_wavePusher->VisibilityOn();
        representation.actor_indenter->VisibilityOff();
    }

    updateGUI();
}


void MainWindow::simulation_data_ready()
{
    updateGUI();
    if(ui->actionTake_Screenshots->isChecked())
        screenshot();

    int snapshot_number = model.prms.AnimationFrameNumber();
    if(snapshot_number % model.prms.SnapshotPeriod == 0)
        snapshot.SaveSnapshot(snapshot_directory, snapshot_number, true);

    if(snapshot_number % 100 == 0) snapshot.previous_frame_exists = false;
    snapshot.SaveFrame(animation_frame_directory, snapshot_number);

    model.UnlockCycleMutex();
}


void MainWindow::updateGUI()
{
    int snapshot_number = model.prms.AnimationFrameNumber();
    int step = model.prms.SimulationStep;
    labelStepCount->setText(QString("%1 (%2)").arg(step, 7, 10, QChar('0')).arg(snapshot_number, 4, 10, QChar('0')));
    labelElapsedTime->setText(QString("%1 s").arg(model.prms.SimulationTime,0,'f',3));
    //statusLabel->setText(QString("per cycle: %1 ms").arg(model.compute_time_per_cycle,0,'f',3));

    if(model.SyncTopologyRequired)
    {
        model.SyncTopologyRequired = false;
        representation.SynchronizeTopology();
    }
    else
    {
        representation.SynchronizeValues();
    }
    renderWindow->Render();
}

void MainWindow::simulation_start_pause(bool checked)
{
    if(!worker->running && checked)
    {
        qDebug() << "starting simulation via GUI";
        statusLabel->setText("starting simulation");

        int snapshot_number = model.prms.AnimationFrameNumber();
        if(snapshot_number == 0)
        {
            snapshot.previous_frame_exists = false;
            snapshot.SaveFrame(animation_frame_directory, snapshot_number);
        }

        worker->Resume();
    }
    else if(worker->running && !checked)
    {
        qDebug() << "pausing simulation via GUI";
        statusLabel->setText("pausing simulation");
        worker->Pause();
        ui->actionStart_Pause->setEnabled(false);
    }
}

void MainWindow::background_worker_paused()
{
    ui->actionStart_Pause->blockSignals(true);
    ui->actionStart_Pause->setEnabled(true);
    ui->actionStart_Pause->setChecked(false);
    ui->actionStart_Pause->blockSignals(false);
    statusLabel->setText("simulation stopped");
}

void MainWindow::screenshot()
{
    if(model.prms.SimulationStep % model.prms.UpdateEveryNthStep) return;
    QString outputPath = QDir::currentPath()+ "/" + screenshot_directory.c_str() + "/" +
                         QString::number(model.prms.AnimationFrameNumber()).rightJustified(5, '0') + ".png";

    QDir pngDir(QDir::currentPath()+ "/"+ screenshot_directory.c_str());
    if(!pngDir.exists()) pngDir.mkdir(QDir::currentPath()+ "/"+ screenshot_directory.c_str());

    renderWindow->DoubleBufferOff();
    renderWindow->Render();
    windowToImageFilter->SetInputBufferTypeToRGBA(); //also record the alpha (transparency) channel
    renderWindow->WaitForCompletion();

    windowToImageFilter->Update();
    windowToImageFilter->Modified();

    writerPNG->Modified();
    writerPNG->SetFileName(outputPath.toUtf8().constData());
    writerPNG->Write();
    renderWindow->DoubleBufferOn();
}
