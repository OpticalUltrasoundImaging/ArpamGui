#include "CoregDisplay.hpp"
#include "CanvasAnnotationModel.hpp"

#include <QHBoxLayout>
#include <QPushButton>
#include <qaction.h>
#include <uspam/defer.h>

CoregDisplay::CoregDisplay(QWidget *parent)
    : QWidget(parent), m_canvasLeft(new Canvas(this)),
      m_canvasRight(new Canvas(this)), m_model(new AnnotationModel),
      actResetZoom(new QAction(QIcon(), "Reset zoom")),
      actCursorPan(new QAction(QIcon(), "Pan")),
      actCursorUndo(new QAction(QIcon(), "Undo")),
      actCursorLine(new QAction(QIcon(), "Line")),
      actCursorLabelRect(new QAction(QIcon(), "Zoom"))

{
  m_model->setParent(this);

  // Signals and slots
  connect(actResetZoom, &QAction::triggered, [=] {
    m_canvasLeft->scaleToSize();
    m_canvasRight->scaleToSize();
  });

  connect(actCursorUndo, &QAction::triggered, this, &CoregDisplay::undo);

  actCursorPan->setCheckable(true);
  connect(actCursorPan, &QAction::triggered, [=] {
    setCursorMode(Canvas::CursorMode::Pan);

    actCursorPan->setChecked(true);
    actCursorLine->setChecked(false);
    actCursorLabelRect->setChecked(false);
  });

  actCursorLine->setCheckable(true);
  defer { actCursorLine->trigger(); };
  connect(actCursorLine, &QAction::triggered, [=] {
    setCursorMode(Canvas::CursorMode::LineMeasure);

    actCursorPan->setChecked(false);
    actCursorLine->setChecked(true);
    actCursorLabelRect->setChecked(false);
  });

  actCursorLabelRect->setCheckable(true);
  connect(actCursorLabelRect, &QAction::triggered, [=] {
    setCursorMode(Canvas::CursorMode::LabelRect);

    actCursorPan->setChecked(false);
    actCursorLine->setChecked(false);
    actCursorLabelRect->setChecked(true);
  });

  // Setup UI
  auto *vlayout = new QVBoxLayout;
  this->setLayout(vlayout);

  // Toolbar
  auto *toolbar = new QToolBar("Cursor type");
  vlayout->addWidget(toolbar);
  toolbar->addAction(actResetZoom);
  toolbar->addSeparator();
  toolbar->addAction(actCursorUndo);
  toolbar->addSeparator();
  toolbar->addAction(actCursorPan);
  toolbar->addAction(actCursorLine);
  toolbar->addAction(actCursorLabelRect);

  // Image Canvas
  auto *hlayout = new QHBoxLayout;
  vlayout->addLayout(hlayout);

  m_canvasLeft->setName("US");
  m_canvasRight->setName("PAUS");

  for (auto *const canvas : {m_canvasLeft, m_canvasRight}) {
    hlayout->addWidget(canvas);
    canvas->setStyleSheet("border: 1px solid black");
    canvas->setDisabled(true);
    connect(canvas, &Canvas::mouseMoved, this, &CoregDisplay::mouseMoved);
  }
}

void CoregDisplay::imshow(const QImage &img1, const QImage &img2,
                          double pix2m) {
  m_canvasLeft->imshow(img1, pix2m);
  m_canvasRight->imshow(img2, pix2m);

  m_canvasLeft->setEnabled(true);
  m_canvasRight->setEnabled(true);
}