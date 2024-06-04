#include "CoregDisplay.hpp"
#include "CanvasAnnotationModel.hpp"

#include <QHBoxLayout>
#include <QPushButton>
#include <uspam/defer.h>

CoregDisplay::CoregDisplay(QWidget *parent)
    : QWidget(parent), m_canvasLeft(new Canvas(this)),
      m_canvasRight(new Canvas(this)), m_model(new AnnotationModel),
      actCursorUndo(new QAction(QIcon(), "Undo")),
      actCursorLine(new QAction(QIcon(), "Line")),
      actCursorZoom(new QAction(QIcon(), "Zoom")) {
  m_model->setParent(this);

  // Signals and slots
  connect(actCursorUndo, &QAction::triggered, this, &CoregDisplay::undo);

  actCursorLine->setCheckable(true);
  defer { actCursorLine->trigger(); };
  connect(actCursorLine, &QAction::triggered, [=] {
    m_canvasLeft->setCursorMode(Canvas::CursorMode::LineMeasure);
    m_canvasRight->setCursorMode(Canvas::CursorMode::LineMeasure);

    actCursorLine->setChecked(true);
    actCursorZoom->setChecked(false);
  });

  actCursorZoom->setCheckable(true);
  connect(actCursorZoom, &QAction::triggered, [=] {
    m_canvasLeft->setCursorMode(Canvas::CursorMode::BoxZoom);
    m_canvasRight->setCursorMode(Canvas::CursorMode::BoxZoom);

    actCursorLine->setChecked(false);
    actCursorZoom->setChecked(true);
  });

  // Setup UI
  auto *vlayout = new QVBoxLayout;
  this->setLayout(vlayout);

  // Toolbar
  auto *toolbar = new QToolBar("Cursor type");
  vlayout->addWidget(toolbar);
  toolbar->addAction(actCursorUndo);
  toolbar->addSeparator();
  toolbar->addAction(actCursorLine);
  toolbar->addAction(actCursorZoom);

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