#include "CoregDisplay.hpp"
#include "CanvasAnnotationModel.hpp"
#include "CanvasAnnotationView.hpp"

#include <QAction>
#include <QBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QTableView>
#include <array>
#include <uspam/defer.h>
#include <utility>

CoregDisplay::CoregDisplay(QWidget *parent)
    : QWidget(parent),

      m_canvasLeft(new Canvas(this)), m_canvasRight(new Canvas(this)),

      m_model(new AnnotationModel),

      m_annoView(new AnnotationView),

      actResetZoom(new QAction(QIcon(), "Reset zoom")),
      actCursorDefault(new QAction(QIcon(), "Default")),
      actCursorPan(new QAction(QIcon(), "Pan")),
      actCursorUndo(new QAction(QIcon(), "Undo")),
      actCursorLine(new QAction(QIcon(), "Line")),
      actCursorLabelRect(new QAction(QIcon(), "Rect")),
      actCursorLabelFan(new QAction(QIcon(), "Fan"))

{
  // Connect annotation model
  m_model->setParent(this);
  m_canvasLeft->setModel(m_model);
  m_canvasRight->setModel(m_model);
  m_annoView->setModel(m_model);

  // Signals from canvas
  connect(m_canvasLeft, &Canvas::error, this, &CoregDisplay::message);
  connect(m_canvasRight, &Canvas::error, this, &CoregDisplay::message);

  // Connection actions
  connect(actResetZoom, &QAction::triggered, [this] {
    m_canvasLeft->scaleToSize();
    m_canvasRight->scaleToSize();
  });

  connect(actCursorUndo, &QAction::triggered, this, &CoregDisplay::undo);

  // Exclusive, checkable actions that change the cursor mode
  using AP = std::pair<QAction *, Canvas::CursorMode>;
  using CursorMode = Canvas::CursorMode;

  const std::array CursorModeActions = {
      AP{actCursorDefault, CursorMode::Default},
      AP{actCursorPan, CursorMode::Pan},
      AP{actCursorLine, CursorMode::MeasureLine},
      AP{actCursorLabelRect, CursorMode::LabelRect},
      AP{actCursorLabelFan, CursorMode::LabelFan}};

  // Init state and connect exclusive checking of actions
  for (const auto &[act1, enum1] : CursorModeActions) {
    act1->setCheckable(true);
    connect(act1, &QAction::triggered, [this, enum1, CursorModeActions, act1] {
      setCursorMode(enum1);

      for (const auto &[act2, enum2] : CursorModeActions) {
        act2->setChecked(false);
      }
      act1->setChecked(true);
    });
  }

  defer { actCursorDefault->trigger(); };

  // Setup UI
  auto *vlayout = new QVBoxLayout;
  this->setLayout(vlayout);

  // Toolbar
  auto *toolbar = new QToolBar("Cursor type");
  vlayout->addWidget(toolbar);

  toolbar->addAction(actResetZoom);
  // toolbar->addSeparator();
  // toolbar->addAction(actCursorUndo);
  toolbar->addSeparator();
  toolbar->addAction(actCursorDefault);
  toolbar->addAction(actCursorPan);
  toolbar->addAction(actCursorLine);
  toolbar->addAction(actCursorLabelRect);
  toolbar->addAction(actCursorLabelFan);

  // Image Canvas
  auto *hlayout = new QHBoxLayout;
  vlayout->addLayout(hlayout);

  m_canvasLeft->overlay()->setModality("US");
  m_canvasRight->overlay()->setModality("PAUS");

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
