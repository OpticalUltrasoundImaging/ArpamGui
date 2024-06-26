#include "CoregDisplay.hpp"

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

      m_model(new annotation::AnnotationModel),
      m_annoView(new annotation::AnnotationView),

      actResetZoom(new QAction(QIcon(), "Reset zoom")),
      actCursorDefault(new QAction(QIcon(), "Default")),
      actCursorSelectAScan(new QAction(QIcon(), "Select AScan")),
      actCursorPan(new QAction(QIcon(), "Pan")),
      actCursorUndo(new QAction(QIcon(), "Undo")),
      actCursorLine(new QAction(QIcon(), "Line")),
      actCursorLabelRect(new QAction(QIcon(), "Rect")),
      actCursorLabelFan(new QAction(QIcon(), "Fan"))

{
  // Connect annotation model to canvas
  m_model->setParent(this);
  m_canvasLeft->setModel(m_model);
  m_canvasRight->setModel(m_model);

  // Connect annotation model to the table view
  m_annoView->setModel(m_model);

  // Signals from canvas
  connect(m_canvasLeft, &Canvas::error, this, &CoregDisplay::message);
  connect(m_canvasRight, &Canvas::error, this, &CoregDisplay::message);

  connect(m_canvasLeft, &Canvas::AScanSelected, this,
          &CoregDisplay::AScanSelected);
  connect(m_canvasRight, &Canvas::AScanSelected, this,
          &CoregDisplay::AScanSelected);

  // Connection actions
  connect(actResetZoom, &QAction::triggered, this, [this] {
    m_canvasLeft->scaleToSize();
    m_canvasRight->scaleToSize();
  });

  connect(actCursorUndo, &QAction::triggered, this, &CoregDisplay::undo);

  // Exclusive, checkable actions that change the cursor mode
  using AP = std::pair<QAction *, Canvas::CursorMode>;
  using CursorMode = Canvas::CursorMode;

  const std::array CursorModeActions = {
      AP{actCursorDefault, CursorMode::Default},
      AP{actCursorSelectAScan, CursorMode::SelectAScan},
      AP{actCursorPan, CursorMode::Pan},
      AP{actCursorLine, CursorMode::MeasureLine},
      AP{actCursorLabelRect, CursorMode::LabelRect},
      AP{actCursorLabelFan, CursorMode::LabelFan}};

  // Init state and connect exclusive checking of actions
  for (const auto &[action, cursorMode] : CursorModeActions) {
    action->setCheckable(true);

    connect(action, &QAction::triggered,
            // Capture everything by copy
            [this, action, cursorMode, CursorModeActions] {
              setCursorMode(cursorMode);

              // Uncheck all actions
              for (const auto &[actOther, _] : CursorModeActions) {
                actOther->setChecked(false);
              }

              // Check the current action
              action->setChecked(true);
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
  toolbar->addAction(actCursorSelectAScan);
  toolbar->addAction(actCursorPan);
  toolbar->addAction(actCursorLine);
  toolbar->addAction(actCursorLabelRect);
  toolbar->addAction(actCursorLabelFan);

  // Image Canvas
  auto *hlayout = new QHBoxLayout;
  vlayout->addLayout(hlayout);

  m_canvasLeft->overlay()->setModality("PAUS");
  m_canvasRight->overlay()->setModality("US");

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
