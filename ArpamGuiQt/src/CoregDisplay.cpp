#include "CoregDisplay.hpp"
#include "CanvasAnnotationModel.hpp"

#include <QAction>
#include <QHBoxLayout>
#include <QPushButton>
#include <array>
#include <qtableview.h>
#include <uspam/defer.h>
#include <utility>

CoregDisplay::CoregDisplay(QWidget *parent)
    : QWidget(parent),

      m_canvasLeft(new Canvas(this)), m_canvasRight(new Canvas(this)),

      m_tableView(new QTableView),

      m_model(new AnnotationModel),
      actResetZoom(new QAction(QIcon(), "Reset zoom")),
      actCursorDefault(new QAction(QIcon(), "Default")),
      actCursorPan(new QAction(QIcon(), "Pan")),
      actCursorUndo(new QAction(QIcon(), "Undo")),
      actCursorLine(new QAction(QIcon(), "Line")),
      actCursorLabelRect(new QAction(QIcon(), "Rect"))

{
  // Connect annotation model
  m_model->setParent(this);
  m_canvasLeft->setModel(m_model);
  m_canvasRight->setModel(m_model);
  m_tableView->setModel(m_model);

  // Signals from canvas
  connect(m_canvasLeft, &Canvas::error, this, &CoregDisplay::message);
  connect(m_canvasRight, &Canvas::error, this, &CoregDisplay::message);

  // Connection actions
  connect(actResetZoom, &QAction::triggered, [=] {
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
      AP{actCursorLabelRect, CursorMode::LabelRect}};

  // Init state and connect exclusive checking of actions
  for (const auto &[act1, enum1] : CursorModeActions) {
    act1->setCheckable(true);
    connect(act1, &QAction::triggered, [=] {
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

  m_tableView->show();
}

void CoregDisplay::imshow(const QImage &img1, const QImage &img2,
                          double pix2m) {
  m_canvasLeft->imshow(img1, pix2m);
  m_canvasRight->imshow(img2, pix2m);

  m_canvasLeft->setEnabled(true);
  m_canvasRight->setEnabled(true);
}

void CoregDisplay::closeEvent(QCloseEvent *event) {
  m_tableView->close();
  QWidget::closeEvent(event);
}