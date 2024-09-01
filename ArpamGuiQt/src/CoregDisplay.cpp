#include "CoregDisplay.hpp"

#include "AScanPlot.hpp"
#include <QAction>
#include <QBoxLayout>
#include <QHBoxLayout>
#include <QKeySequence>
#include <QMenu>
#include <QPushButton>
#include <QSizePolicy>
#include <QTableView>
#include <Qt>
#include <array>
#include <qkeysequence.h>
#include <qnamespace.h>
#include <uspam/defer.h>
#include <utility>

CoregDisplay::CoregDisplay(QWidget *parent)
    : QWidget(parent),

      m_canvasPAUS(new Canvas), m_canvasUS(new Canvas),

      m_model(new annotation::AnnotationModel),
      m_annoView(new annotation::AnnotationView),

      m_cursorToolbar(new QToolBar(this)),
      m_cursorMenu(new QMenu("Cursor mode", this)),

      actCursorDefault(new QAction("Default")),
      actCursorSelectAScan(new QAction("Select AScan")),
      actCursorPan(new QAction("Pan")), actCursorUndo(new QAction("Undo")),
      actCursorLine(new QAction("Line")),
      actCursorLabelRect(new QAction("Rect")),
      actCursorLabelFan(new QAction("Fan")),

      m_viewMenu(new QMenu("Frame Display")),
      actResetZoom(new QAction("Reset Zoom")),
      actShowUSCanvas(new QAction("Show US"))

{
  // Connect annotation model to canvas
  m_model->setParent(this);
  m_canvasPAUS->setModel(m_model);
  m_canvasUS->setModel(m_model);

  // Connect annotation model to the table view
  m_annoView->setModel(m_model);

  // Signals from canvas
  connect(m_canvasPAUS, &Canvas::error, this, &CoregDisplay::message);
  connect(m_canvasUS, &Canvas::error, this, &CoregDisplay::message);

  connect(m_canvasPAUS, &Canvas::AScanSelected, this,
          &CoregDisplay::AScanSelected);
  connect(m_canvasUS, &Canvas::AScanSelected, this,
          &CoregDisplay::AScanSelected);

  /*
   *  Connect cursor actions
   */
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

  /*
   * Connect view actions
   */
  actResetZoom->setShortcut({Qt::CTRL | Qt::Key_R});
  connect(actResetZoom, &QAction::triggered, this, [this] {
    m_canvasPAUS->scaleToSize();
    m_canvasUS->scaleToSize();
  });
  m_viewMenu->addAction(actResetZoom);

  actShowUSCanvas->setCheckable(true);
  connect(actShowUSCanvas, &QAction::toggled,
          [this](bool checked) { m_canvasUS->setVisible(checked); });
  m_viewMenu->addAction(actShowUSCanvas);

  /*
   * Setup UI
   */
  auto *vlayout = new QVBoxLayout;
  this->setLayout(vlayout);

  // Toolbar and context menu
  {
    vlayout->addWidget(m_cursorToolbar);

    // toolbar->addSeparator();
    // toolbar->addAction(actCursorUndo);

    m_cursorMenu->setTitle("Cursor mode");

    m_cursorToolbar->addSeparator();

    actCursorDefault->setShortcut(QKeySequence(Qt::ALT | Qt::Key_D));
    m_cursorToolbar->addAction(actCursorDefault);
    m_cursorMenu->addAction(actCursorDefault);

    actCursorSelectAScan->setShortcut(QKeySequence(Qt::ALT | Qt::Key_A));
    m_cursorToolbar->addAction(actCursorSelectAScan);
    m_cursorMenu->addAction(actCursorSelectAScan);

    actCursorPan->setShortcut(QKeySequence(Qt::ALT | Qt::Key_P));
    m_cursorToolbar->addAction(actCursorPan);
    m_cursorMenu->addAction(actCursorPan);

    actCursorLine->setShortcut(QKeySequence(Qt::ALT | Qt::Key_L));
    m_cursorToolbar->addAction(actCursorLine);
    m_cursorMenu->addAction(actCursorLine);

    actCursorLabelRect->setShortcut(QKeySequence(Qt::ALT | Qt::Key_R));
    m_cursorToolbar->addAction(actCursorLabelRect);
    m_cursorMenu->addAction(actCursorLabelRect);

    actCursorLabelFan->setShortcut(QKeySequence(Qt::ALT | Qt::Key_F));
    m_cursorToolbar->addAction(actCursorLabelFan);
    m_cursorMenu->addAction(actCursorLabelFan);

    // Add stretchable spacer
    // toolbar->addSeparator();
    {
      auto *emptyStretchable = new QWidget;
      emptyStretchable->setSizePolicy(QSizePolicy::Expanding,
                                      QSizePolicy::Preferred);
      m_cursorToolbar->addWidget(emptyStretchable);
    }
    m_cursorToolbar->addAction(actResetZoom);
    m_cursorToolbar->addAction(actShowUSCanvas);
  }

  // Image Canvas
  {
    auto *hlayout = new QHBoxLayout;
    vlayout->addLayout(hlayout);

    m_canvasPAUS->overlay()->setModality("PAUS");
    m_canvasUS->overlay()->setModality("US");

    for (auto *const canvas : {m_canvasPAUS, m_canvasUS}) {
      hlayout->addWidget(canvas);
      canvas->setStyleSheet("border: 1px solid black");
      canvas->setDisabled(true);
      connect(canvas, &Canvas::mouseMoved, this, &CoregDisplay::mouseMoved);
    }
  }

  // By default don't show the US canvas and show AScan
  actShowUSCanvas->setChecked(false);
  m_canvasUS->hide();
}

void CoregDisplay::imshow(const QImage &imgPAUS, const QImage &imgUS,
                          double pix2m) {
  m_canvasPAUS->imshow(imgPAUS, pix2m);
  m_canvasUS->imshow(imgUS, pix2m);

  m_canvasPAUS->setEnabled(true);
  m_canvasUS->setEnabled(true);
}

QImage CoregDisplay::captureScreenshot() { return this->grab().toImage(); }
