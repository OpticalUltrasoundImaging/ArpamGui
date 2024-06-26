#include "CoregDisplay.hpp"

#include "AScanPlot.hpp"
#include <QAction>
#include <QBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QTableView>
#include <array>
#include <qsizepolicy.h>
#include <uspam/defer.h>
#include <utility>

CoregDisplay::CoregDisplay(AScanPlot *ascanPlot, QWidget *parent)
    : QWidget(parent),

      m_canvasPAUS(new Canvas(this)), m_canvasUS(new Canvas(this)),
      m_AScanPlot(ascanPlot),

      m_model(new annotation::AnnotationModel),
      m_annoView(new annotation::AnnotationView),

      actResetZoom(new QAction(QIcon(), "Reset zoom")),
      actCursorDefault(new QAction(QIcon(), "Default")),
      actCursorSelectAScan(new QAction(QIcon(), "Select AScan")),
      actCursorPan(new QAction(QIcon(), "Pan")),
      actCursorUndo(new QAction(QIcon(), "Undo")),
      actCursorLine(new QAction(QIcon(), "Line")),
      actCursorLabelRect(new QAction(QIcon(), "Rect")),
      actCursorLabelFan(new QAction(QIcon(), "Fan")),

      actToggleUSCanvas(new QAction(QIcon(), "Toggle US")),
      actToggleAScanPlot(new QAction(QIcon(), "Toggle AScan"))

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

  // Connection actions
  connect(actResetZoom, &QAction::triggered, this, [this] {
    m_canvasPAUS->scaleToSize();
    m_canvasUS->scaleToSize();
  });

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
   *  Connect cursor actions
   */
  connect(actToggleUSCanvas, &QAction::triggered,
          [this] { m_canvasUS->setVisible(!m_canvasUS->isVisible()); });
  connect(actToggleAScanPlot, &QAction::triggered,
          [this] { m_AScanPlot->setVisible(!m_AScanPlot->isVisible()); });

  /*
   * Setup UI
   */
  auto *vlayout = new QVBoxLayout;
  this->setLayout(vlayout);

  // Toolbar
  {
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

    // Add stretchable spacer
    // toolbar->addSeparator();
    {
      auto *emptyStretchable = new QWidget;
      emptyStretchable->setSizePolicy(QSizePolicy::Expanding,
                                      QSizePolicy::Preferred);
      toolbar->addWidget(emptyStretchable);
    }
    toolbar->addAction(actToggleUSCanvas);
    toolbar->addAction(actToggleAScanPlot);
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

    // hlayout->addWidget(m_AScanPlot);
  }
}

void CoregDisplay::imshow(const QImage &imgPAUS, const QImage &imgUS,
                          double pix2m) {
  m_canvasPAUS->imshow(imgPAUS, pix2m);
  m_canvasUS->imshow(imgUS, pix2m);

  m_canvasPAUS->setEnabled(true);
  m_canvasUS->setEnabled(true);
}
