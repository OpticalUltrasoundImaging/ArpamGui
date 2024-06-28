#pragma once

#include "AScanPlot.hpp"
#include "Annotation/AnnotationModel.hpp"
#include "Annotation/AnnotationView.hpp"
#include "Canvas.hpp"
#include <QLabel>
#include <QTableView>
#include <QWidget>

class CoregDisplay : public QWidget {
  Q_OBJECT
public:
  using AnnotationView = annotation::AnnotationView;
  using AnnotationModel = annotation::AnnotationModel;
  using Annotation = annotation::Annotation;

  explicit CoregDisplay(AScanPlot *ascanPlot, QWidget *parent = nullptr);

  auto annotationView() { return m_annoView; }
  auto model() { return m_model; }

  [[nodiscard]] auto actionDefault() const { return actCursorDefault; }
  [[nodiscard]] auto actionSelectAScan() const { return actCursorSelectAScan; }
  [[nodiscard]] auto actionPan() const { return actCursorPan; }
  [[nodiscard]] auto actionLine() const { return actCursorLine; }
  [[nodiscard]] auto actionLabelRect() const { return actCursorLabelRect; }
  [[nodiscard]] auto actionLabelFan() const { return actCursorLabelFan; }

  [[nodiscard]] auto actionToggleUSCanvas() const { return actToggleUSCanvas; }
  [[nodiscard]] auto actionToggleAScanPlot() const {
    return actToggleAScanPlot;
  }

signals:
  void message(const QString &msg);
  void mouseMoved(QPoint pos, double depth_mm);
  void AScanSelected(int idx);

public slots:
  void setCursorMode(Canvas::CursorMode mode) {
    m_canvasPAUS->setCursorMode(mode);
    m_canvasUS->setCursorMode(mode);
  }

  void undo() {
    m_canvasPAUS->undo();
    m_canvasUS->undo();
  }

  void imshow(const QImage &imgPAUS, const QImage &imgUS, double pix2m);

  void resetZoom() { actResetZoom->trigger(); }
  void resetZoomOnNextImshow() {
    m_canvasPAUS->resetZoomOnNextImshow();
    m_canvasUS->resetZoomOnNextImshow();
  }

  void setSequenceName(const QString &sequence) {
    m_canvasPAUS->overlay()->setSequence(sequence);
    m_canvasUS->overlay()->setSequence(sequence);
  }

  void setMaxIdx(int maxIdx) {
    m_canvasPAUS->overlay()->setMaxIdx(maxIdx);
    m_canvasUS->overlay()->setMaxIdx(maxIdx);
  }

  void setIdx(int idx) {
    m_canvasPAUS->overlay()->setIdx(idx);
    m_canvasUS->overlay()->setIdx(idx);
  }

private:
  Canvas *m_canvasPAUS;
  Canvas *m_canvasUS;
  AScanPlot *m_AScanPlot;

  AnnotationModel *m_model;

  // AnnotationView *m_annoView;
  AnnotationView *m_annoView;

  // Reset image zoom
  QAction *actResetZoom;

  /*
   * Cursor Actions
   */
  // Action to use default cursor
  QAction *actCursorDefault;
  // Action to use cursor to select AScan for plot
  QAction *actCursorSelectAScan;
  // Action to use cursor to pan
  QAction *actCursorPan;
  // Action to undo the last cursor action in the canvas
  QAction *actCursorUndo;
  // Action to set the cursor mode to line measure
  QAction *actCursorLine;
  // Action to set the cursor mode to label rect
  QAction *actCursorLabelRect;
  // Action to set the cursor mode to label fan
  QAction *actCursorLabelFan;

  /*
   * Canvas View Actions
   */
  QAction *actToggleUSCanvas;
  QAction *actToggleAScanPlot;
};
