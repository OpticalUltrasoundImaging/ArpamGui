#pragma once

#include "Annotation/AnnotationModel.hpp"
#include "Annotation/AnnotationView.hpp"
#include "Canvas.hpp"
#include <QLabel>
#include <QMenu>
#include <QTableView>
#include <QToolBar>
#include <QWidget>

class CoregDisplay : public QWidget {
  Q_OBJECT
public:
  using AnnotationView = annotation::AnnotationView;
  using AnnotationModel = annotation::AnnotationModel;
  using Annotation = annotation::Annotation;

  explicit CoregDisplay(QWidget *parent = nullptr);

  [[nodiscard]] auto annotationView() const { return m_annoView; }
  [[nodiscard]] auto model() const { return m_model; }

  [[nodiscard]] auto cursorToolbar() const { return m_cursorToolbar; }
  [[nodiscard]] auto cursorMenu() const { return m_cursorMenu; }
  [[nodiscard]] auto viewMenu() const { return m_viewMenu; }

  [[nodiscard]] auto actionResetZoom() const { return actResetZoom; }
  [[nodiscard]] auto actionShowUSCanvas() const { return actShowUSCanvas; }

  void setAlinesPerBscan(int alinesPerBscan) {
    m_canvasPAUS->setAlinesPerBscan(alinesPerBscan);
    m_canvasUS->setAlinesPerBscan(alinesPerBscan);
  }

  // Render the current widget view into a QImage
  QImage captureScreenshot();

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

  AnnotationModel *m_model;

  // AnnotationView *m_annoView;
  AnnotationView *m_annoView;

  /*
   * Cursor Actions
   */
  QToolBar *m_cursorToolbar;
  QMenu *m_cursorMenu;

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
  // Menu that holds the view actions
  QMenu *m_viewMenu;
  // Reset image zoom
  QAction *actResetZoom;
  QAction *actShowUSCanvas;
};
