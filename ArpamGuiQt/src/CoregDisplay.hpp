#pragma once

#include "Canvas.hpp"
#include "CanvasAnnotationModel.hpp"
#include <QLabel>
#include <QTableView>
#include <QWidget>

class CoregDisplay : public QWidget {
  Q_OBJECT
public:
  explicit CoregDisplay(QWidget *parent = nullptr);

  auto getTableView() { return m_tableView; }

signals:
  void message(const QString &msg);
  void mouseMoved(QPoint pos, double depth_mm);

public slots:
  void setCursorMode(Canvas::CursorMode mode) {
    m_canvasLeft->setCursorMode(mode);
    m_canvasRight->setCursorMode(mode);
  }

  void undo() {
    m_canvasLeft->undo();
    m_canvasRight->undo();
  }

  void imshow(const QImage &img1, const QImage &img2, double pix2m);

  void resetZoom() { actResetZoom->trigger(); }
  void resetZoomOnNextImshow() {
    m_canvasLeft->resetZoomOnNextImshow();
    m_canvasRight->resetZoomOnNextImshow();
  }

protected:
  void closeEvent(QCloseEvent *event) override;

private:
  Canvas *m_canvasLeft;
  Canvas *m_canvasRight;

  QTableView *m_tableView;

  AnnotationModel *m_model;

  // Reset image zoom
  QAction *actResetZoom;

  // Action to use cursor to pan
  QAction *actCursorPan;
  // Action to undo the last cursor action in the canvas
  QAction *actCursorUndo;
  // Action to set the cursor mode to line measure
  QAction *actCursorLine;
  // Action to set the cursor mode to box zoom
  QAction *actCursorLabelRect;
};