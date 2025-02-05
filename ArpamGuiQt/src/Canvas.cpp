#include "Canvas.hpp"
#include "Annotation/AnnotationModel.hpp"
#include "Annotation/GraphicsItemBase.hpp"
#include "Annotation/GraphicsItems.hpp"
#include "geometryUtils.hpp"
#include <QGestureEvent>
#include <QGraphicsView>
#include <QGraphicsWidget>
#include <QHBoxLayout>
#include <QImage>
#include <QInputDialog>
#include <QPainter>
#include <QPinchGesture>
#include <QVBoxLayout>
#include <Qt>
#include <QtCore>
#include <QtDebug>
#include <QtLogging>
#include <uspam/timeit.hpp>

Canvas::Canvas(QWidget *parent)
    : QGraphicsView(parent), m_scene(new QGraphicsScene),
      m_overlay(new CanvasOverlay(viewport()))

{
  setBackgroundBrush(QBrush(Qt::black));

  setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
  setAlignment(Qt::AlignCenter);

  // Hide scrollbars
  setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
  setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);

  // Enable mouse tracking
  setMouseTracking(true);
  setCursor(Qt::CrossCursor);
  setFocusPolicy(Qt::StrongFocus);

  // Enable pinch gesture handling
  // setDragMode(DragMode::ScrollHandDrag);
  viewport()->setAttribute(Qt::WA_AcceptTouchEvents);
  grabGesture(Qt::PinchGesture);

  // Graphics rendering parameters
  setRenderHints(QPainter::Antialiasing | QPainter::SmoothPixmapTransform);
  setScene(m_scene);

  // Hide the overlay at init since no images are shown yet.
  m_overlay->hide();
}

void Canvas::setModel(AnnotationModel *model) {
  this->m_model = model;

  // When existing data changes
  connect(model, &AnnotationModel::dataChanged, this, &Canvas::onDataChanged);

  // When existing data is removed
  connect(model, &AnnotationModel::rowsRemoved, this, &Canvas::onRowsRemoved);

  // When new data is inserted
  connect(model, &AnnotationModel::rowsInserted, this, &Canvas::onRowsInserted);

  // // When existing data is moved
  // connect(model, &AnnotationModel::rowsMoved, this, &Canvas::onRowsMoved);

  // Right click context menu popup
  this->setContextMenuPolicy(Qt::CustomContextMenu);
  connect(this, &QGraphicsView::customContextMenuRequested,
          [this](const QPoint &pos) {
            // Item under cursor
            if (const int rowIdx =
                    static_cast<int>(m_graphicsItems.indexOf(itemAt(pos)));
                rowIdx >= 0) {

              // Create context menu
              QMenu contextMenu;

              QAction deleteAction("Delete", &contextMenu);
              contextMenu.addAction(&deleteAction);
              connect(&deleteAction, &QAction::triggered,
                      [this, rowIdx] { m_model->removeRow(rowIdx); });

              QAction renameAction("Rename", &contextMenu);
              contextMenu.addAction(&renameAction);
              connect(&renameAction, &QAction::triggered, [this, rowIdx] {
                openDialogToUpdateAnnotationName(rowIdx);
              });

              contextMenu.exec(viewport()->mapToGlobal(pos));
            }
          });
}

void Canvas::imshow(const cv::Mat &cv_img, double pix2m) {
  QImage qi(cv_img.data, cv_img.cols, cv_img.rows,
            static_cast<int>(cv_img.step), QImage::Format_BGR888);
  this->imshow(QPixmap::fromImage(qi), pix2m);
}

void Canvas::updateMinScaleFactor() {
  if (m_Pixmap.isNull()) [[unlikely]] {
    return;
  }

  const auto w = width();
  const auto h = height();
  const auto pw = m_Pixmap.width();
  const auto ph = m_Pixmap.height();

  // Calculate the scaleFactor needed to fill the viewport with the image
  // This is also the minimum scale factor
  const qreal scale =
      qMin(w / static_cast<qreal>(pw), h / static_cast<qreal>(ph));
  m_scaleFactorMin = scale;
}

void Canvas::scaleToSize() {
  updateMinScaleFactor();
  m_scaleFactor = m_scaleFactorMin;

  updateTransform();
}

void Canvas::imshow(const QImage &img, double pix2mm) {
  this->imshow(QPixmap::fromImage(img), pix2mm);
}

void Canvas::imshow(const QPixmap &pixmap, double pix2mm) {
  m_Pixmap = pixmap;
  m_pix2mm = pix2mm;

  // Remove current image
  if (m_PixmapItem != nullptr) {
    m_scene->removeItem(m_PixmapItem);
    delete m_PixmapItem;
  }

  // Add image pixmap to scene
  m_PixmapItem = m_scene->addPixmap(m_Pixmap);
  // Use negative Z value to make sure pixmap is at the bottom
  m_PixmapItem->setZValue(-1);

  if (m_resetZoomOnNextImshow) {
    scaleToSize();
    m_resetZoomOnNextImshow = false;
  }

  m_overlay->setSize(m_Pixmap.size());
  m_overlay->show();
}

// NOLINTBEGIN(*-casting, *-narrowing-conversions)

// Compute the distance between two points in the scaled pixmap domain
double Canvas::computeDistance_mm(QPointF pt1, QPointF pt2) const {
  const auto distance = geometry::calcMagnitude(pt1 - pt2);
  const auto distance_mm = distance * m_pix2mm;
  return distance_mm;
}

bool Canvas::event(QEvent *event) {
  if (event->type() == QEvent::Gesture) {
    return gestureEvent(
        static_cast<QGestureEvent *>(event)); // NOLINT(*downcast)
  }
  return QGraphicsView::event(event);
}

void Canvas::paintEvent(QPaintEvent *event) {
  uspam::TimeIt timeit;

  m_overlay->move(0, 0);
  QGraphicsView::paintEvent(event);

  // QPainter painter(this);

  // Measure rendering time
  // {
  // const auto renderTime_ms = timeit.get_ms();
  // const auto msg = QString("Rendering time %1 ms").arg(renderTime_ms);
  // emit error(msg);
  // }
}

void Canvas::undo() {
  // switch (m_cursorMode) {

  // case CursorMode::MeasureLine:
  //   break;

  // case CursorMode::LabelRect:
  //   break;

  // default:
  //   break;
  // }
}

void Canvas::mousePressEvent(QMouseEvent *event) {
  // Compute position in the pixmap domain
  m_cursor.startPos = m_PixmapItem->mapFromScene(mapToScene(event->pos()));
  m_cursor.pos = m_cursor.startPos;

  if (event->button() == Qt::LeftButton) {
    m_cursor.leftButtonDown = true;

    switch (m_cursorMode) {
    case (CursorMode::Default):
      QGraphicsView::mousePressEvent(event);
      break;

    case CursorMode::SelectAScan: {
      // Select an AScan with the cursor
      // Add a graphics item represending the AScan
      // Emit a signal to tell the AScan plot

      const auto [idx, line] =
          m_cursor.selectAScan(m_Pixmap.rect(), m_alinesPerBscan);

      // Insert ALine graphics here in canvas
      const auto color = Qt::green;
      m_currItem = new annotation::LineItem(Annotation(line, color, ""));
      m_currItem->updateScaleFactor(m_scaleFactor);
      m_currItem->setPenStyle(Qt::PenStyle::DashLine);
      m_scene->addItem(m_currItem);

      // Emit
      emit AScanSelected(idx);
    } break;

    case CursorMode::Pan:
      panStartEvent(event);
      break;

    case CursorMode::MeasureLine: {
      event->accept();

      const auto line = m_cursor.line();
      const auto color = Qt::white;

      m_currItem = new annotation::LineItem(Annotation(line, color, ""));
      m_currItem->updateScaleFactor(m_scaleFactor);
      m_scene->addItem(m_currItem);

    } break;

    case CursorMode::LabelRect: {
      event->accept();

      const auto rect = m_cursor.rect();

      m_currItem = new annotation::RectItem(Annotation(rect, Qt::white));
      m_currItem->updateScaleFactor(m_scaleFactor);
      m_scene->addItem(m_currItem);

    } break;

    case CursorMode::LabelFan: {
      event->accept();

      // Convert mouse pos to angle
      m_cursor.angleOffset = 0;
      m_cursor.lastAngle = 180.0; // NOLINT(*-magic-numbers)
      const auto angle = m_cursor.angleDeg(m_Pixmap.rect());

      const auto anno =
          Annotation(annotation::Arc{angle, 0}, m_Pixmap.rect(), Qt::white);
      m_currItem = new annotation::FanItem(anno);
      m_currItem->updateScaleFactor(m_scaleFactor);
      m_scene->addItem(m_currItem);

    } break;
    }

  } else if (event->button() == Qt::MiddleButton) {
    // Middle button pan
    m_cursor.middleButtonDown = true;
    panStartEvent(event);

  } else if (event->button() == Qt::RightButton) {
    m_cursor.rightButtonDown = true;
    // event->accept();
    // undo();

    QGraphicsView::mousePressEvent(event);
  }
}

void Canvas::mouseMoveEvent(QMouseEvent *event) {
  // Compute position in the pixmap domain
  m_cursor.pos = m_PixmapItem->mapFromScene(mapToScene(event->pos()));

  // [px] Compute distance to center
  const auto center = m_Pixmap.rect().center();
  const auto distanceToCenter_mm = computeDistance_mm(center, m_cursor.pos);
  emit mouseMoved(m_cursor.pos.toPoint(), distanceToCenter_mm);

  if (m_cursor.leftButtonDown) {
    switch (m_cursorMode) {
    case CursorMode::Default:
      QGraphicsView::mouseMoveEvent(event);
      break;

    case (CursorMode::SelectAScan): {
      const auto [idx, line] =
          m_cursor.selectAScan(m_Pixmap.rect(), m_alinesPerBscan);

      // Move ALine graphics here in canvas
      if (auto *lineItem = dynamic_cast<annotation::LineItem *>(m_currItem);
          lineItem != nullptr) [[likely]] {
        lineItem->setLine(line);
      }

      // Emit
      emit AScanSelected(idx);

    } break;

    case CursorMode::Pan: {
      panMoveEvent(event);
    } break;

    case CursorMode::MeasureLine: {
      event->accept();

      if (auto *item = dynamic_cast<annotation::LineItem *>(m_currItem);
          item != nullptr) [[likely]] {

        const auto line = m_cursor.line();
        const auto dist = computeDistance_mm(line.p1(), line.p2());
        item->setLine(line);
        // NOLINTNEXTLINE(*-magic-numbers)
        item->setText(QString("%1 mm").arg(dist, 5, 'f', 2));
      }
    } break;

    case CursorMode::LabelRect: {
      event->accept();
      if (auto *item = dynamic_cast<annotation::RectItem *>(m_currItem);
          item != nullptr) [[likely]] {
        item->setRect(m_cursor.rect());
      }
    } break;

    case CursorMode::LabelFan: {
      event->accept();
      if (auto *item = dynamic_cast<annotation::FanItem *>(m_currItem);
          item != nullptr) [[likely]] {

        constexpr double FULL_CIRCLE = 360.0;
        constexpr double UPPER_THRESH = 340.0;
        constexpr double LOWER_THRESH = 20.0;

        const double angle = m_cursor.angleDeg(m_Pixmap.rect());

        if ((m_cursor.lastAngle > UPPER_THRESH && angle < LOWER_THRESH)) {
          m_cursor.angleOffset += FULL_CIRCLE;
        } else if (m_cursor.lastAngle < LOWER_THRESH && angle > UPPER_THRESH) {
          m_cursor.angleOffset -= FULL_CIRCLE;
        }
        m_cursor.lastAngle = angle;

        item->setSpanAngle(angle + m_cursor.angleOffset - item->startAngle());
      }
    } break;
    }

  } else if (m_cursor.middleButtonDown) {
    // Panning
    panMoveEvent(event);

  } else {
    QGraphicsView::mouseMoveEvent(event);
  }
}

void Canvas::mouseReleaseEvent(QMouseEvent *event) {
  if (event->button() == Qt::LeftButton) {
    m_cursor.leftButtonDown = false;

    switch (m_cursorMode) {

    case CursorMode::Default:
      QGraphicsView::mouseReleaseEvent(event);
      break;

    case (CursorMode::SelectAScan):
      // Remove graphics Item
      m_scene->removeItem(m_currItem);
      delete m_currItem;
      m_currItem = nullptr;

      break;

    case CursorMode::Pan:
      panEndEvent(event);
      break;

    case CursorMode::MeasureLine:
    case CursorMode::LabelRect:
    case CursorMode::LabelFan: {

      // Save annotation to the data model
      // Saving to the model automatically triggers the creation of the correct
      // graphics items, so we can delete the current working item.
      if (m_currItem != nullptr) {
        const auto anno = m_currItem->annotation();

        // If annotation is too small, ignore
        if (!anno.tooSmall()) {
          m_model->addAnnotation(anno);
        }

        m_scene->removeItem(m_currItem);
        delete m_currItem;
        m_currItem = nullptr;
      }
    } break;
    }
  } else if (event->button() == Qt::MiddleButton) {
    m_cursor.middleButtonDown = false;
    panEndEvent(event);
  } else if (event->button() == Qt::RightButton) {
    m_cursor.rightButtonDown = false;
    QGraphicsView::mouseReleaseEvent(event);
  }
}

void Canvas::openDialogToUpdateAnnotationName(int rowIdx) {
  auto anno = m_model->getAnnotation(rowIdx);

  bool ok{true};
  const auto inp = QInputDialog::getText(this, "Update annotation", "Name",
                                         QLineEdit::Normal, anno.name, &ok);
  if (ok) {
    anno.name = inp;
    const auto index = m_model->index(rowIdx);
    m_model->setData(index, inp, AnnotationModel::NameRole);
  }
}

void Canvas::mouseDoubleClickEvent(QMouseEvent *event) {
  if (event->button() == Qt::LeftButton) {
    const auto pos = event->pos();

    // Item under cursor
    if (const int rowIdx =
            static_cast<int>(m_graphicsItems.indexOf(itemAt(pos)));
        rowIdx >= 0) {
      openDialogToUpdateAnnotationName(rowIdx);
    }
  } else {
    QGraphicsView::mouseDoubleClickEvent(event);
  }
}

void Canvas::wheelEvent(QWheelEvent *event) {
  const auto WHEEL_ZOOM_MODIFIER = Qt::ControlModifier;

  if (event->modifiers().testFlag(WHEEL_ZOOM_MODIFIER)) {
    // modifier + scroll -> Zoom
    event->accept();

    // Calculate the scale factor adjustment
    const double numDegrees = event->angleDelta().y() / 8.0;
    const double numSteps = numDegrees / 15.0;
    const double sensitivity = 0.1;
    const double scaleFactor = 1.0 - numSteps * sensitivity;
    m_scaleFactor = std::max(m_scaleFactor * scaleFactor, m_scaleFactorMin);

    updateTransform();
  } else {
    QGraphicsView::wheelEvent(event);
  }
}

bool Canvas::gestureEvent(QGestureEvent *event) {
  if (QGesture *pinch = event->gesture(Qt::PinchGesture)) {
    pinchTriggered(
        static_cast<QPinchGesture *>(pinch)); // NOLINT(*-static-cast-downcast)
  }
  return true;
}

void Canvas::pinchTriggered(QPinchGesture *gesture) {
  if (gesture->state() == Qt::GestureState::GestureUpdated) {
    const qreal scaleFactor = gesture->scaleFactor();
    m_scaleFactor = std::max(m_scaleFactor * scaleFactor, m_scaleFactorMin);
    updateTransform();
  }
}

void Canvas::panStartEvent(QMouseEvent *event) {
  event->accept();
  m_lastPanPoint = event->pos();

  m_panLastCursor = cursor();
  setCursor(Qt::ClosedHandCursor);
}

void Canvas::panMoveEvent(QMouseEvent *event) {
  event->accept();
  const auto delta = event->pos() - m_lastPanPoint;
  horizontalScrollBar()->setValue(horizontalScrollBar()->value() - delta.x());
  verticalScrollBar()->setValue(verticalScrollBar()->value() - delta.y());
  m_lastPanPoint = event->pos();
}

void Canvas::panEndEvent(QMouseEvent *event) { setCursor(m_panLastCursor); }

void Canvas::updateTransform() {
  // Update transform for the QGraphicsView
  setTransformationAnchor(AnchorUnderMouse);
  auto transform = QTransform();
  transform.scale(m_scaleFactor, m_scaleFactor);
  setTransform(transform);

  // Update font factor for the annotation text labels
  for (auto *item : m_graphicsItems) {
    item->updateScaleFactor(m_scaleFactor);
  }

  // Update zoom factor displayed in the overlay
  m_overlay->setZoom(m_scaleFactor);
}

void Canvas::resizeEvent(QResizeEvent *event) {
  updateMinScaleFactor();

  QGraphicsView::resizeEvent(event);

  m_overlay->resize(viewport()->size());
}

void Canvas::setCursorMode(CursorMode mode) {
  m_cursorMode = mode;

  // Set cursor
  switch (mode) {
  case CursorMode::Pan:
    setCursor(Qt::OpenHandCursor);
    break;
  default:
    setCursor(Qt::CrossCursor);
  }
}

void Canvas::addGraphicsItemFromModel(int row) {
  auto *item = annotation::makeGraphicsItem(m_model->getAnnotation(row));
  item->updateScaleFactor(m_scaleFactor);
  m_scene->addItem(item);
  m_graphicsItems.append(item);
}

void Canvas::updateGraphicsItemFromModel(int row) {
  auto *item = m_graphicsItems[row];
  const auto &annotation = m_model->getAnnotation(row);
  item->updateAnnotation(annotation);
}

void Canvas::removeGraphicsItem(int row) {
  auto *item = m_graphicsItems.takeAt(row);
  m_scene->removeItem(item);
  delete item;
}

// NOLINTEND(*-casting, *-narrowing-conversions)
void Canvas::removeCurrItem() {
  if (m_currItem != nullptr) {
    m_scene->removeItem(m_currItem);
    delete m_currItem;
    m_currItem = nullptr;
  }
}

void Canvas::onDataChanged(const QModelIndex &topLeft,
                           const QModelIndex &bottomRight,
                           const QVector<int> &roles) {
  for (int row = topLeft.row(); row <= bottomRight.row(); ++row) {
    updateGraphicsItemFromModel(row);
  }
}

void Canvas::onRowsInserted(const QModelIndex &parent, int first, int last) {
  Q_UNUSED(parent);
  for (int row = first; row <= last; ++row) {
    addGraphicsItemFromModel(row);
  }
}

void Canvas::onRowsRemoved(const QModelIndex &parent, int first, int last) {
  Q_UNUSED(parent);
  assert((first >= 0) && (first <= last)); // NOLINT
  assert(last < m_graphicsItems.size());
  for (int row = last; row >= first; --row) {
    removeGraphicsItem(row);
  }
}
