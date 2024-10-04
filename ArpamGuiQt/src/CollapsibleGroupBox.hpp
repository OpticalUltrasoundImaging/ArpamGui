#pragma once

#include <QGroupBox>
#include <QLabel>
#include <QPushButton>
#include <QToolButton>
#include <QVBoxLayout>
#include <QWidget>

class CollapsibleGroupBox : public QGroupBox {
  Q_OBJECT
public:
  CollapsibleGroupBox(const QString &title, QWidget *parent = nullptr)
      : QGroupBox(title, parent), titleText(title), content(new QWidget) {
    // Set the group box checkable, but hide the checkbox itself
    setCheckable(true);
    setChecked(true);
    // Hides the checkbox
    setStyleSheet("QGroupBox::indicator { width: 0px; height: 0px; }");
    setTitle(getFormattedTitle(true));

    {
      auto *layout = new QVBoxLayout;
      layout->setContentsMargins(0, 0, 0, 0); // No margins
      layout->setSpacing(0);                  // No spacing between widgets
      ((QGroupBox *)this)->setLayout(layout);
      layout->addWidget(content);
    }

    // Connect the toggled signal to the visibility and arrow toggle
    connect(this, &QGroupBox::toggled, this,
            &CollapsibleGroupBox::toggleContents);
  }

  void setLayout(QLayout *layout) { content->setLayout(layout); }
  auto layout() const { return content->layout(); }

private slots:
  void toggleContents(bool checked) {
    setTitle(getFormattedTitle(checked));
    content->setVisible(checked);
  }

private:
  [[nodiscard]] QString getFormattedTitle(bool expanded) const {
    const QString arrow = expanded ? "▼" : "▶";    // Use unicode arrows
    return QString("%1 %2").arg(arrow, titleText); // Arrow + Title
  }

  QString titleText; // Store the original title text
  QWidget *content;
};