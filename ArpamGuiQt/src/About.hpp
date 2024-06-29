#pragma once

#include <QString>
#include <QWidget>

namespace arpam_about {

auto aboutString() -> QString;
void showAboutDialog(QWidget *parent);
} // namespace arpam_about
