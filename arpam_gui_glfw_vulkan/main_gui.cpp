#include <imgui.h>

#include <implot.h>

#include "main_gui.hpp"

namespace arpam_gui {

void MainGui::render() {
  // 1. Show the big demo window (Most of the sample code is in
  // ImGui::ShowDemoWindow()!
  if (show_demo_window) {
    ImGui::ShowDemoWindow(&show_demo_window);
  }

  {
    ImGui::Begin("Settings");
    ImGui::End();
  }

  { ImPlot::ShowDemoWindow(); }
}

} // namespace arpam_gui
