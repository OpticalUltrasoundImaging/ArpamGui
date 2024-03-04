#pragma once

#include <filesystem>
#include <iostream>

#include <Eigen/Dense>
#include <imgui.h>

#include "libarpam/libarpam.hpp"

namespace arpam_gui {

class MainGui {
public:
  explicit MainGui(ImGuiIO &io) : io(io) {
    std::filesystem::path filepath(
        "/Users/tnie/Downloads/135245/135245/NormalUS4.bin");

    if (!std::filesystem::exists(filepath)) {
      std::cerr << "File doesn't exist: " << filepath << "\n";
    } else {
      m_rf = arpam::io::load_bin<double>(filepath.c_str(), std::endian::big);
      // for (int i = 0; i < rf.cols(); i++) {
      // Print the first column
      int i = 0;
      const auto col = m_rf.col(i).head(20);
      std::cout << col << "\n";
      // }

      Eigen::MatrixX<double> result(m_rf.rows(), m_rf.cols());
      std::cout << "recon...\n";
      arpam::recon::recon(m_rf, result);

      std::cout << "Result: " << result.col(0).head(10);
    }
  }

  MainGui(MainGui &&) = delete;
  MainGui(const MainGui &) = delete;
  auto operator=(const MainGui &) -> MainGui = delete;
  auto operator=(MainGui &&) -> MainGui = delete;

  ~MainGui() {}

  void render();

private:
  ImGuiIO &io;

  // Example states
  bool show_demo_window = true;

  // Buffer for current bin
  Eigen::MatrixX<double> m_rf;
};

} // namespace arpam_gui
