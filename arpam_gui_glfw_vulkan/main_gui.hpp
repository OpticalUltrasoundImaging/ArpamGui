#pragma once

#include <filesystem>
#include <iostream>
#include <memory>

#include <Eigen/Dense>
#include <imgui.h>

#include "libarpam/libarpam.hpp"

#include "utils/utils.hpp"
#include "vulkan_texture_loader.hpp"

namespace arpam_gui {

class MainGui {
public:
  explicit MainGui(VulkanTextureLoader &texture_loader);

  MainGui(MainGui &&) = delete;
  MainGui(const MainGui &) = delete;
  auto operator=(const MainGui &) -> MainGui = delete;
  auto operator=(MainGui &&) -> MainGui = delete;

  ~MainGui() { texture_loader.RemoveTexture(&my_texture); }

  void render();

private:
  VulkanTextureLoader &texture_loader;
  MyTextureData my_texture;
  std::unique_ptr<vk::su::TextureData> texture1;

  // Example states
  bool show_demo_window = true;

  // Buffer for current bin
  Eigen::MatrixX<double> m_rf;
  uint16_t m_rf_idx{0};
};

} // namespace arpam_gui
