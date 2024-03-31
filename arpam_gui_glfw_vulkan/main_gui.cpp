#include <array>

#include <imgui.h>
#include <implot.h>

#include "libarpam/io.hpp"
#include "stb_image.h"

#include "libarpam/libarpam.hpp"
#include "libarpam/recon.hpp"
#include "main_gui.hpp"

namespace arpam_gui {

MainGui::MainGui(VulkanTextureLoader &texture_loader)
    : texture_loader(texture_loader) {
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

    Eigen::MatrixX<uint8_t> result(m_rf.rows(), m_rf.cols());
    std::cout << "recon...\n";

    {
      // Get FIR filter kernel
      const int numtaps = 65;
      Eigen::ArrayXd freq(4);
      freq << 0., 0.1, 0.3, 1.;
      Eigen::ArrayXd gain(4);
      gain << 0., 1., 1., 0.;
      arpam::recon::ReconParams params{
          arpam::recon::FIRFilterParams{numtaps, freq, gain}, 45.0F};

      arpam::recon::recon(params, m_rf, result);
    }

    std::cout << "Result: " << result.col(0).head(10);

    // texture_loader.LoadTextureFromFile("cyberpunk-buildings-cityscape-art.jpg",
    //                                    &my_texture);

    texture_loader.loadTexture(result.data(), result.rows(), result.cols(), 1,
                               vk::Format::eR8Unorm, &my_texture);
    arpam::io::to_bin("output.bin",
                      std::span<const uint8_t>{
                          result.data(), static_cast<size_t>(result.size())});
  }
}

void MainGui::render() {
  // 1. Show the big demo window (Most of the sample code is in
  // ImGui::ShowDemoWindow()!
  if (show_demo_window) {
    ImGui::ShowDemoWindow(&show_demo_window);
  }

  {
    ImGui::Begin("Vulkan Texture Test");
    ImGui::Text("pointer = %p", my_texture.ds);
    ImGui::Text("size = %d x %d", my_texture.width, my_texture.height);

    // ImVec2 displaySize(my_texture.width, my_texture.height);
    static ImVec2 displaySize{512, 512};

    float width = displaySize.x;
    float height = displaySize.y;
    if (ImGui::SliderFloat("Width", &width, 1.0F, 1000.0F)) {
      displaySize.x = width;
    }
    if (ImGui::SliderFloat("Height", &height, 1.0F, 1000.0F)) {
      displaySize.y = height;
    }

    ImGui::Image((ImTextureID)my_texture.ds, displaySize);

    ImGui::End();
  }

  {
    ImGui::Begin("Settings");

    ImGui::End();
  }

  { ImPlot::ShowDemoWindow(); }

  {
    // 65.536 KB of stack memory
    static std::array<double, 8192> vizbuf;

    static bool use_filter{false};
    static bool use_hilbert{false};
    static bool use_logcompression{false};
    static float dynamic_range{25.0F};

    ImGui::Begin("A scan");

    ImGui::Checkbox("FIR filter", &use_filter);
    // ImGui::SameLine()
    ImGui::Checkbox("Hilbert", &use_hilbert);
    ImGui::Checkbox("Log compress", &use_logcompression);
    ImGui::SameLine();
    ImGui::SliderFloat("Dynamic range", &dynamic_range, 0.0f, 60.0f);

    const size_t aline_size = m_rf.rows();

    // Get FIR filter kernel
    const int numtaps = 65;
    Eigen::ArrayXd freq(4);
    freq << 0., 0.1, 0.3, 1.;
    Eigen::ArrayXd gain(4);
    gain << 0., 1., 1., 0.;

    arpam::recon::pipeline::FIRFilter fir_filter(aline_size,
                                                 {numtaps, freq, gain});
    arpam::recon::pipeline::EnvelopeDetection envelope_detection(aline_size);

    arpam::recon::pipeline::LogCompression log_compress(aline_size,
                                                        dynamic_range);

    // Plot A-line RF
    if (ImPlot::BeginPlot("RF signal")) {
      const std::span<const double> rf_line{m_rf.col(m_rf_idx).data(),
                                            aline_size};

      if (use_filter) {
        fir_filter.forward(rf_line, vizbuf);
      } else {
        std::copy(rf_line.begin(), rf_line.end(), vizbuf.begin());
      }

      if (use_hilbert) {
        envelope_detection.forward(vizbuf, vizbuf);
      }

      if (use_logcompression) {
        log_compress.set_db(static_cast<double>(dynamic_range));
        log_compress.forward(vizbuf, vizbuf);
      }

      ImPlot::SetupAxes("Samples", "Val");
      ImPlot::PlotLine("RF", vizbuf.data(), static_cast<int>(vizbuf.size()));
      ImPlot::EndPlot();

      // Increment counter
      // TODO increment at 1000 Hz for testing
      m_rf_idx = (m_rf_idx + 1) % (m_rf.cols());
    }

    ImGui::End();
  }
}

} // namespace arpam_gui
