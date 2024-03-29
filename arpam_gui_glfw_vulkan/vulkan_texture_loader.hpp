#pragma once
#include "utils/utils.hpp"
#include <memory>
#include <vulkan/vulkan_core.h>

#include <imgui.h>
#include <imgui_impl_vulkan.h>
#include <vector>
#include <vulkan/vulkan.h> // TODO: remove
#include <vulkan/vulkan.hpp>

// A struct to manage data related to one image in vulkan
struct MyTextureData {
  VkDescriptorSet ds{}; // Descriptor set: this is what you'll pass to Image()
  int width{};
  int height{};
  int channels{};

  // Need to keep track of these to properly cleanup
  vk::ImageView imageView{};
  vk::Image image{};
  vk::DeviceMemory imageMemory{};
  vk::Sampler sampler{};

  vk::Buffer uploadBuffer{};
  vk::DeviceMemory uploadBufferMemory{};
};

class VulkanTextureLoader {
public:
  using check_vk_result_fn = void (*)(VkResult);

  VulkanTextureLoader(vk::PhysicalDevice physical_device, vk::Device device,
                      vk::CommandPool command_pool, vk::Queue queue,
                      vk::AllocationCallbacks *allocator);
  ~VulkanTextureLoader() = default;

  VulkanTextureLoader(const VulkanTextureLoader &) = delete;
  auto operator=(const VulkanTextureLoader &) -> VulkanTextureLoader & = delete;

  VulkanTextureLoader(VulkanTextureLoader &&) = delete;
  auto operator=(VulkanTextureLoader &&) -> VulkanTextureLoader & = delete;

  void loadTexture(const uint8_t *image_data, uint32_t width, uint32_t height,
                   uint32_t channels, vk::Format format,
                   MyTextureData *texData);

  // Helper function to load an image with common settings and return a
  // MyTextureData with a VkDescriptorSet as a sort of Vulkan pointer
  void LoadTextureFromFile(const char *filename, MyTextureData *texData);

  auto CreateTextureFromBuffer(const uint8_t *bufferData, uint32_t width,
                               uint32_t height, uint32_t channels,
                               vk::Format format, MyTextureData *texData)
      -> bool;

  // Helper function to cleanup an image loaded with LoadTextureFromFile
  void RemoveTexture(MyTextureData *tex_data) const;

  vk::PhysicalDevice physical_device;
  vk::Device device;
  vk::CommandPool command_pool;
  vk::Queue queue;
  vk::AllocationCallbacks *allocator;

private:
  // Helper function to find Vulkan memory type bits. See
  // ImGui_ImplVulkan_MemoryType() in imgui_impl_vulkan.cpp
  [[nodiscard]] auto findMemoryType(uint32_t type_filter,
                                    vk::MemoryPropertyFlags properties) const
      -> uint32_t;
};