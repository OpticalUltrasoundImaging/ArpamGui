#include "vulkan_texture_loader.hpp"
#include <vulkan/vulkan.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "utils/utils.hpp"

VulkanTextureLoader::VulkanTextureLoader(vk::PhysicalDevice physical_device,
                                         vk::Device device,
                                         vk::CommandPool command_pool,
                                         vk::Queue queue,
                                         vk::AllocationCallbacks *allocator)
    : physical_device(physical_device), device(device),
      command_pool(command_pool), queue(queue), allocator(allocator) {}

void VulkanTextureLoader::loadTexture(const uint8_t *image_data, uint32_t width,
                                      uint32_t height, uint32_t channels,
                                      vk::Format format,
                                      MyTextureData *texData) {
  // Calculate allocation size (in number of bytes)
  const size_t image_size = width * height * channels;
  texData->width = width;
  texData->height = height;
  texData->channels = channels;

  // Create the Vulkan image
  texData->image = device.createImage(
      vk::ImageCreateInfo(
          {}, vk::ImageType::e2D, format, vk::Extent3D{width, height, 1}, 1, 1,
          vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal,
          vk::ImageUsageFlagBits::eSampled |
              vk::ImageUsageFlagBits::eTransferDst,
          vk::SharingMode::eExclusive, {}, {}, vk::ImageLayout::eUndefined),
      allocator);

  // Allocate memory for the image
  vk::MemoryRequirements memRequirements =
      device.getImageMemoryRequirements(texData->image);

  texData->imageMemory = device.allocateMemory(
      vk::MemoryAllocateInfo(
          memRequirements.size,
          findMemoryType(memRequirements.memoryTypeBits,
                         vk::MemoryPropertyFlagBits::eDeviceLocal)),
      allocator);

  // Create the Image View
  texData->imageView = device.createImageView(
      vk::ImageViewCreateInfo{
          {},
          texData->image,
          vk::ImageViewType::e2D,
          format,
          {},
          vk::ImageSubresourceRange{
              vk::ImageAspectFlagBits::eColor, {}, 1, {}, 1}},
      allocator);

  // Create the sampler
  texData->sampler = device.createSampler(
      vk::SamplerCreateInfo{{},
                            vk::Filter::eLinear,
                            vk::Filter::eLinear,
                            vk::SamplerMipmapMode::eLinear,
                            vk::SamplerAddressMode::eRepeat,
                            vk::SamplerAddressMode::eRepeat,
                            vk::SamplerAddressMode::eRepeat,
                            {},
                            {},
                            1.0f,
                            {},
                            {},
                            -1000,
                            1000},
      allocator);

  // Create Descriptor Set using ImGUI's implementation
  // TODO(tnie): create custom descriptor set
  texData->ds =
      ImGui_ImplVulkan_AddTexture(texData->sampler, texData->imageView,
                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

  // Create Upload Buffer
  {
    texData->uploadBuffer = device.createBuffer(
        vk::BufferCreateInfo{{},
                             image_size,
                             vk::BufferUsageFlagBits::eTransferSrc,
                             vk::SharingMode::eExclusive},
        allocator);

    const auto req = device.getBufferMemoryRequirements(texData->uploadBuffer);
    texData->uploadBufferMemory = device.allocateMemory(
        vk::MemoryAllocateInfo{
            req.size, findMemoryType(req.memoryTypeBits,
                                     vk::MemoryPropertyFlagBits::eHostVisible)},
        allocator);

    device.bindBufferMemory(texData->uploadBuffer, texData->uploadBufferMemory,
                            0);
  }

  // Upload to Buffer:
  {
    void *map =
        device.mapMemory(texData->uploadBufferMemory, 0, image_size, {});
    memcpy(map, image_data, image_size);

    device.flushMappedMemoryRanges(
        vk::MappedMemoryRange{texData->uploadBufferMemory, 0, image_size});

    device.unmapMemory(texData->uploadBufferMemory);
  }

  // Create a command buffer
  std::vector<vk::CommandBuffer> command_buffers =
      device.allocateCommandBuffers(vk::CommandBufferAllocateInfo{
          command_pool, vk::CommandBufferLevel::ePrimary, 1});

  vk::CommandBuffer &command_buffer = command_buffers[0];
  command_buffer.begin(vk::CommandBufferBeginInfo{
      vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

  // Copy to Image
  {
    vk::ImageMemoryBarrier copy_barrier{
        {},
        vk::AccessFlagBits::eTransferWrite,
        vk::ImageLayout::eUndefined,
        vk::ImageLayout::eTransferDstOptimal,
        vk::QueueFamilyIgnored,
        vk::QueueFamilyIgnored,
        texData->image,
        vk::ImageSubresourceRange{
            vk::ImageAspectFlagBits::eColor, {}, 1, {}, 1}};

    command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eHost,
                                   vk::PipelineStageFlagBits::eTransfer, {},
                                   nullptr, {}, copy_barrier);

    vk::BufferImageCopy region{
        {},
        {},
        {},
        vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, {}, {}, 1},
        {},
        vk::Extent3D{static_cast<uint32_t>(texData->width),
                     static_cast<uint32_t>(texData->height), 1}};

    command_buffer.copyBufferToImage(texData->uploadBuffer, texData->image,
                                     vk::ImageLayout::eTransferDstOptimal,
                                     region);

    vk::ImageMemoryBarrier use_barrier{
        vk::AccessFlagBits::eTransferWrite,
        vk::AccessFlagBits::eShaderRead,
        vk::ImageLayout::eTransferDstOptimal,
        vk::ImageLayout::eReadOnlyOptimal,
        vk::QueueFamilyIgnored,
        vk::QueueFamilyIgnored,
        texData->image,
        vk::ImageSubresourceRange{
            vk::ImageAspectFlagBits::eColor, {}, 1, {}, 1}};

    command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                   vk::PipelineStageFlagBits::eFragmentShader,
                                   {}, {}, {}, use_barrier);
  }

  // End command buffer
  command_buffer.end();
  queue.submit(vk::SubmitInfo{{}, {}, {}, 1, command_buffers.data()});

  // Wait for sync
  device.waitIdle();
}

void VulkanTextureLoader::LoadTextureFromFile(const char *filename,
                                              MyTextureData *texData) {
  // Specifying 4 channels forces stb to load the image in RGBA which is an
  // easy format for Vulkan
  texData->channels = 4;
  unsigned char *image_data = stbi_load(filename, &texData->width,
                                        &texData->height, 0, texData->channels);

  if (image_data == nullptr) {
    return;
  }

  const vk::Format format = vk::Format::eR8G8B8A8Unorm;
  loadTexture(image_data, texData->width, texData->height, texData->channels,
              format, texData);
  stbi_image_free(image_data);
}

void VulkanTextureLoader::RemoveTexture(MyTextureData *tex_data) const {

  device.freeMemory(tex_data->uploadBufferMemory);
  device.destroyBuffer(tex_data->uploadBuffer);
  device.destroySampler(tex_data->sampler);
  device.destroyImageView(tex_data->imageView);
  device.destroyImage(tex_data->image);
  device.freeMemory(tex_data->imageMemory);

  // TODO(tnie) use custom DS
  ImGui_ImplVulkan_RemoveTexture(tex_data->ds);
}

auto VulkanTextureLoader::findMemoryType(
    uint32_t type_filter, vk::MemoryPropertyFlags properties) const
    -> uint32_t {
  const auto mem_properties = physical_device.getMemoryProperties();

  for (uint32_t i = 0; i < mem_properties.memoryTypeCount; i++) {
    if ((type_filter & (1 << i)) &&
        (mem_properties.memoryTypes[i].propertyFlags & properties) ==
            properties) {
      return i;
    }
  }

  return 0xFFFFFFFF; // Unable to find memoryType
}