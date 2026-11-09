#include "carflare.hpp"


#define GLFW_INCLUDE_VULKAN
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#include <GLFW/glfw3.h>
#include <png.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef SHADER_DIR
#define SHADER_DIR "shaders"
#endif

#ifndef ASSET_DIR
#define ASSET_DIR "assets"
#endif

constexpr uint32_t WIDTH = 1000;
constexpr uint32_t HEIGHT = 700;
constexpr int MAX_FRAMES = 2;
constexpr uint32_t MAX_HUD_GLYPHS = 12;

struct Vertex {
    glm::vec3 position;
    glm::vec3 color;
    glm::vec2 texCoord{0.0f, 0.0f};
};

// Keep one shared, non-overlapping block below Vulkan's guaranteed 128-byte
// push-constant limit.  Some drivers do not preserve distinct values for
// overlapping per-stage ranges, which corrupts transforms.
struct PushData {
    glm::mat4 mvp;
    glm::vec4 color;
    glm::vec4 params;
};

static_assert(sizeof(PushData) <= 128,
              "Push constants must fit Vulkan's guaranteed minimum");

const std::vector<Vertex> vertices = {
    // Front: red
    {{-1,-1, 1},{1,0.15f,0.15f}},
    {{ 1,-1, 1},{1,0.15f,0.15f}},
    {{ 1, 1, 1},{1,0.15f,0.15f}},
    {{-1,-1, 1},{1,0.15f,0.15f}},
    {{ 1, 1, 1},{1,0.15f,0.15f}},
    {{-1, 1, 1},{1,0.15f,0.15f}},

    // Back: blue
    {{ 1,-1,-1},{0.2f,0.35f,1}},
    {{-1,-1,-1},{0.2f,0.35f,1}},
    {{-1, 1,-1},{0.2f,0.35f,1}},
    {{ 1,-1,-1},{0.2f,0.35f,1}},
    {{-1, 1,-1},{0.2f,0.35f,1}},
    {{ 1, 1,-1},{0.2f,0.35f,1}},

    // Right: green
    {{ 1,-1, 1},{0.2f,1,0.3f}},
    {{ 1,-1,-1},{0.2f,1,0.3f}},
    {{ 1, 1,-1},{0.2f,1,0.3f}},
    {{ 1,-1, 1},{0.2f,1,0.3f}},
    {{ 1, 1,-1},{0.2f,1,0.3f}},
    {{ 1, 1, 1},{0.2f,1,0.3f}},

    // Left: yellow
    {{-1,-1,-1},{1,0.85f,0.15f}},
    {{-1,-1, 1},{1,0.85f,0.15f}},
    {{-1, 1, 1},{1,0.85f,0.15f}},
    {{-1,-1,-1},{1,0.85f,0.15f}},
    {{-1, 1, 1},{1,0.85f,0.15f}},
    {{-1, 1,-1},{1,0.85f,0.15f}},

    // Top: magenta
    {{-1, 1, 1},{1,0.2f,0.8f}},
    {{ 1, 1, 1},{1,0.2f,0.8f}},
    {{ 1, 1,-1},{1,0.2f,0.8f}},
    {{-1, 1, 1},{1,0.2f,0.8f}},
    {{ 1, 1,-1},{1,0.2f,0.8f}},
    {{-1, 1,-1},{1,0.2f,0.8f}},

    // Bottom: cyan
    {{-1,-1,-1},{0.15f,0.9f,1}},
    {{ 1,-1,-1},{0.15f,0.9f,1}},
    {{ 1,-1, 1},{0.15f,0.9f,1}},
    {{-1,-1,-1},{0.15f,0.9f,1}},
    {{ 1,-1, 1},{0.15f,0.9f,1}},
    {{-1,-1, 1},{0.15f,0.9f,1}}
};

struct QueueFamilies {
    std::optional<uint32_t> graphics;
    std::optional<uint32_t> present;

    bool complete() const {
        return graphics.has_value() && present.has_value();
    }
};

struct SwapSupport {
    VkSurfaceCapabilitiesKHR capabilities{};
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> modes;
};

class CarFlareImpl {
    GLFWwindow* window = nullptr;

    VkInstance instance = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkPhysicalDevice gpu = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;

    VkQueue graphicsQueue = VK_NULL_HANDLE;
    VkQueue presentQueue = VK_NULL_HANDLE;

    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    std::vector<VkImage> swapImages;
    std::vector<VkImageView> imageViews;
    VkFormat swapFormat{};
    VkExtent2D swapExtent{};

    VkRenderPass renderPass = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    std::vector<VkFramebuffer> framebuffers;

    VkCommandPool commandPool = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> commandBuffers;

    VkBuffer vertexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory vertexMemory = VK_NULL_HANDLE;
    VkBuffer carModelBuffer = VK_NULL_HANDLE;
    VkDeviceMemory carModelMemory = VK_NULL_HANDLE;
    uint32_t carModelVertexCount = 0;
    bool carModelLoaded = false;

    VkImage carTextureImage = VK_NULL_HANDLE;
    VkDeviceMemory carTextureMemory = VK_NULL_HANDLE;
    VkImageView carTextureView = VK_NULL_HANDLE;
    VkSampler carTextureSampler = VK_NULL_HANDLE;
    VkDescriptorSetLayout textureSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool textureDescriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet textureDescriptorSet = VK_NULL_HANDLE;

    VkImage hudFontImage = VK_NULL_HANDLE;
    VkDeviceMemory hudFontMemory = VK_NULL_HANDLE;
    VkImageView hudFontView = VK_NULL_HANDLE;
    VkSampler hudFontSampler = VK_NULL_HANDLE;
    VkDescriptorSet hudFontDescriptorSet = VK_NULL_HANDLE;
    VkBuffer hudVertexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory hudVertexMemory = VK_NULL_HANDLE;
    VkPipeline hudPipeline = VK_NULL_HANDLE;
    uint32_t hudVertexCount = 0;

    std::vector<VkImage> depthImages;
    std::vector<VkDeviceMemory> depthMemories;
    std::vector<VkImageView> depthViews;
    VkFormat depthFormat{};

    VkSampleCountFlagBits msaaSamples = VK_SAMPLE_COUNT_1_BIT;
    std::vector<VkImage> msaaColorImages;
    std::vector<VkDeviceMemory> msaaColorMemories;
    std::vector<VkImageView> msaaColorViews;

    std::array<VkSemaphore, MAX_FRAMES> imageAvailable{};
    std::array<VkSemaphore, MAX_FRAMES> renderFinished{};
    std::array<VkFence, MAX_FRAMES> inFlight{};
    std::vector<VkFence> imagesInFlight;

    size_t currentFrame = 0;
    bool framebufferResized = false;
    bool gameStarted = false;
    uint32_t raceScore = 0;
    float scoredDistance = 0.0f;

    static void mouseButtonCallback(GLFWwindow* w, int button, int action, int) {
        if (button != GLFW_MOUSE_BUTTON_LEFT || action != GLFW_PRESS)
            return;

        auto* app = static_cast<CarFlareImpl*>(glfwGetWindowUserPointer(w));
        if (!app || app->gameStarted)
            return;

        double x = 0.0, y = 0.0;
        int width = 0, height = 0;
        glfwGetCursorPos(w, &x, &y);
        glfwGetWindowSize(w, &width, &height);

        // The green PLAY panel occupies the central lower portion of the menu.
        if (width > 0 && height > 0 &&
            x >= width * 0.35 && x <= width * 0.65 &&
            y >= height * 0.53 && y <= height * 0.70) {
            app->gameStarted = true;
            app->raceScore = 0;
            app->scoredDistance = 0.0f;
            glfwSetWindowTitle(w, "CarFlare - Score: 0 | W A S D");
        }
    }

    static void framebufferCallback(GLFWwindow* w, int, int) {
        auto* app = static_cast<CarFlareImpl*>(
            glfwGetWindowUserPointer(w));
        app->framebufferResized = true;
    }

    void initWindow() {
        if (!glfwInit())
            throw std::runtime_error("GLFW initialization failed");

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

        window = glfwCreateWindow(
            WIDTH, HEIGHT, "CarFlare - Endless Drive | W A S D",
            nullptr, nullptr);

        if (!window)
            throw std::runtime_error("Window creation failed");

        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebufferCallback);
        glfwSetMouseButtonCallback(window, mouseButtonCallback);
        glfwSetWindowTitle(window, "CarFlare | Click the green PLAY button");
    }

    void initVulkan() {
        createInstance();

        if (glfwCreateWindowSurface(
                instance, window, nullptr, &surface) != VK_SUCCESS)
            throw std::runtime_error("Surface creation failed");

        pickGPU();
        createDevice();
        createCommandPool();
        createVertexBuffer();
        createHUDVertexBuffer();
        loadCarOBJ(std::string(ASSET_DIR) + "/car.obj");
        createCarTexture(std::string(ASSET_DIR) + "/colormap.png");
        createFontTexture(std::string(ASSET_DIR) + "/fonts/anton_digits.png");
        createTextureDescriptor();
        createSwapchainResources();
        createSync();
    }

    void createInstance() {
        VkApplicationInfo app{};
        app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app.pApplicationName = "Car Flare";
        app.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        app.pEngineName = "Car Flare";
        app.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        app.apiVersion = VK_API_VERSION_1_0;

        uint32_t count = 0;
        const char** extensions =
            glfwGetRequiredInstanceExtensions(&count);

        VkInstanceCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        info.pApplicationInfo = &app;
        info.enabledExtensionCount = count;
        info.ppEnabledExtensionNames = extensions;

        if (vkCreateInstance(&info, nullptr, &instance) != VK_SUCCESS)
            throw std::runtime_error("Instance creation failed");
    }

    QueueFamilies findQueues(VkPhysicalDevice candidate) {
        QueueFamilies result;
        uint32_t count = 0;

        vkGetPhysicalDeviceQueueFamilyProperties(
            candidate, &count, nullptr);

        std::vector<VkQueueFamilyProperties> families(count);
        vkGetPhysicalDeviceQueueFamilyProperties(
            candidate, &count, families.data());

        for (uint32_t i = 0; i < count; ++i) {
            if (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
                result.graphics = i;

            VkBool32 supported = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(
                candidate, i, surface, &supported);

            if (supported)
                result.present = i;

            if (result.complete())
                break;
        }

        return result;
    }

    bool hasSwapchainExtension(VkPhysicalDevice candidate) {
        uint32_t count = 0;
        vkEnumerateDeviceExtensionProperties(
            candidate, nullptr, &count, nullptr);

        std::vector<VkExtensionProperties> extensions(count);
        vkEnumerateDeviceExtensionProperties(
            candidate, nullptr, &count, extensions.data());

        for (const auto& ext : extensions) {
            if (std::string(ext.extensionName) ==
                VK_KHR_SWAPCHAIN_EXTENSION_NAME)
                return true;
        }

        return false;
    }

    SwapSupport querySwap(VkPhysicalDevice candidate) {
        SwapSupport result;

        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
            candidate, surface, &result.capabilities);

        uint32_t count = 0;
        vkGetPhysicalDeviceSurfaceFormatsKHR(
            candidate, surface, &count, nullptr);
        result.formats.resize(count);
        vkGetPhysicalDeviceSurfaceFormatsKHR(
            candidate, surface, &count, result.formats.data());

        count = 0;
        vkGetPhysicalDeviceSurfacePresentModesKHR(
            candidate, surface, &count, nullptr);
        result.modes.resize(count);
        vkGetPhysicalDeviceSurfacePresentModesKHR(
            candidate, surface, &count, result.modes.data());

        return result;
    }

    bool suitable(VkPhysicalDevice candidate) {
        if (!findQueues(candidate).complete() ||
            !hasSwapchainExtension(candidate))
            return false;

        auto support = querySwap(candidate);
        return !support.formats.empty() && !support.modes.empty();
    }

    void pickGPU() {
        uint32_t count = 0;
        vkEnumeratePhysicalDevices(instance, &count, nullptr);

        if (count == 0)
            throw std::runtime_error("No Vulkan GPU found");

        std::vector<VkPhysicalDevice> devices(count);
        vkEnumeratePhysicalDevices(instance, &count, devices.data());

        for (auto candidate : devices) {
            if (suitable(candidate)) {
                gpu = candidate;
                break;
            }
        }

        if (gpu == VK_NULL_HANDLE)
            throw std::runtime_error(
                "No suitable graphics/presentation GPU found");

        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(gpu, &properties);
        const VkSampleCountFlags supported =
            properties.limits.framebufferColorSampleCounts &
            properties.limits.framebufferDepthSampleCounts;
        if (supported & VK_SAMPLE_COUNT_8_BIT) msaaSamples = VK_SAMPLE_COUNT_8_BIT;
        else if (supported & VK_SAMPLE_COUNT_4_BIT) msaaSamples = VK_SAMPLE_COUNT_4_BIT;
        else if (supported & VK_SAMPLE_COUNT_2_BIT) msaaSamples = VK_SAMPLE_COUNT_2_BIT;
        std::cout << "GPU: " << properties.deviceName << '\n';
    }

    void createDevice() {
        auto queues = findQueues(gpu);

        std::vector<uint32_t> unique = {
            *queues.graphics, *queues.present
        };
        std::sort(unique.begin(), unique.end());
        unique.erase(std::unique(unique.begin(), unique.end()),
                     unique.end());

        float priority = 1.0f;
        std::vector<VkDeviceQueueCreateInfo> queueInfos;

        for (uint32_t family : unique) {
            VkDeviceQueueCreateInfo q{};
            q.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            q.queueFamilyIndex = family;
            q.queueCount = 1;
            q.pQueuePriorities = &priority;
            queueInfos.push_back(q);
        }

        VkPhysicalDeviceFeatures features{};

        VkDeviceCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        info.queueCreateInfoCount =
            static_cast<uint32_t>(queueInfos.size());
        info.pQueueCreateInfos = queueInfos.data();
        const char* extensions[] = {
            VK_KHR_SWAPCHAIN_EXTENSION_NAME
        };
        info.enabledExtensionCount = 1;
        info.ppEnabledExtensionNames = extensions;
        info.pEnabledFeatures = &features;

        if (vkCreateDevice(gpu, &info, nullptr, &device) != VK_SUCCESS)
            throw std::runtime_error("Device creation failed");

        vkGetDeviceQueue(device, *queues.graphics, 0, &graphicsQueue);
        vkGetDeviceQueue(device, *queues.present, 0, &presentQueue);
    }

    uint32_t findMemoryType(uint32_t filter,
                            VkMemoryPropertyFlags properties) {
        VkPhysicalDeviceMemoryProperties memory{};
        vkGetPhysicalDeviceMemoryProperties(gpu, &memory);

        for (uint32_t i = 0; i < memory.memoryTypeCount; ++i) {
            if ((filter & (1u << i)) &&
                (memory.memoryTypes[i].propertyFlags & properties)
                    == properties)
                return i;
        }

        throw std::runtime_error("Suitable GPU memory type not found");
    }

    void createBuffer(VkDeviceSize size,
                      VkBufferUsageFlags usage,
                      VkMemoryPropertyFlags properties,
                      VkBuffer& buffer,
                      VkDeviceMemory& memory) {
        VkBufferCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        info.size = size;
        info.usage = usage;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(device, &info, nullptr, &buffer) != VK_SUCCESS)
            throw std::runtime_error("Buffer creation failed");

        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(device, buffer, &requirements);

        VkMemoryAllocateInfo allocation{};
        allocation.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = findMemoryType(
            requirements.memoryTypeBits, properties);

        if (vkAllocateMemory(
                device, &allocation, nullptr, &memory) != VK_SUCCESS)
            throw std::runtime_error("Buffer memory allocation failed");

        vkBindBufferMemory(device, buffer, memory, 0);
    }

    void createVertexBuffer() {
        VkDeviceSize size = sizeof(vertices[0]) * vertices.size();

        createBuffer(
            size,
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            vertexBuffer, vertexMemory);

        void* data = nullptr;
        vkMapMemory(device, vertexMemory, 0, size, 0, &data);
        std::memcpy(data, vertices.data(), static_cast<size_t>(size));
        vkUnmapMemory(device, vertexMemory);
    }

    void createHUDVertexBuffer() {
        createBuffer(sizeof(Vertex) * MAX_HUD_GLYPHS * 6,
                     VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     hudVertexBuffer, hudVertexMemory);
    }

    void updateHUDText(const std::string& score, float top, float glyphWidth,
                       float glyphHeight) {
        const uint32_t glyphCount = std::min<uint32_t>(
            static_cast<uint32_t>(score.size()), MAX_HUD_GLYPHS);
        std::vector<Vertex> glyphs;
        glyphs.reserve(glyphCount * 6);
        // Clip-space HUD: keep the complete label centred at the top edge.
        const float left = -0.5f * glyphWidth * glyphCount;
        for (uint32_t i = 0; i < glyphCount; ++i) {
            const float x0 = left + glyphWidth * i;
            const float x1 = x0 + glyphWidth;
            const float y0 = top + glyphHeight;
            const float y1 = top;
            const std::string atlasCharacters = "CARFLEPY0123456789";
            const size_t glyphIndex = atlasCharacters.find(score[i]);
            if (glyphIndex == std::string::npos) continue;
            const float u0 = static_cast<float>(glyphIndex) / 18.0f;
            const float u1 = static_cast<float>(glyphIndex + 1) / 18.0f;
            const glm::vec3 white(1.0f);
            glyphs.insert(glyphs.end(), {
                {{x0,y0,0},white,{u0,1}}, {{x1,y0,0},white,{u1,1}}, {{x1,y1,0},white,{u1,0}},
                {{x0,y0,0},white,{u0,1}}, {{x1,y1,0},white,{u1,0}}, {{x0,y1,0},white,{u0,0}}
            });
        }
        void* mapped = nullptr;
        const VkDeviceSize bytes = sizeof(Vertex) * glyphs.size();
        if (vkMapMemory(device, hudVertexMemory, 0, bytes, 0, &mapped) != VK_SUCCESS)
            throw std::runtime_error("Could not map HUD vertex buffer");
        std::memcpy(mapped, glyphs.data(), static_cast<size_t>(bytes));
        vkUnmapMemory(device, hudVertexMemory);
        hudVertexCount = static_cast<uint32_t>(glyphs.size());
    }

    void drawScoreHUD(VkCommandBuffer command) {
        // Compact numeric-only racing HUD.
        updateHUDText(std::to_string(raceScore), -0.90f, 0.052f, 0.092f);
        if (hudVertexCount == 0) return;
        VkDeviceSize offset = 0;
        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, hudPipeline);
        vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_GRAPHICS,
            pipelineLayout, 0, 1, &hudFontDescriptorSet, 0, nullptr);
        vkCmdBindVertexBuffers(command, 0, 1, &hudVertexBuffer, &offset);
        vkCmdDraw(command, hudVertexCount, 1, 0, 0);
    }

    void drawMenuTitleHUD(VkCommandBuffer command) {
        updateHUDText("CARFLARE", -0.62f, 0.095f, 0.15f);
        if (hudVertexCount == 0) return;
        VkDeviceSize offset = 0;
        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, hudPipeline);
        vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_GRAPHICS,
            pipelineLayout, 0, 1, &hudFontDescriptorSet, 0, nullptr);
        vkCmdBindVertexBuffers(command, 0, 1, &hudVertexBuffer, &offset);
        vkCmdDraw(command, hudVertexCount, 1, 0, 0);
    }

    VkCommandBuffer beginOneTimeCommands() {
        VkCommandBufferAllocateInfo alloc{};
        alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc.commandPool = commandPool;
        alloc.commandBufferCount = 1;
        VkCommandBuffer command{};
        if (vkAllocateCommandBuffers(device, &alloc, &command) != VK_SUCCESS)
            throw std::runtime_error("One-time command buffer allocation failed");
        VkCommandBufferBeginInfo begin{};
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer(command, &begin) != VK_SUCCESS)
            throw std::runtime_error("One-time command buffer begin failed");
        return command;
    }

    void endOneTimeCommands(VkCommandBuffer command) {
        if (vkEndCommandBuffer(command) != VK_SUCCESS)
            throw std::runtime_error("One-time command buffer end failed");
        VkSubmitInfo submit{};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command;
        if (vkQueueSubmit(graphicsQueue, 1, &submit, VK_NULL_HANDLE) != VK_SUCCESS ||
            vkQueueWaitIdle(graphicsQueue) != VK_SUCCESS)
            throw std::runtime_error("Texture upload submission failed");
        vkFreeCommandBuffers(device, commandPool, 1, &command);
    }

    void transitionTexture(VkImage image, VkImageLayout oldLayout,
                          VkImageLayout newLayout) {
        VkCommandBuffer command = beginOneTimeCommands();
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = oldLayout;
        barrier.newLayout = newLayout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;
        VkPipelineStageFlags sourceStage, destinationStage;
        if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
            newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
                   newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        } else {
            vkFreeCommandBuffers(device, commandPool, 1, &command);
            throw std::runtime_error("Unsupported texture image transition");
        }
        vkCmdPipelineBarrier(command, sourceStage, destinationStage, 0,
                             0, nullptr, 0, nullptr, 1, &barrier);
        endOneTimeCommands(command);
    }

    void createTexture(const std::string& path, VkImage& image, VkDeviceMemory& memory,
                       VkImageView& view, VkSampler& outSampler) {
        png_image png{};
        png.version = PNG_IMAGE_VERSION;
        if (!png_image_begin_read_from_file(&png, path.c_str()))
            throw std::runtime_error("Could not open texture " + path + ": " + png.message);
        png.format = PNG_FORMAT_RGBA;
        std::vector<png_byte> pixels(PNG_IMAGE_SIZE(png));
        if (!png_image_finish_read(&png, nullptr, pixels.data(), 0, nullptr)) {
            std::string message = png.message;
            png_image_free(&png);
            throw std::runtime_error("Could not decode texture " + path + ": " + message);
        }
        const uint32_t width = png.width;
        const uint32_t height = png.height;
        png_image_free(&png);
        if (!width || !height)
            throw std::runtime_error("Texture has invalid dimensions: " + path);

        const VkDeviceSize imageSize = static_cast<VkDeviceSize>(pixels.size());
        VkBuffer stagingBuffer = VK_NULL_HANDLE;
        VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
        createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     stagingBuffer, stagingMemory);
        void* mapped = nullptr;
        if (vkMapMemory(device, stagingMemory, 0, imageSize, 0, &mapped) != VK_SUCCESS)
            throw std::runtime_error("Could not map texture staging buffer");
        std::memcpy(mapped, pixels.data(), pixels.size());
        vkUnmapMemory(device, stagingMemory);

        VkImageCreateInfo imageInfo{};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.extent = {width, height, 1};
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.format = VK_FORMAT_R8G8B8A8_SRGB;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS)
            throw std::runtime_error("Texture image creation failed");
        VkMemoryRequirements requirements{};
        vkGetImageMemoryRequirements(device, image, &requirements);
        VkMemoryAllocateInfo allocation{};
        allocation.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = findMemoryType(requirements.memoryTypeBits,
                                                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (vkAllocateMemory(device, &allocation, nullptr, &memory) != VK_SUCCESS)
            throw std::runtime_error("Texture image memory allocation failed");
        vkBindImageMemory(device, image, memory, 0);
        transitionTexture(image, VK_IMAGE_LAYOUT_UNDEFINED,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkCommandBuffer command = beginOneTimeCommands();
        VkBufferImageCopy region{};
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;
        region.imageExtent = {width, height, 1};
        vkCmdCopyBufferToImage(command, stagingBuffer, image,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        endOneTimeCommands(command);
        transitionTexture(image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingMemory, nullptr);
        view = createImageView(image, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT);

        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.anisotropyEnable = VK_FALSE;
        samplerInfo.maxAnisotropy = 1.0f;
        samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        samplerInfo.unnormalizedCoordinates = VK_FALSE;
        samplerInfo.compareEnable = VK_FALSE;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        if (vkCreateSampler(device, &samplerInfo, nullptr, &outSampler) != VK_SUCCESS)
            throw std::runtime_error("Texture sampler creation failed");
        std::cout << "Loaded texture: " << path << " (" << width << "x" << height << ")\n";
    }

    void createCarTexture(const std::string& path) {
        createTexture(path, carTextureImage, carTextureMemory, carTextureView, carTextureSampler);
    }
    void createFontTexture(const std::string& path) {
        createTexture(path, hudFontImage, hudFontMemory, hudFontView, hudFontSampler);
    }

    void createTextureDescriptor() {
        VkDescriptorSetLayoutBinding binding{};
        binding.binding = 0;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        binding.descriptorCount = 1;
        binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        VkDescriptorSetLayoutCreateInfo layout{};
        layout.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layout.bindingCount = 1;
        layout.pBindings = &binding;
        if (vkCreateDescriptorSetLayout(device, &layout, nullptr, &textureSetLayout) != VK_SUCCESS)
            throw std::runtime_error("Texture descriptor-set layout creation failed");

        VkDescriptorPoolSize poolSize{};
        poolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSize.descriptorCount = 2;
        VkDescriptorPoolCreateInfo pool{};
        pool.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool.maxSets = 2;
        pool.poolSizeCount = 1;
        pool.pPoolSizes = &poolSize;
        if (vkCreateDescriptorPool(device, &pool, nullptr, &textureDescriptorPool) != VK_SUCCESS)
            throw std::runtime_error("Texture descriptor pool creation failed");
        VkDescriptorSetAllocateInfo alloc{};
        alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc.descriptorPool = textureDescriptorPool;
        alloc.descriptorSetCount = 1;
        alloc.pSetLayouts = &textureSetLayout;
        if (vkAllocateDescriptorSets(device, &alloc, &textureDescriptorSet) != VK_SUCCESS)
            throw std::runtime_error("Texture descriptor-set allocation failed");
        if (vkAllocateDescriptorSets(device, &alloc, &hudFontDescriptorSet) != VK_SUCCESS)
            throw std::runtime_error("HUD font descriptor-set allocation failed");
        VkDescriptorImageInfo image{};
        image.sampler = carTextureSampler;
        image.imageView = carTextureView;
        image.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = textureDescriptorSet;
        write.dstBinding = 0;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write.pImageInfo = &image;
        vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);

        image.sampler = hudFontSampler;
        image.imageView = hudFontView;
        write.dstSet = hudFontDescriptorSet;
        vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
    }

    // OBJ loader: keeps each face corner's UV coordinate when triangulating.
    void loadCarOBJ(const std::string& path) {
        std::ifstream file(path);
        if (!file) {
            std::cerr << "No " << path << " found; using the built-in car.\n";
            return;
        }
        std::vector<glm::vec3> positions;
        std::vector<glm::vec2> texCoords;
        std::vector<Vertex> mesh;
        std::string line;
        while (std::getline(file, line)) {
            std::istringstream row(line);
            std::string kind;
            row >> kind;
            if (kind == "v") {
                glm::vec3 v{};
                if (row >> v.x >> v.y >> v.z) positions.push_back(v);
            } else if (kind == "vt") {
                glm::vec2 uv{};
                if (row >> uv.x >> uv.y) texCoords.push_back(uv);
            } else if (kind == "f") {
                struct Corner { int position = -1; int uv = -1; };
                std::vector<Corner> face;
                std::string token;
                while (row >> token) {
                    std::istringstream parts(token);
                    std::string p, uv;
                    std::getline(parts, p, '/');
                    std::getline(parts, uv, '/');
                    if (p.empty()) continue;
                    int pi = std::stoi(p);
                    if (pi < 0) pi = static_cast<int>(positions.size()) + pi;
                    else --pi;
                    int ti = -1;
                    if (!uv.empty()) {
                        ti = std::stoi(uv);
                        if (ti < 0) ti = static_cast<int>(texCoords.size()) + ti;
                        else --ti;
                    }
                    if (pi >= 0 && static_cast<size_t>(pi) < positions.size())
                        face.push_back({pi, ti});
                }
                for (size_t i = 1; i + 1 < face.size(); ++i) {
                    for (const Corner& c : {face[0], face[i], face[i + 1]}) {
                        glm::vec2 uv(0.0f);
                        if (c.uv >= 0 && static_cast<size_t>(c.uv) < texCoords.size())
                            uv = texCoords[static_cast<size_t>(c.uv)];
                        // OBJ's V axis is opposite the usual Vulkan texture convention.
                        uv.y = 1.0f - uv.y;
                        mesh.push_back({positions[static_cast<size_t>(c.position)],
                                        {1.0f, 1.0f, 1.0f}, uv});
                    }
                }
            }
        }
        if (mesh.empty()) {
            std::cerr << "OBJ contains no usable triangles: " << path << "; using built-in car.\n";
            return;
        }
        glm::vec3 minimum(std::numeric_limits<float>::max());
        glm::vec3 maximum(std::numeric_limits<float>::lowest());
        for (const auto& v : mesh) {
            minimum = glm::min(minimum, v.position);
            maximum = glm::max(maximum, v.position);
        }
        const glm::vec3 dimensions = maximum - minimum;
        const glm::vec3 center = (maximum + minimum) * 0.5f;
        const float largest = std::max({dimensions.x, dimensions.y, dimensions.z});
        if (largest <= 0.0001f) {
            std::cerr << "OBJ has invalid dimensions; using built-in car.\n";
            return;
        }
        const float scale = 3.4f / std::max(dimensions.x,
                                  std::max(dimensions.z, dimensions.y * 1.8f));
        for (auto& v : mesh) {
            v.position = (v.position - center) * scale;
            v.position.y += 0.55f;
        }
        const VkDeviceSize size = sizeof(Vertex) * mesh.size();
        createBuffer(size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     carModelBuffer, carModelMemory);
        void* data = nullptr;
        if (vkMapMemory(device, carModelMemory, 0, size, 0, &data) != VK_SUCCESS)
            throw std::runtime_error("Could not map car model buffer");
        std::memcpy(data, mesh.data(), static_cast<size_t>(size));
        vkUnmapMemory(device, carModelMemory);
        carModelVertexCount = static_cast<uint32_t>(mesh.size());
        carModelLoaded = true;
        std::cout << "Loaded OBJ car model: " << path << " (" << carModelVertexCount / 3
                  << " triangles, " << texCoords.size() << " UVs)\n";
    }

    void drawCarModel(VkCommandBuffer command, const glm::vec3& position,
                      const glm::vec3& scale, const glm::vec4& color,
                      float yaw = 0.0f, float pitch = 0.0f, float roll = 0.0f) {
        if (!carModelLoaded) return;
        glm::mat4 model(1.0f);
        model = glm::translate(model, position);
        model = glm::rotate(model, yaw, glm::vec3(0.0f, 1.0f, 0.0f));
        model = glm::rotate(model, pitch, glm::vec3(1.0f, 0.0f, 0.0f));
        model = glm::rotate(model, roll, glm::vec3(0.0f, 0.0f, 1.0f));
        model = glm::scale(model, scale);
        const glm::mat4 mvp = makeMVP(model);
        const PushData pushData{
            mvp, color, glm::vec4(1.0f, 0.0f, 0.0f, 0.0f)};
        vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_GRAPHICS,
            pipelineLayout, 0, 1, &textureDescriptorSet, 0, nullptr);
        vkCmdPushConstants(command, pipelineLayout,
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            0, sizeof(pushData), &pushData);
        VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(command, 0, 1, &carModelBuffer, &offset);
        vkCmdDraw(command, carModelVertexCount, 1, 0, 0);
        vkCmdBindVertexBuffers(command, 0, 1, &vertexBuffer, &offset);
    }

    VkSurfaceFormatKHR chooseFormat(
        const std::vector<VkSurfaceFormatKHR>& formats) {
        for (const auto& f : formats) {
            if (f.format == VK_FORMAT_B8G8R8A8_SRGB &&
                f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
                return f;
        }
        return formats.front();
    }

    VkPresentModeKHR choosePresent(
        const std::vector<VkPresentModeKHR>& modes) {
        for (auto mode : modes)
            if (mode == VK_PRESENT_MODE_MAILBOX_KHR)
                return mode;

        return VK_PRESENT_MODE_FIFO_KHR;
    }

    VkExtent2D chooseExtent(
        const VkSurfaceCapabilitiesKHR& capabilities) {
        if (capabilities.currentExtent.width !=
            std::numeric_limits<uint32_t>::max())
            return capabilities.currentExtent;

        int w, h;
        glfwGetFramebufferSize(window, &w, &h);

        VkExtent2D extent{
            static_cast<uint32_t>(std::max(w, 0)),
            static_cast<uint32_t>(std::max(h, 0))
        };

        extent.width = std::clamp(
            extent.width,
            capabilities.minImageExtent.width,
            capabilities.maxImageExtent.width);
        extent.height = std::clamp(
            extent.height,
            capabilities.minImageExtent.height,
            capabilities.maxImageExtent.height);

        return extent;
    }

    void createSwapchain() {
        auto support = querySwap(gpu);
        auto format = chooseFormat(support.formats);
        auto mode = choosePresent(support.modes);
        auto extent = chooseExtent(support.capabilities);

        uint32_t count = support.capabilities.minImageCount + 1;
        if (support.capabilities.maxImageCount > 0)
            count = std::min(count, support.capabilities.maxImageCount);

        auto queues = findQueues(gpu);
        uint32_t indices[] = {
            *queues.graphics, *queues.present
        };

        VkSwapchainCreateInfoKHR info{};
        info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        info.surface = surface;
        info.minImageCount = count;
        info.imageFormat = format.format;
        info.imageColorSpace = format.colorSpace;
        info.imageExtent = extent;
        info.imageArrayLayers = 1;
        info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

        if (queues.graphics != queues.present) {
            info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            info.queueFamilyIndexCount = 2;
            info.pQueueFamilyIndices = indices;
        } else {
            info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }

        info.preTransform = support.capabilities.currentTransform;
        info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        info.presentMode = mode;
        info.clipped = VK_TRUE;

        if (vkCreateSwapchainKHR(
                device, &info, nullptr, &swapchain) != VK_SUCCESS)
            throw std::runtime_error("Swapchain creation failed");

        vkGetSwapchainImagesKHR(
            device, swapchain, &count, nullptr);
        swapImages.resize(count);
        vkGetSwapchainImagesKHR(
            device, swapchain, &count, swapImages.data());

        swapFormat = format.format;
        swapExtent = extent;
    }

    VkImageView createImageView(VkImage image, VkFormat format,
                                VkImageAspectFlags aspect) {
        VkImageViewCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        info.image = image;
        info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        info.format = format;
        info.subresourceRange.aspectMask = aspect;
        info.subresourceRange.baseMipLevel = 0;
        info.subresourceRange.levelCount = 1;
        info.subresourceRange.baseArrayLayer = 0;
        info.subresourceRange.layerCount = 1;

        VkImageView view;
        if (vkCreateImageView(
                device, &info, nullptr, &view) != VK_SUCCESS)
            throw std::runtime_error("Image view creation failed");

        return view;
    }

    void createImageViews() {
        imageViews.clear();
        for (auto image : swapImages)
            imageViews.push_back(
                createImageView(image, swapFormat,
                                VK_IMAGE_ASPECT_COLOR_BIT));
    }

    VkFormat findDepthFormat() {
        const VkFormat candidates[] = {
            VK_FORMAT_D32_SFLOAT,
            VK_FORMAT_D32_SFLOAT_S8_UINT,
            VK_FORMAT_D24_UNORM_S8_UINT
        };

        for (auto format : candidates) {
            VkFormatProperties properties{};
            vkGetPhysicalDeviceFormatProperties(
                gpu, format, &properties);

            if (properties.optimalTilingFeatures &
                VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)
                return format;
        }

        throw std::runtime_error("No supported depth format");
    }

    void createDepthResources() {
        depthFormat = findDepthFormat();

        depthImages.resize(swapImages.size());
        depthMemories.resize(swapImages.size());
        depthViews.resize(swapImages.size());

        for (size_t i = 0; i < swapImages.size(); ++i) {
            VkImageCreateInfo info{};
            info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            info.imageType = VK_IMAGE_TYPE_2D;
            info.extent = {swapExtent.width, swapExtent.height, 1};
            info.mipLevels = 1;
            info.arrayLayers = 1;
            info.format = depthFormat;
            info.tiling = VK_IMAGE_TILING_OPTIMAL;
            info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            info.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
            info.samples = msaaSamples;
            info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

            if (vkCreateImage(
                    device, &info, nullptr, &depthImages[i]) != VK_SUCCESS)
                throw std::runtime_error("Depth image creation failed");

            VkMemoryRequirements requirements{};
            vkGetImageMemoryRequirements(
                device, depthImages[i], &requirements);

            VkMemoryAllocateInfo allocation{};
            allocation.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            allocation.allocationSize = requirements.size;
            allocation.memoryTypeIndex = findMemoryType(
                requirements.memoryTypeBits,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

            if (vkAllocateMemory(
                    device, &allocation, nullptr,
                    &depthMemories[i]) != VK_SUCCESS)
                throw std::runtime_error("Depth memory allocation failed");

            vkBindImageMemory(
                device, depthImages[i], depthMemories[i], 0);

            depthViews[i] = createImageView(
                depthImages[i], depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT);
        }
    }

    void createMSAAColorResources() {
        msaaColorImages.resize(swapImages.size());
        msaaColorMemories.resize(swapImages.size());
        msaaColorViews.resize(swapImages.size());

        for (size_t i = 0; i < swapImages.size(); ++i) {
            VkImageCreateInfo info{};
            info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            info.imageType = VK_IMAGE_TYPE_2D;
            info.extent = {swapExtent.width, swapExtent.height, 1};
            info.mipLevels = 1;
            info.arrayLayers = 1;
            info.format = swapFormat;
            info.tiling = VK_IMAGE_TILING_OPTIMAL;
            info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            info.usage = VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT |
                         VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
            info.samples = msaaSamples;
            info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

            if (vkCreateImage(device, &info, nullptr, &msaaColorImages[i]) != VK_SUCCESS)
                throw std::runtime_error("MSAA color image creation failed");

            VkMemoryRequirements requirements{};
            vkGetImageMemoryRequirements(device, msaaColorImages[i], &requirements);
            VkMemoryAllocateInfo allocation{};
            allocation.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            allocation.allocationSize = requirements.size;
            allocation.memoryTypeIndex = findMemoryType(
                requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            if (vkAllocateMemory(device, &allocation, nullptr, &msaaColorMemories[i]) != VK_SUCCESS)
                throw std::runtime_error("MSAA color memory allocation failed");
            vkBindImageMemory(device, msaaColorImages[i], msaaColorMemories[i], 0);
            msaaColorViews[i] = createImageView(
                msaaColorImages[i], swapFormat, VK_IMAGE_ASPECT_COLOR_BIT);
        }
    }

    void createRenderPass() {
        VkAttachmentDescription color{};
        color.format = swapFormat;
        color.samples = msaaSamples;
        color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        color.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        color.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        color.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        color.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        color.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkAttachmentDescription depth{};
        depth.format = depthFormat;
        depth.samples = msaaSamples;
        depth.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depth.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depth.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depth.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depth.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depth.finalLayout =
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentDescription resolve{};
        resolve.format = swapFormat;
        resolve.samples = VK_SAMPLE_COUNT_1_BIT;
        resolve.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        resolve.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        resolve.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        resolve.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        resolve.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        resolve.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentDescription attachments[] = {color, depth, resolve};

        VkAttachmentReference colorRef{};
        colorRef.attachment = 0;
        colorRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkAttachmentReference depthRef{};
        depthRef.attachment = 1;
        depthRef.layout =
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentReference resolveRef{};
        resolveRef.attachment = 2;
        resolveRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorRef;
        subpass.pDepthStencilAttachment = &depthRef;
        subpass.pResolveAttachments = &resolveRef;

        VkSubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask =
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependency.dstStageMask =
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependency.dstAccessMask =
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

        VkRenderPassCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        info.attachmentCount = 3;
        info.pAttachments = attachments;
        info.subpassCount = 1;
        info.pSubpasses = &subpass;
        info.dependencyCount = 1;
        info.pDependencies = &dependency;

        if (vkCreateRenderPass(
                device, &info, nullptr, &renderPass) != VK_SUCCESS)
            throw std::runtime_error("Render pass creation failed");
    }

    std::vector<char> readShader(const std::string& path) {
        std::ifstream file(path, std::ios::ate | std::ios::binary);
        if (!file)
            throw std::runtime_error("Cannot open shader: " + path);

        size_t size = static_cast<size_t>(file.tellg());
        std::vector<char> code(size);
        file.seekg(0);
        file.read(code.data(), static_cast<std::streamsize>(size));
        return code;
    }

    VkShaderModule createShaderModule(
        const std::vector<char>& code) {
        VkShaderModuleCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        info.codeSize = code.size();
        info.pCode = reinterpret_cast<const uint32_t*>(code.data());

        VkShaderModule module;
        if (vkCreateShaderModule(
                device, &info, nullptr, &module) != VK_SUCCESS)
            throw std::runtime_error("Shader module creation failed");

        return module;
    }

    void createPipeline() {
        auto vertCode = readShader(
            std::string(SHADER_DIR) + "/carflare.vert.spv");
        auto fragCode = readShader(
            std::string(SHADER_DIR) + "/carflare.frag.spv");

        VkShaderModule vert = createShaderModule(vertCode);
        VkShaderModule frag = createShaderModule(fragCode);

        VkPipelineShaderStageCreateInfo stages[2]{};

        stages[0].sType =
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vert;
        stages[0].pName = "main";

        stages[1].sType =
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = frag;
        stages[1].pName = "main";

        VkVertexInputBindingDescription binding{};
        binding.binding = 0;
        binding.stride = sizeof(Vertex);
        binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        std::array<VkVertexInputAttributeDescription, 3> attributes{};

        attributes[0].location = 0;
        attributes[0].binding = 0;
        attributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributes[0].offset = offsetof(Vertex, position);

        attributes[1].location = 1;
        attributes[1].binding = 0;
        attributes[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributes[1].offset = offsetof(Vertex, color);

        attributes[2].location = 2;
        attributes[2].binding = 0;
        attributes[2].format = VK_FORMAT_R32G32_SFLOAT;
        attributes[2].offset = offsetof(Vertex, texCoord);

        VkPipelineVertexInputStateCreateInfo vertexInput{};
        vertexInput.sType =
            VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInput.vertexBindingDescriptionCount = 1;
        vertexInput.pVertexBindingDescriptions = &binding;
        vertexInput.vertexAttributeDescriptionCount =
            static_cast<uint32_t>(attributes.size());
        vertexInput.pVertexAttributeDescriptions = attributes.data();

        VkPipelineInputAssemblyStateCreateInfo assembly{};
        assembly.sType =
            VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineViewportStateCreateInfo viewport{};
        viewport.sType =
            VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewport.viewportCount = 1;
        viewport.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo raster{};
        raster.sType =
            VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        raster.polygonMode = VK_POLYGON_MODE_FILL;
        raster.cullMode = VK_CULL_MODE_NONE;
        raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        raster.lineWidth = 1.0f;

        VkPipelineMultisampleStateCreateInfo multisample{};
        multisample.sType =
            VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisample.rasterizationSamples = msaaSamples;

        VkPipelineDepthStencilStateCreateInfo depth{};
        depth.sType =
            VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depth.depthTestEnable = VK_TRUE;
        depth.depthWriteEnable = VK_TRUE;
        depth.depthCompareOp = VK_COMPARE_OP_LESS;
        depth.depthBoundsTestEnable = VK_FALSE;
        depth.stencilTestEnable = VK_FALSE;

        VkPipelineColorBlendAttachmentState blend{};
        blend.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT |
            VK_COLOR_COMPONENT_G_BIT |
            VK_COLOR_COMPONENT_B_BIT |
            VK_COLOR_COMPONENT_A_BIT;

        VkPipelineColorBlendStateCreateInfo blending{};
        blending.sType =
            VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        blending.attachmentCount = 1;
        blending.pAttachments = &blend;

        VkDynamicState dynamicStates[] = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR
        };

        VkPipelineDynamicStateCreateInfo dynamic{};
        dynamic.sType =
            VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamic.dynamicStateCount = 2;
        dynamic.pDynamicStates = dynamicStates;

        VkPushConstantRange push{};
        push.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        push.offset = 0;
        push.size = sizeof(PushData);

        VkPipelineLayoutCreateInfo layout{};
        layout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layout.setLayoutCount = 1;
        layout.pSetLayouts = &textureSetLayout;
        layout.pushConstantRangeCount = 1;
        layout.pPushConstantRanges = &push;

        if (vkCreatePipelineLayout(
                device, &layout, nullptr, &pipelineLayout) != VK_SUCCESS)
            throw std::runtime_error("Pipeline layout creation failed");

        VkGraphicsPipelineCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        info.stageCount = 2;
        info.pStages = stages;
        info.pVertexInputState = &vertexInput;
        info.pInputAssemblyState = &assembly;
        info.pViewportState = &viewport;
        info.pRasterizationState = &raster;
        info.pMultisampleState = &multisample;
        info.pDepthStencilState = &depth;
        info.pColorBlendState = &blending;
        info.pDynamicState = &dynamic;
        info.layout = pipelineLayout;
        info.renderPass = renderPass;
        info.subpass = 0;

        VkResult result = vkCreateGraphicsPipelines(
            device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline);

        vkDestroyShaderModule(device, vert, nullptr);
        vkDestroyShaderModule(device, frag, nullptr);

        if (result != VK_SUCCESS)
            throw std::runtime_error("Graphics pipeline creation failed");

        auto hudVertCode = readShader(std::string(SHADER_DIR) + "/hud.vert.spv");
        auto hudFragCode = readShader(std::string(SHADER_DIR) + "/hud.frag.spv");
        VkShaderModule hudVert = createShaderModule(hudVertCode);
        VkShaderModule hudFrag = createShaderModule(hudFragCode);
        stages[0].module = hudVert;
        stages[1].module = hudFrag;
        depth.depthTestEnable = VK_FALSE;
        depth.depthWriteEnable = VK_FALSE;
        blend.blendEnable = VK_TRUE;
        blend.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        blend.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        blend.colorBlendOp = VK_BLEND_OP_ADD;
        blend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        blend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        result = vkCreateGraphicsPipelines(
            device, VK_NULL_HANDLE, 1, &info, nullptr, &hudPipeline);
        vkDestroyShaderModule(device, hudVert, nullptr);
        vkDestroyShaderModule(device, hudFrag, nullptr);
        if (result != VK_SUCCESS)
            throw std::runtime_error("HUD graphics pipeline creation failed");
    }

    void createFramebuffers() {
        framebuffers.resize(imageViews.size());

        for (size_t i = 0; i < imageViews.size(); ++i) {
            VkImageView attachments[] = {
                msaaColorViews[i], depthViews[i], imageViews[i]
            };

            VkFramebufferCreateInfo info{};
            info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            info.renderPass = renderPass;
            info.attachmentCount = 3;
            info.pAttachments = attachments;
            info.width = swapExtent.width;
            info.height = swapExtent.height;
            info.layers = 1;

            if (vkCreateFramebuffer(
                    device, &info, nullptr, &framebuffers[i])
                != VK_SUCCESS)
                throw std::runtime_error("Framebuffer creation failed");
        }
    }

    void createCommandPool() {
        auto queues = findQueues(gpu);

        VkCommandPoolCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        info.queueFamilyIndex = *queues.graphics;

        if (vkCreateCommandPool(
                device, &info, nullptr, &commandPool) != VK_SUCCESS)
            throw std::runtime_error("Command pool creation failed");
    }

    void createCommandBuffers() {
        commandBuffers.resize(framebuffers.size());

        VkCommandBufferAllocateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        info.commandPool = commandPool;
        info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        info.commandBufferCount =
            static_cast<uint32_t>(commandBuffers.size());

        if (vkAllocateCommandBuffers(
                device, &info, commandBuffers.data()) != VK_SUCCESS)
            throw std::runtime_error("Command buffer allocation failed");
    }

    void createSync() {
        VkSemaphoreCreateInfo semaphore{};
        semaphore.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fence{};
        fence.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fence.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        for (int i = 0; i < MAX_FRAMES; ++i) {
            if (vkCreateSemaphore(
                    device, &semaphore, nullptr, &imageAvailable[i])
                    != VK_SUCCESS ||
                vkCreateSemaphore(
                    device, &semaphore, nullptr, &renderFinished[i])
                    != VK_SUCCESS ||
                vkCreateFence(
                    device, &fence, nullptr, &inFlight[i])
                    != VK_SUCCESS)
                throw std::runtime_error("Sync object creation failed");
        }
    }

    void createSwapchainResources() {
        createSwapchain();
        createImageViews();
        createMSAAColorResources();
        createDepthResources();
        createRenderPass();
        createPipeline();
        createFramebuffers();
        createCommandBuffers();
        imagesInFlight.assign(swapImages.size(), VK_NULL_HANDLE);
    }

    glm::mat4 makeMVP(const glm::mat4& model) {
        glm::vec3 cameraPos(0.0f, 5.2f, 10.5f);
        glm::vec3 target(0.0f, 0.2f, -12.0f);

        glm::mat4 view = glm::lookAt(
            cameraPos, target, glm::vec3(0.0f, 1.0f, 0.0f));

        float aspect = static_cast<float>(swapExtent.width) /
                       static_cast<float>(swapExtent.height);

        glm::mat4 projection = glm::perspective(
            glm::radians(58.0f), aspect, 0.1f, 250.0f);
        projection[1][1] *= -1.0f;

        return projection * view * model;
    }

    void drawCube(VkCommandBuffer command, const glm::vec3& position,
                  const glm::vec3& scale, const glm::vec4& color,
                  float rotationY = 0.0f) {
        glm::mat4 model(1.0f);
        model = glm::translate(model, position);
        model = glm::rotate(model, rotationY, glm::vec3(0, 1, 0));
        model = glm::scale(model, scale);

        glm::mat4 mvp = makeMVP(model);
        const PushData pushData{mvp, color, glm::vec4(0.0f)};

        vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_GRAPHICS,
            pipelineLayout, 0, 1, &textureDescriptorSet, 0, nullptr);
        vkCmdPushConstants(command, pipelineLayout,
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            0, sizeof(pushData), &pushData);
        vkCmdDraw(command, static_cast<uint32_t>(vertices.size()), 1, 0, 0);
    }

    void recordCommandBuffer(VkCommandBuffer command, uint32_t index) {
        VkCommandBufferBeginInfo begin{};
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

        if (vkBeginCommandBuffer(command, &begin) != VK_SUCCESS)
            throw std::runtime_error("Command buffer begin failed");

        std::array<VkClearValue, 2> clear{};
        clear[0].color = {{0.43f, 0.68f, 0.88f, 1.0f}};
        clear[1].depthStencil = {1.0f, 0};

        VkRenderPassBeginInfo render{};
        render.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        render.renderPass = renderPass;
        render.framebuffer = framebuffers[index];
        render.renderArea.offset = {0, 0};
        render.renderArea.extent = swapExtent;
        render.clearValueCount = static_cast<uint32_t>(clear.size());
        render.pClearValues = clear.data();

        vkCmdBeginRenderPass(
            command, &render, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(
            command, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

        VkViewport viewport{};
        viewport.x = 0;
        viewport.y = 0;
        viewport.width = static_cast<float>(swapExtent.width);
        viewport.height = static_cast<float>(swapExtent.height);
        viewport.minDepth = 0;
        viewport.maxDepth = 1;

        VkRect2D scissor{};
        scissor.offset = {0, 0};
        scissor.extent = swapExtent;

        vkCmdSetViewport(command, 0, 1, &viewport);
        vkCmdSetScissor(command, 0, 1, &scissor);

        VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(
            command, 0, 1, &vertexBuffer, &offset);

        // Home screen – redesigned for the PBR lighting so panels stay readable
        // and the PLAY button is large, bright, and clearly clickable.
        if (!gameStarted) {
            // Outer frame (deep navy)
            drawCube(command, {0.0f, 2.15f, -2.05f}, {5.10f, 2.35f, 0.22f},
                     {0.10f, 0.14f, 0.22f, 1.0f});
            // Main panel face
            drawCube(command, {0.0f, 2.15f, -1.90f}, {4.70f, 2.05f, 0.10f},
                     {0.16f, 0.22f, 0.34f, 1.0f});
            // Inner recessed panel
            drawCube(command, {0.0f, 2.15f, -1.82f}, {4.30f, 1.75f, 0.06f},
                     {0.12f, 0.17f, 0.27f, 1.0f});

            // Top cyan accent bar
            drawCube(command, {0.0f, 3.15f, -1.72f}, {4.00f, 0.055f, 0.04f},
                     {0.25f, 0.85f, 1.00f, 1.0f});
            // Bottom cyan accent bar
            drawCube(command, {0.0f, 1.15f, -1.72f}, {4.00f, 0.055f, 0.04f},
                     {0.25f, 0.85f, 1.00f, 1.0f});

            // Side vertical accents
            drawCube(command, {-2.25f, 2.15f, -1.72f}, {0.055f, 0.90f, 0.04f},
                     {0.25f, 0.85f, 1.00f, 1.0f});
            drawCube(command, { 2.25f, 2.15f, -1.72f}, {0.055f, 0.90f, 0.04f},
                     {0.25f, 0.85f, 1.00f, 1.0f});

            // Title bar background
            drawCube(command, {0.0f, 2.75f, -1.74f}, {3.40f, 0.28f, 0.05f},
                     {0.08f, 0.12f, 0.20f, 1.0f});

            // "CAR FLARE" title letters (simple block style, bright white)
            const glm::vec4 titleCol(0.95f, 0.97f, 1.00f, 1.0f);
            const float ty = 2.75f;
            const float tz = -1.68f;
            // C
            drawCube(command, {-1.45f, ty, tz}, {0.06f, 0.18f, 0.03f}, titleCol);
            drawCube(command, {-1.32f, ty + 0.09f, tz}, {0.14f, 0.05f, 0.03f}, titleCol);
            drawCube(command, {-1.32f, ty - 0.09f, tz}, {0.14f, 0.05f, 0.03f}, titleCol);
            // A
            drawCube(command, {-1.05f, ty, tz}, {0.05f, 0.18f, 0.03f}, titleCol);
            drawCube(command, {-0.88f, ty, tz}, {0.05f, 0.18f, 0.03f}, titleCol);
            drawCube(command, {-0.965f, ty + 0.09f, tz}, {0.14f, 0.05f, 0.03f}, titleCol);
            drawCube(command, {-0.965f, ty - 0.02f, tz}, {0.12f, 0.04f, 0.03f}, titleCol);
            // R
            drawCube(command, {-0.65f, ty, tz}, {0.05f, 0.18f, 0.03f}, titleCol);
            drawCube(command, {-0.52f, ty + 0.09f, tz}, {0.12f, 0.05f, 0.03f}, titleCol);
            drawCube(command, {-0.52f, ty + 0.02f, tz}, {0.12f, 0.04f, 0.03f}, titleCol);
            drawCube(command, {-0.48f, ty + 0.055f, tz}, {0.05f, 0.07f, 0.03f}, titleCol);
            drawCube(command, {-0.48f, ty - 0.07f, tz}, {0.05f, 0.08f, 0.03f}, titleCol);
            // space
            // F
            drawCube(command, {-0.15f, ty, tz}, {0.05f, 0.18f, 0.03f}, titleCol);
            drawCube(command, {-0.02f, ty + 0.09f, tz}, {0.14f, 0.05f, 0.03f}, titleCol);
            drawCube(command, {-0.02f, ty + 0.01f, tz}, {0.11f, 0.04f, 0.03f}, titleCol);
            // L
            drawCube(command, {0.25f, ty, tz}, {0.05f, 0.18f, 0.03f}, titleCol);
            drawCube(command, {0.38f, ty - 0.09f, tz}, {0.13f, 0.05f, 0.03f}, titleCol);
            // A
            drawCube(command, {0.60f, ty, tz}, {0.05f, 0.18f, 0.03f}, titleCol);
            drawCube(command, {0.77f, ty, tz}, {0.05f, 0.18f, 0.03f}, titleCol);
            drawCube(command, {0.685f, ty + 0.09f, tz}, {0.14f, 0.05f, 0.03f}, titleCol);
            drawCube(command, {0.685f, ty - 0.02f, tz}, {0.12f, 0.04f, 0.03f}, titleCol);
            // R
            drawCube(command, {1.00f, ty, tz}, {0.05f, 0.18f, 0.03f}, titleCol);
            drawCube(command, {1.13f, ty + 0.09f, tz}, {0.12f, 0.05f, 0.03f}, titleCol);
            drawCube(command, {1.13f, ty + 0.02f, tz}, {0.12f, 0.04f, 0.03f}, titleCol);
            drawCube(command, {1.17f, ty + 0.055f, tz}, {0.05f, 0.07f, 0.03f}, titleCol);
            drawCube(command, {1.17f, ty - 0.07f, tz}, {0.05f, 0.08f, 0.03f}, titleCol);
            // E
            drawCube(command, {1.40f, ty, tz}, {0.05f, 0.18f, 0.03f}, titleCol);
            drawCube(command, {1.53f, ty + 0.09f, tz}, {0.13f, 0.05f, 0.03f}, titleCol);
            drawCube(command, {1.53f, ty, tz}, {0.11f, 0.04f, 0.03f}, titleCol);
            drawCube(command, {1.53f, ty - 0.09f, tz}, {0.13f, 0.05f, 0.03f}, titleCol);

            // Large green PLAY button (outer)
            drawCube(command, {0.0f, 1.55f, -1.70f}, {1.90f, 0.42f, 0.12f},
                     {0.10f, 0.65f, 0.32f, 1.0f});
            // Inner brighter face
            drawCube(command, {0.0f, 1.55f, -1.60f}, {1.70f, 0.30f, 0.06f},
                     {0.18f, 0.88f, 0.42f, 1.0f});

            // PLAY letters on the button (larger & clearer)
            const glm::vec4 playCol(1.00f, 1.00f, 0.98f, 1.0f);
            const float py = 1.55f;
            const float pz = -1.52f;
            // P
            drawCube(command, {-0.55f, py, pz}, {0.07f, 0.20f, 0.035f}, playCol);
            drawCube(command, {-0.42f, py + 0.085f, pz}, {0.16f, 0.055f, 0.035f}, playCol);
            drawCube(command, {-0.42f, py + 0.01f, pz}, {0.16f, 0.05f, 0.035f}, playCol);
            drawCube(command, {-0.36f, py + 0.05f, pz}, {0.06f, 0.08f, 0.035f}, playCol);
            // L
            drawCube(command, {-0.15f, py, pz}, {0.07f, 0.20f, 0.035f}, playCol);
            drawCube(command, {-0.02f, py - 0.085f, pz}, {0.15f, 0.055f, 0.035f}, playCol);
            // A
            drawCube(command, {0.22f, py, pz}, {0.07f, 0.20f, 0.035f}, playCol);
            drawCube(command, {0.40f, py, pz}, {0.07f, 0.20f, 0.035f}, playCol);
            drawCube(command, {0.31f, py + 0.085f, pz}, {0.18f, 0.055f, 0.035f}, playCol);
            drawCube(command, {0.31f, py - 0.01f, pz}, {0.15f, 0.045f, 0.035f}, playCol);
            // Y
            drawCube(command, {0.58f, py + 0.06f, pz}, {0.07f, 0.10f, 0.035f}, playCol, -0.55f);
            drawCube(command, {0.72f, py + 0.06f, pz}, {0.07f, 0.10f, 0.035f}, playCol,  0.55f);
            drawCube(command, {0.65f, py - 0.07f, pz}, {0.07f, 0.10f, 0.035f}, playCol);

            // Anton title overlay replaces the menu's legacy cube-letter look.
            drawMenuTitleHUD(command);

            vkCmdEndRenderPass(command);
            if (vkEndCommandBuffer(command) != VK_SUCCESS)
                throw std::runtime_error("Command buffer recording failed");
            return;
        }

        // Time-based movement makes the road markings loop forever.
        const float time = static_cast<float>(glfwGetTime());
        const bool forward = glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS;
        const bool reverse = glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS;
        const bool left = glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS;
        const bool right = glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS;

        static float speed = 0.0f;       // World units per second; forward is positive.
        static float carX = 0.0f;
        static float lateralVelocity = 0.0f;
        static float yaw = 0.0f;
        static float steeringAngle = 0.0f;
        static float roadTravel = 0.0f;
        static float lastTime = time;
        const float dt = std::clamp(time - lastTime, 0.0f, 0.033f);
        lastTime = time;

        // Fixed-step-friendly, physically inspired longitudinal dynamics.
        // Values are tuned for this game's deliberately compact world scale.
        constexpr float MAX_FORWARD_SPEED = 38.0f;
        constexpr float MAX_REVERSE_SPEED = 9.0f;
        constexpr float ENGINE_ACCEL = 8.2f;
        constexpr float SERVICE_BRAKE = 18.0f;
        constexpr float ROLLING_RESISTANCE = 0.32f;
        constexpr float AERO_DRAG = 0.0085f;
        constexpr float WHEELBASE = 2.55f;
        constexpr float MAX_STEER = 0.30f;
        constexpr float LATERAL_GRIP = 7.8f;

        const bool braking = (forward && speed < -0.05f) || (reverse && speed > 0.05f);
        if (braking) {
            const float sign = speed > 0.0f ? 1.0f : -1.0f;
            speed -= sign * SERVICE_BRAKE * dt;
            if (std::abs(speed) < SERVICE_BRAKE * dt) speed = 0.0f;
        } else if (forward) {
            speed = std::min(MAX_FORWARD_SPEED, speed + ENGINE_ACCEL * dt);
        } else if (reverse) {
            speed = std::max(-MAX_REVERSE_SPEED, speed - ENGINE_ACCEL * 0.58f * dt);
        } else {
            const float resistance = ROLLING_RESISTANCE + AERO_DRAG * speed * speed;
            if (speed > 0.0f) speed = std::max(0.0f, speed - resistance * dt);
            else if (speed < 0.0f) speed = std::min(0.0f, speed + resistance * dt);
        }

        const float steerInput = (right ? 1.0f : 0.0f) - (left ? 1.0f : 0.0f);
        const float speedFactor = std::clamp(std::abs(speed) / 12.0f, 0.0f, 1.0f);
        const float targetSteer = steerInput * MAX_STEER * (1.0f - 0.62f * speedFactor);
        // Steering rack has a finite response rate instead of teleporting angles.
        steeringAngle += (targetSteer - steeringAngle) * std::min(1.0f, dt * 3.6f);

        // Bicycle-model yaw rate, plus lateral tyre force that damps slip.
        const float yawRate = (std::abs(speed) > 0.01f)
            ? (speed / WHEELBASE) * std::tan(steeringAngle) : 0.0f;
        yaw += yawRate * dt;
        yaw = std::remainder(yaw, 6.28318530718f);
        const float desiredLateral = speed * std::sin(yaw);
        const float lateralAcceleration = std::clamp(
            (desiredLateral - lateralVelocity) * LATERAL_GRIP,
            -13.0f, 13.0f);
        lateralVelocity += lateralAcceleration * dt;
        carX += lateralVelocity * dt;

        // Grass has much less grip; the guardrail behaves as a compliant barrier.
        if (std::abs(carX) > 4.15f) {
            speed *= std::max(0.0f, 1.0f - 1.4f * dt);
            lateralVelocity *= std::max(0.0f, 1.0f - 2.5f * dt);
        }
        if (std::abs(carX) > 4.65f) {
            carX = std::copysign(4.65f, carX);
            lateralVelocity = -lateralVelocity * 0.22f;
            speed *= 0.72f;
        }
        carX = std::clamp(carX, -4.65f, 4.65f);

        // Road scrolling follows vehicle speed; integration remains frame-time based.
        roadTravel += speed * dt;
        // Score distance travelled under power; reversing cannot farm points.
        if (speed > 0.0f) {
            scoredDistance += speed * dt;
            const uint32_t newScore = static_cast<uint32_t>(scoredDistance * 10.0f);
            if (newScore != raceScore) {
                raceScore = newScore;
                glfwSetWindowTitle(window,
                    ("CarFlare - Score: " + std::to_string(raceScore) +
                     " | W A S D").c_str());
            }
        }
        const float roadOffset = std::fmod(roadTravel * 0.9f + 800.0f, 8.0f);

        // Realistic segmented highway – extends to the far horizon.
        // Darker materials so the near field no longer looks washed out.
        const glm::vec4 grassCol(0.22f, 0.34f, 0.18f, 1.0f);
        const glm::vec4 dirtCol(0.34f, 0.29f, 0.21f, 1.0f);
        const glm::vec4 asphaltCol(0.16f, 0.17f, 0.18f, 1.0f);
        const glm::vec4 wearCol(0.12f, 0.13f, 0.14f, 1.0f);
        const glm::vec4 shoulderCol(0.28f, 0.27f, 0.24f, 1.0f);
        const glm::vec4 kerbCol(0.45f, 0.44f, 0.41f, 1.0f);
        const glm::vec4 lineCol(0.88f, 0.87f, 0.82f, 1.0f);
        const glm::vec4 railCol(0.50f, 0.52f, 0.50f, 1.0f);
        const glm::vec4 postCol(0.34f, 0.35f, 0.34f, 1.0f);
        const glm::vec4 reflectorCol(0.88f, 0.72f, 0.22f, 1.0f);

        const float segLen = 7.0f;
        const float segOverlap = 0.12f;
        // Enough segments so the road reaches the top edge of the window
        const int numSegs = 28;

        for (int i = 0; i < numSegs; ++i) {
            const float z = 4.0f - i * segLen;
            // drawCube takes half-extents.  Using the full segment length here
            // made neighbouring coplanar surfaces overlap by almost a whole
            // segment, causing z-fighting on rails and painted edge lines.
            const float halfLen = (segLen + segOverlap) * 0.5f;

            // Grass
            drawCube(command, {-14.5f, -0.55f, z}, {16.0f, 0.22f, halfLen}, grassCol);
            drawCube(command, { 14.5f, -0.55f, z}, {16.0f, 0.22f, halfLen}, grassCol);

            // Dirt verge
            drawCube(command, {-7.4f, -0.40f, z}, {2.4f, 0.14f, halfLen}, dirtCol);
            drawCube(command, { 7.4f, -0.40f, z}, {2.4f, 0.14f, halfLen}, dirtCol);

            // Gravel shoulder
            drawCube(command, {0.0f, -0.30f, z}, {6.0f, 0.14f, halfLen}, shoulderCol);

            // Asphalt
            drawCube(command, {0.0f, -0.20f, z}, {5.20f, 0.11f, halfLen}, asphaltCol);

            // Tyre-wear strips
            drawCube(command, {-1.55f, -0.175f, z}, {1.15f, 0.02f, halfLen}, wearCol);
            drawCube(command, { 1.55f, -0.175f, z}, {1.15f, 0.02f, halfLen}, wearCol);

            // Kerbs
            drawCube(command, {-5.35f, -0.14f, z}, {0.18f, 0.12f, halfLen}, kerbCol);
            drawCube(command, { 5.35f, -0.14f, z}, {0.18f, 0.12f, halfLen}, kerbCol);

            // Solid white edge (shoulder) lines – continuous
            drawCube(command, {-4.92f, -0.045f, z}, {0.09f, 0.022f, halfLen}, lineCol);
            drawCube(command, { 4.92f, -0.045f, z}, {0.09f, 0.022f, halfLen}, lineCol);
        }

        // Broken centre line – realistic highway proportions:
        // ~3 unit dash, ~9 unit gap (1:3 ratio), ~10 cm wide paint.
        const float dashLen = 3.2f;
        const float dashGap = 9.0f;
        const float dashPeriod = dashLen + dashGap;
        for (int i = 0; i < 28; ++i) {
            const float z = -1.5f - i * dashPeriod
                          + std::fmod(roadOffset * 0.85f, dashPeriod);
            drawCube(command, {0.0f, -0.042f, z},
                     {0.10f, 0.020f, dashLen}, lineCol);
        }

        // Guardrail posts + amber reflectors
        for (int i = 0; i < 36; ++i) {
            const float z = -125.0f + std::fmod(i * 4.5f + roadOffset * 0.7f + 135.0f, 135.0f);
            for (int side : {-1, 1}) {
                const float x = side * 5.75f;
                drawCube(command, {x, 0.05f, z}, {0.08f, 0.45f, 0.08f}, postCol);
                drawCube(command, {x - side * 0.07f, 0.28f, z},
                         {0.03f, 0.08f, 0.08f}, reflectorCol);
            }
        }

        // Segmented dual rails
        for (int i = 0; i < numSegs; ++i) {
            const float z = 4.0f - i * segLen;
            const float halfLen = (segLen + segOverlap) * 0.5f;
            drawCube(command, {-5.75f, 0.48f, z}, {0.10f, 0.08f, halfLen}, railCol);
            drawCube(command, { 5.75f, 0.48f, z}, {0.10f, 0.08f, halfLen}, railCol);
            drawCube(command, {-5.75f, 0.28f, z}, {0.10f, 0.07f, halfLen}, railCol);
            drawCube(command, { 5.75f, 0.28f, z}, {0.10f, 0.07f, halfLen}, railCol);
        }

        // Load an external OBJ when assets/car.obj exists; otherwise keep the
        // original procedural car so the game remains playable out of the box.
        const float carZ = 2.0f;
        if (carModelLoaded) {
            drawCarModel(command, {carX, 0.0f, carZ}, {1.0f, 1.0f, 1.0f},
                         {1.0f, 1.0f, 1.0f, 1.0f}, -yaw * 0.65f,
                         std::clamp(-speed * 0.0015f, -0.045f, 0.045f),
                         std::clamp(-lateralVelocity * 0.025f, -0.12f, 0.12f));
        } else {
            drawCube(command, {carX, 0.40f, carZ}, {0.94f, 0.31f, 1.70f},
                     {0.055f, 0.22f, 0.48f, 1.0f});
            drawCube(command, {carX, 0.47f, carZ - 1.18f}, {0.78f, 0.20f, 0.48f},
                     {0.07f, 0.28f, 0.57f, 1.0f});
            drawCube(command, {carX, 0.74f, carZ - 0.32f}, {0.66f, 0.35f, 0.79f},
                     {0.055f, 0.075f, 0.095f, 1.0f});
            drawCube(command, {carX, 1.08f, carZ - 0.34f}, {0.52f, 0.055f, 0.53f},
                     {0.07f, 0.24f, 0.43f, 1.0f});
            drawCube(command, {carX, 0.29f, carZ - 1.69f}, {0.78f, 0.09f, 0.12f},
                     {0.10f, 0.11f, 0.12f, 1.0f});
            drawCube(command, {carX, 0.29f, carZ + 1.69f}, {0.78f, 0.09f, 0.12f},
                     {0.10f, 0.11f, 0.12f, 1.0f});
        }

        // Built-in lights and wheels are used only for the fallback car.
        if (!carModelLoaded) {
        // Warm-white headlights and red tail lamps.
        for (int side : {-1, 1}) {
            drawCube(command, {carX + side * 0.61f, 0.43f, carZ - 1.73f},
                     {0.19f, 0.085f, 0.045f},
                     {0.95f, 0.91f, 0.76f, 1.0f});
            drawCube(command, {carX + side * 0.61f, 0.43f, carZ + 1.73f},
                     {0.19f, 0.085f, 0.045f},
                     {0.72f, 0.035f, 0.025f, 1.0f});
        }

        // Four dark tyres and understated metallic hubs.
        for (int side : {-1, 1}) {
            for (int axle : {-1, 1}) {
                const float wheelX = carX + side * 0.91f;
                const float wheelZ = carZ + axle * 1.03f;
                drawCube(command, {wheelX, 0.13f, wheelZ},
                         {0.18f, 0.30f, 0.39f},
                         {0.035f, 0.038f, 0.04f, 1.0f});
                drawCube(command, {wheelX + side * 0.095f, 0.13f, wheelZ},
                         {0.025f, 0.16f, 0.20f},
                         {0.43f, 0.45f, 0.46f, 1.0f});
            }
        }
        }

        // HUD is recorded last so it is always visible over the racing scene.
        drawScoreHUD(command);

        vkCmdEndRenderPass(command);

        if (vkEndCommandBuffer(command) != VK_SUCCESS)
            throw std::runtime_error("Command buffer recording failed");
    }

    void cleanupSwapchain() {
        if (!commandBuffers.empty()) {
            vkFreeCommandBuffers(
                device, commandPool,
                static_cast<uint32_t>(commandBuffers.size()),
                commandBuffers.data());
            commandBuffers.clear();
        }

        for (auto fb : framebuffers)
            vkDestroyFramebuffer(device, fb, nullptr);
        framebuffers.clear();

        if (hudPipeline != VK_NULL_HANDLE)
            vkDestroyPipeline(device, hudPipeline, nullptr);
        hudPipeline = VK_NULL_HANDLE;

        if (pipeline != VK_NULL_HANDLE)
            vkDestroyPipeline(device, pipeline, nullptr);
        pipeline = VK_NULL_HANDLE;

        if (pipelineLayout != VK_NULL_HANDLE)
            vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        pipelineLayout = VK_NULL_HANDLE;

        if (renderPass != VK_NULL_HANDLE)
            vkDestroyRenderPass(device, renderPass, nullptr);
        renderPass = VK_NULL_HANDLE;

        for (auto view : imageViews)
            vkDestroyImageView(device, view, nullptr);
        imageViews.clear();

        for (auto view : msaaColorViews)
            vkDestroyImageView(device, view, nullptr);
        msaaColorViews.clear();

        for (auto image : msaaColorImages)
            vkDestroyImage(device, image, nullptr);
        msaaColorImages.clear();

        for (auto memory : msaaColorMemories)
            vkFreeMemory(device, memory, nullptr);
        msaaColorMemories.clear();

        for (auto view : depthViews)
            vkDestroyImageView(device, view, nullptr);
        depthViews.clear();

        for (auto image : depthImages)
            vkDestroyImage(device, image, nullptr);
        depthImages.clear();

        for (auto memory : depthMemories)
            vkFreeMemory(device, memory, nullptr);
        depthMemories.clear();

        if (swapchain != VK_NULL_HANDLE)
            vkDestroySwapchainKHR(device, swapchain, nullptr);
        swapchain = VK_NULL_HANDLE;
    }

    void recreateSwapchain() {
        int w = 0, h = 0;
        while (w == 0 || h == 0) {
            glfwGetFramebufferSize(window, &w, &h);
            glfwWaitEvents();
        }

        vkDeviceWaitIdle(device);
        cleanupSwapchain();
        createSwapchainResources();
        framebufferResized = false;
    }

    void drawFrame() {
        vkWaitForFences(
            device, 1, &inFlight[currentFrame],
            VK_TRUE, UINT64_MAX);

        uint32_t index = 0;
        VkResult result = vkAcquireNextImageKHR(
            device, swapchain, UINT64_MAX,
            imageAvailable[currentFrame], VK_NULL_HANDLE, &index);

        if (result == VK_ERROR_OUT_OF_DATE_KHR) {
            recreateSwapchain();
            return;
        }

        if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
            throw std::runtime_error("Swapchain image acquisition failed");

        if (imagesInFlight[index] != VK_NULL_HANDLE)
            vkWaitForFences(
                device, 1, &imagesInFlight[index], VK_TRUE, UINT64_MAX);

        imagesInFlight[index] = inFlight[currentFrame];

        vkResetFences(device, 1, &inFlight[currentFrame]);
        vkResetCommandBuffer(commandBuffers[index], 0);
        recordCommandBuffer(commandBuffers[index], index);

        VkPipelineStageFlags waitStage =
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

        VkSubmitInfo submit{};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.waitSemaphoreCount = 1;
        submit.pWaitSemaphores = &imageAvailable[currentFrame];
        submit.pWaitDstStageMask = &waitStage;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &commandBuffers[index];
        submit.signalSemaphoreCount = 1;
        submit.pSignalSemaphores = &renderFinished[currentFrame];

        if (vkQueueSubmit(
                graphicsQueue, 1, &submit, inFlight[currentFrame])
            != VK_SUCCESS)
            throw std::runtime_error("Graphics submission failed");

        VkPresentInfoKHR present{};
        present.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        present.waitSemaphoreCount = 1;
        present.pWaitSemaphores = &renderFinished[currentFrame];
        present.swapchainCount = 1;
        present.pSwapchains = &swapchain;
        present.pImageIndices = &index;

        result = vkQueuePresentKHR(presentQueue, &present);

        if (result == VK_ERROR_OUT_OF_DATE_KHR ||
            result == VK_SUBOPTIMAL_KHR || framebufferResized) {
            recreateSwapchain();
        } else if (result != VK_SUCCESS) {
            throw std::runtime_error("Presentation failed");
        }

        currentFrame = (currentFrame + 1) % MAX_FRAMES;
    }

    void cleanup() {
        if (device != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(device);
            cleanupSwapchain();

            if (vertexBuffer != VK_NULL_HANDLE)
                vkDestroyBuffer(device, vertexBuffer, nullptr);
            if (vertexMemory != VK_NULL_HANDLE)
                vkFreeMemory(device, vertexMemory, nullptr);
            if (carModelBuffer != VK_NULL_HANDLE)
                vkDestroyBuffer(device, carModelBuffer, nullptr);
            if (carModelMemory != VK_NULL_HANDLE)
                vkFreeMemory(device, carModelMemory, nullptr);
            if (textureDescriptorPool != VK_NULL_HANDLE)
                vkDestroyDescriptorPool(device, textureDescriptorPool, nullptr);
            if (textureSetLayout != VK_NULL_HANDLE)
                vkDestroyDescriptorSetLayout(device, textureSetLayout, nullptr);
            if (carTextureSampler != VK_NULL_HANDLE)
                vkDestroySampler(device, carTextureSampler, nullptr);
            if (carTextureView != VK_NULL_HANDLE)
                vkDestroyImageView(device, carTextureView, nullptr);
            if (carTextureImage != VK_NULL_HANDLE)
                vkDestroyImage(device, carTextureImage, nullptr);
            if (carTextureMemory != VK_NULL_HANDLE)
                vkFreeMemory(device, carTextureMemory, nullptr);
            if (hudVertexBuffer != VK_NULL_HANDLE)
                vkDestroyBuffer(device, hudVertexBuffer, nullptr);
            if (hudVertexMemory != VK_NULL_HANDLE)
                vkFreeMemory(device, hudVertexMemory, nullptr);
            if (hudFontSampler != VK_NULL_HANDLE)
                vkDestroySampler(device, hudFontSampler, nullptr);
            if (hudFontView != VK_NULL_HANDLE)
                vkDestroyImageView(device, hudFontView, nullptr);
            if (hudFontImage != VK_NULL_HANDLE)
                vkDestroyImage(device, hudFontImage, nullptr);
            if (hudFontMemory != VK_NULL_HANDLE)
                vkFreeMemory(device, hudFontMemory, nullptr);

            for (int i = 0; i < MAX_FRAMES; ++i) {
                if (imageAvailable[i])
                    vkDestroySemaphore(device, imageAvailable[i], nullptr);
                if (renderFinished[i])
                    vkDestroySemaphore(device, renderFinished[i], nullptr);
                if (inFlight[i])
                    vkDestroyFence(device, inFlight[i], nullptr);
            }

            if (commandPool)
                vkDestroyCommandPool(device, commandPool, nullptr);

            vkDestroyDevice(device, nullptr);
        }

        if (surface)
            vkDestroySurfaceKHR(instance, surface, nullptr);
        if (instance)
            vkDestroyInstance(instance, nullptr);
        if (window)
            glfwDestroyWindow(window);

        glfwTerminate();
    }

public:
    void run() {
        initWindow();
        initVulkan();

        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            drawFrame();
        }

        vkDeviceWaitIdle(device);
        cleanup();
    }
};


// Public interface implementation. Vulkan/game internals remain in this translation unit.
class CarFlare::Impl : public CarFlareImpl {};

CarFlare::CarFlare() : impl(std::make_unique<Impl>()) {}
CarFlare::~CarFlare() = default;
CarFlare::CarFlare(CarFlare&&) noexcept = default;
CarFlare& CarFlare::operator=(CarFlare&&) noexcept = default;
void CarFlare::run() { impl->run(); }
