
#define GLFW_INCLUDE_VULKAN
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#include <GLFW/glfw3.h>
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
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef SHADER_DIR
#define SHADER_DIR "shaders"
#endif

constexpr uint32_t WIDTH = 1000;
constexpr uint32_t HEIGHT = 700;
constexpr int MAX_FRAMES = 2;

struct Vertex {
    glm::vec3 position;
    glm::vec3 color;
};

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

class CarFlare {
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

    std::vector<VkImage> depthImages;
    std::vector<VkDeviceMemory> depthMemories;
    std::vector<VkImageView> depthViews;
    VkFormat depthFormat{};

    std::array<VkSemaphore, MAX_FRAMES> imageAvailable{};
    std::array<VkSemaphore, MAX_FRAMES> renderFinished{};
    std::array<VkFence, MAX_FRAMES> inFlight{};
    std::vector<VkFence> imagesInFlight;

    size_t currentFrame = 0;
    bool framebufferResized = false;

    static void framebufferCallback(GLFWwindow* w, int, int) {
        auto* app = static_cast<CarFlare*>(
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
            info.samples = VK_SAMPLE_COUNT_1_BIT;
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

    void createRenderPass() {
        VkAttachmentDescription color{};
        color.format = swapFormat;
        color.samples = VK_SAMPLE_COUNT_1_BIT;
        color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        color.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        color.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        color.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        color.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentDescription depth{};
        depth.format = depthFormat;
        depth.samples = VK_SAMPLE_COUNT_1_BIT;
        depth.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depth.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depth.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depth.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depth.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depth.finalLayout =
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentDescription attachments[] = {color, depth};

        VkAttachmentReference colorRef{};
        colorRef.attachment = 0;
        colorRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkAttachmentReference depthRef{};
        depthRef.attachment = 1;
        depthRef.layout =
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorRef;
        subpass.pDepthStencilAttachment = &depthRef;

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
        info.attachmentCount = 2;
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

        std::array<VkVertexInputAttributeDescription, 2> attributes{};

        attributes[0].location = 0;
        attributes[0].binding = 0;
        attributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributes[0].offset = offsetof(Vertex, position);

        attributes[1].location = 1;
        attributes[1].binding = 0;
        attributes[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributes[1].offset = offsetof(Vertex, color);

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
        multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

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
        push.size = sizeof(glm::mat4) + sizeof(glm::vec4);

        VkPipelineLayoutCreateInfo layout{};
        layout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
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
    }

    void createFramebuffers() {
        framebuffers.resize(imageViews.size());

        for (size_t i = 0; i < imageViews.size(); ++i) {
            VkImageView attachments[] = {
                imageViews[i], depthViews[i]
            };

            VkFramebufferCreateInfo info{};
            info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            info.renderPass = renderPass;
            info.attachmentCount = 2;
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
            glm::radians(58.0f), aspect, 0.1f, 180.0f);
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
        struct PushData {
            glm::mat4 mvp;
            glm::vec4 color;
        } pushData{mvp, color};

        vkCmdPushConstants(
            command, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT |
            VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushData), &pushData);
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

        // Time-based movement makes the road markings loop forever.
        const float time = static_cast<float>(glfwGetTime());
        const bool forward = glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS;
        const bool reverse = glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS;
        const bool left = glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS;
        const bool right = glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS;

        static float speed = 0.0f;
        static float carX = 0.0f;
        static float lastTime = time;
        float dt = std::clamp(time - lastTime, 0.0f, 0.05f);
        lastTime = time;

        if (forward) speed = std::min(speed + 13.0f * dt, 30.0f);
        else if (reverse) speed = std::max(speed - 20.0f * dt, 0.0f);
        else speed = std::max(speed - 4.0f * dt, 0.0f);

        if (left) carX -= (3.0f + speed * 0.10f) * dt;
        if (right) carX += (3.0f + speed * 0.10f) * dt;
        carX = std::clamp(carX, -3.1f, 3.1f);

        const float roadOffset = std::fmod(time * speed * 0.9f, 8.0f);

        // Ground: muted grass and dry roadside earth beyond the asphalt.
        drawCube(command, {0.0f, -0.52f, -30.0f}, {45.0f, 0.22f, 70.0f},
                 {0.25f, 0.34f, 0.20f, 1.0f});
        drawCube(command, {0.0f, -0.31f, -28.0f}, {5.65f, 0.16f, 62.0f},
                 {0.31f, 0.32f, 0.31f, 1.0f});

        // Asphalt carriageway and narrow, pale concrete road edges.
        drawCube(command, {0.0f, -0.20f, -28.0f}, {5.15f, 0.12f, 62.0f},
                 {0.19f, 0.20f, 0.21f, 1.0f});
        drawCube(command, {-5.28f, -0.17f, -28.0f}, {0.16f, 0.10f, 62.0f},
                 {0.57f, 0.58f, 0.55f, 1.0f});
        drawCube(command, { 5.28f, -0.17f, -28.0f}, {0.16f, 0.10f, 62.0f},
                 {0.57f, 0.58f, 0.55f, 1.0f});

        // Continuous white edge lines.
        drawCube(command, {-4.90f, -0.065f, -28.0f}, {0.045f, 0.018f, 62.0f},
                 {0.86f, 0.87f, 0.84f, 1.0f});
        drawCube(command, { 4.90f, -0.065f, -28.0f}, {0.045f, 0.018f, 62.0f},
                 {0.86f, 0.87f, 0.84f, 1.0f});

        // Broken white centre line, scrolling toward the camera.
        for (int i = 0; i < 18; ++i) {
            const float z = -78.0f + std::fmod(i * 8.0f + roadOffset + 88.0f, 88.0f);
            drawCube(command, {0.0f, -0.055f, z}, {0.055f, 0.018f, 2.15f},
                     {0.91f, 0.92f, 0.88f, 1.0f});
        }

        // Distant low hills break up the horizon. Their subdued colours keep
        // the road and car as the visual focus.
        drawCube(command, {-13.0f, 1.8f, -58.0f}, {9.0f, 2.5f, 7.0f},
                 {0.39f, 0.45f, 0.34f, 1.0f});
        drawCube(command, {-25.0f, 2.4f, -68.0f}, {11.0f, 3.0f, 8.0f},
                 {0.43f, 0.48f, 0.37f, 1.0f});
        drawCube(command, { 14.0f, 1.9f, -61.0f}, {10.0f, 2.7f, 8.0f},
                 {0.40f, 0.46f, 0.35f, 1.0f});
        drawCube(command, { 27.0f, 2.6f, -70.0f}, {12.0f, 3.2f, 9.0f},
                 {0.45f, 0.49f, 0.39f, 1.0f});

        // Steel guardrails with regularly spaced grey support posts.
        for (int i = 0; i < 20; ++i) {
            const float z = -82.0f + std::fmod(i * 5.0f + roadOffset * 0.7f + 90.0f, 90.0f);
            for (int side : {-1, 1}) {
                const float x = side * 5.65f;
                drawCube(command, {x, 0.02f, z}, {0.075f, 0.42f, 0.075f},
                         {0.43f, 0.45f, 0.44f, 1.0f});
                // Small reflector on each post.
                drawCube(command, {x - side * 0.06f, 0.20f, z},
                         {0.025f, 0.075f, 0.07f},
                         {0.82f, 0.78f, 0.58f, 1.0f});
            }
        }
        drawCube(command, {-5.66f, 0.42f, -28.0f}, {0.09f, 0.09f, 45.0f},
                 {0.48f, 0.50f, 0.49f, 1.0f});
        drawCube(command, { 5.66f, 0.42f, -28.0f}, {0.09f, 0.09f, 45.0f},
                 {0.48f, 0.50f, 0.49f, 1.0f});

        // Compact metallic-blue sports car. The body is built from overlapping
        // simple meshes, so it remains lightweight on integrated graphics.
        const float carZ = 2.0f;
        drawCube(command, {carX, 0.40f, carZ}, {0.94f, 0.31f, 1.70f},
                 {0.055f, 0.22f, 0.48f, 1.0f});
        drawCube(command, {carX, 0.47f, carZ - 1.18f}, {0.78f, 0.20f, 0.48f},
                 {0.07f, 0.28f, 0.57f, 1.0f});
        drawCube(command, {carX, 0.74f, carZ - 0.32f}, {0.66f, 0.35f, 0.79f},
                 {0.055f, 0.075f, 0.095f, 1.0f});
        // Subtle roof panel and front/rear bumpers.
        drawCube(command, {carX, 1.08f, carZ - 0.34f}, {0.52f, 0.055f, 0.53f},
                 {0.07f, 0.24f, 0.43f, 1.0f});
        drawCube(command, {carX, 0.29f, carZ - 1.69f}, {0.78f, 0.09f, 0.12f},
                 {0.10f, 0.11f, 0.12f, 1.0f});
        drawCube(command, {carX, 0.29f, carZ + 1.69f}, {0.78f, 0.09f, 0.12f},
                 {0.10f, 0.11f, 0.12f, 1.0f});

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

int main() {
    try {
        CarFlare app;
        app.run();
    } catch (const std::exception& e) {
        std::cerr << "Car Flare error: " << e.what() << '\n';
        return 1;
    }

    return 0;
}
