
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
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
constexpr int MAX_FRAMES_IN_FLIGHT = 2;

const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
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

private:
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

    std::array<VkSemaphore, MAX_FRAMES_IN_FLIGHT> imageAvailable{};
    std::array<VkSemaphore, MAX_FRAMES_IN_FLIGHT> renderFinished{};
    std::array<VkFence, MAX_FRAMES_IN_FLIGHT> inFlight{};

    std::vector<VkFence> imagesInFlight;
    size_t currentFrame = 0;
    bool framebufferResized = false;

    static void framebufferCallback(
        GLFWwindow* w, int, int
    ) {
        auto* app = static_cast<CarFlare*>(
            glfwGetWindowUserPointer(w)
        );
        app->framebufferResized = true;
    }

    void initWindow() {
        if (!glfwInit())
            throw std::runtime_error("GLFW initialization failed");

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

        window = glfwCreateWindow(
            WIDTH, HEIGHT, "Car Flare - Vulkan", nullptr, nullptr
        );

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
            throw std::runtime_error("Vulkan instance creation failed");
    }

    QueueFamilies findQueues(VkPhysicalDevice candidate) {
        QueueFamilies result;

        uint32_t count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(
            candidate, &count, nullptr
        );

        std::vector<VkQueueFamilyProperties> families(count);
        vkGetPhysicalDeviceQueueFamilyProperties(
            candidate, &count, families.data()
        );

        for (uint32_t i = 0; i < count; ++i) {
            if (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
                result.graphics = i;

            VkBool32 supported = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(
                candidate, i, surface, &supported
            );

            if (supported)
                result.present = i;

            if (result.complete())
                break;
        }

        return result;
    }

    bool hasExtensions(VkPhysicalDevice candidate) {
        uint32_t count = 0;
        vkEnumerateDeviceExtensionProperties(
            candidate, nullptr, &count, nullptr
        );

        std::vector<VkExtensionProperties> available(count);
        vkEnumerateDeviceExtensionProperties(
            candidate, nullptr, &count, available.data()
        );

        for (const char* required : deviceExtensions) {
            bool found = false;

            for (const auto& extension : available) {
                if (std::string(extension.extensionName) == required) {
                    found = true;
                    break;
                }
            }

            if (!found)
                return false;
        }

        return true;
    }

    SwapSupport querySwap(VkPhysicalDevice candidate) {
        SwapSupport result;

        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
            candidate, surface, &result.capabilities
        );

        uint32_t count = 0;
        vkGetPhysicalDeviceSurfaceFormatsKHR(
            candidate, surface, &count, nullptr
        );
        result.formats.resize(count);
        vkGetPhysicalDeviceSurfaceFormatsKHR(
            candidate, surface, &count, result.formats.data()
        );

        count = 0;
        vkGetPhysicalDeviceSurfacePresentModesKHR(
            candidate, surface, &count, nullptr
        );
        result.modes.resize(count);
        vkGetPhysicalDeviceSurfacePresentModesKHR(
            candidate, surface, &count, result.modes.data()
        );

        return result;
    }

    bool suitable(VkPhysicalDevice candidate) {
        QueueFamilies queues = findQueues(candidate);

        if (!queues.complete() || !hasExtensions(candidate))
            return false;

        SwapSupport support = querySwap(candidate);

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
                "No GPU supports graphics and presentation"
            );

        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(gpu, &properties);
        std::cout << "Using GPU: " << properties.deviceName << '\n';
    }

    void createDevice() {
        QueueFamilies queues = findQueues(gpu);

        std::vector<uint32_t> unique = {
            *queues.graphics, *queues.present
        };

        std::sort(unique.begin(), unique.end());
        unique.erase(std::unique(unique.begin(), unique.end()),
                     unique.end());

        float priority = 1.0f;
        std::vector<VkDeviceQueueCreateInfo> queueInfos;

        for (uint32_t family : unique) {
            VkDeviceQueueCreateInfo info{};
            info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            info.queueFamilyIndex = family;
            info.queueCount = 1;
            info.pQueuePriorities = &priority;
            queueInfos.push_back(info);
        }

        VkPhysicalDeviceFeatures features{};

        VkDeviceCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        info.queueCreateInfoCount =
            static_cast<uint32_t>(queueInfos.size());
        info.pQueueCreateInfos = queueInfos.data();
        info.enabledExtensionCount =
            static_cast<uint32_t>(deviceExtensions.size());
        info.ppEnabledExtensionNames = deviceExtensions.data();
        info.pEnabledFeatures = &features;

        if (vkCreateDevice(gpu, &info, nullptr, &device) != VK_SUCCESS)
            throw std::runtime_error("Logical device creation failed");

        vkGetDeviceQueue(device, *queues.graphics, 0, &graphicsQueue);
        vkGetDeviceQueue(device, *queues.present, 0, &presentQueue);
    }

    VkSurfaceFormatKHR chooseFormat(
        const std::vector<VkSurfaceFormatKHR>& formats
    ) {
        for (const auto& format : formats) {
            if (format.format == VK_FORMAT_B8G8R8A8_SRGB &&
                format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
                return format;
        }

        return formats.front();
    }

    VkPresentModeKHR choosePresent(
        const std::vector<VkPresentModeKHR>& modes
    ) {
        for (auto mode : modes) {
            if (mode == VK_PRESENT_MODE_MAILBOX_KHR)
                return mode;
        }

        return VK_PRESENT_MODE_FIFO_KHR;
    }

    VkExtent2D chooseExtent(
        const VkSurfaceCapabilitiesKHR& capabilities
    ) {
        if (capabilities.currentExtent.width !=
            std::numeric_limits<uint32_t>::max())
            return capabilities.currentExtent;

        int width, height;
        glfwGetFramebufferSize(window, &width, &height);

        VkExtent2D extent = {
            static_cast<uint32_t>(width),
            static_cast<uint32_t>(height)
        };

        extent.width = std::clamp(
            extent.width,
            capabilities.minImageExtent.width,
            capabilities.maxImageExtent.width
        );

        extent.height = std::clamp(
            extent.height,
            capabilities.minImageExtent.height,
            capabilities.maxImageExtent.height
        );

        return extent;
    }

    void createSwapchain() {
        SwapSupport support = querySwap(gpu);
        auto format = chooseFormat(support.formats);
        auto mode = choosePresent(support.modes);
        auto extent = chooseExtent(support.capabilities);

        uint32_t imageCount =
            support.capabilities.minImageCount + 1;

        if (support.capabilities.maxImageCount > 0)
            imageCount = std::min(
                imageCount, support.capabilities.maxImageCount
            );

        QueueFamilies queues = findQueues(gpu);
        uint32_t indices[] = {
            *queues.graphics, *queues.present
        };

        VkSwapchainCreateInfoKHR info{};
        info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        info.surface = surface;
        info.minImageCount = imageCount;
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
            device, swapchain, &imageCount, nullptr
        );
        swapImages.resize(imageCount);
        vkGetSwapchainImagesKHR(
            device, swapchain, &imageCount, swapImages.data()
        );

        swapFormat = format.format;
        swapExtent = extent;
    }

    void createImageViews() {
        imageViews.resize(swapImages.size());

        for (size_t i = 0; i < swapImages.size(); ++i) {
            VkImageViewCreateInfo info{};
            info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            info.image = swapImages[i];
            info.viewType = VK_IMAGE_VIEW_TYPE_2D;
            info.format = swapFormat;
            info.components = {
                VK_COMPONENT_SWIZZLE_IDENTITY,
                VK_COMPONENT_SWIZZLE_IDENTITY,
                VK_COMPONENT_SWIZZLE_IDENTITY,
                VK_COMPONENT_SWIZZLE_IDENTITY
            };
            info.subresourceRange.aspectMask =
                VK_IMAGE_ASPECT_COLOR_BIT;
            info.subresourceRange.baseMipLevel = 0;
            info.subresourceRange.levelCount = 1;
            info.subresourceRange.baseArrayLayer = 0;
            info.subresourceRange.layerCount = 1;

            if (vkCreateImageView(
                    device, &info, nullptr, &imageViews[i])
                != VK_SUCCESS)
                throw std::runtime_error("Image view creation failed");
        }
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

    VkShaderModule shaderModule(const std::vector<char>& code) {
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

        VkAttachmentReference reference{};
        reference.attachment = 0;
        reference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &reference;

        VkSubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask =
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.dstStageMask =
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.dstAccessMask =
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        VkRenderPassCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        info.attachmentCount = 1;
        info.pAttachments = &color;
        info.subpassCount = 1;
        info.pSubpasses = &subpass;
        info.dependencyCount = 1;
        info.pDependencies = &dependency;

        if (vkCreateRenderPass(
                device, &info, nullptr, &renderPass) != VK_SUCCESS)
            throw std::runtime_error("Render pass creation failed");
    }

    void createPipeline() {
        auto vertCode = readShader(
            std::string(SHADER_DIR) + "/triangle.vert.spv"
        );
        auto fragCode = readShader(
            std::string(SHADER_DIR) + "/triangle.frag.spv"
        );

        VkShaderModule vert = shaderModule(vertCode);
        VkShaderModule frag = shaderModule(fragCode);

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

        VkPipelineVertexInputStateCreateInfo vertexInput{};
        vertexInput.sType =
            VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

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
        raster.depthClampEnable = VK_FALSE;
        raster.rasterizerDiscardEnable = VK_FALSE;
        raster.polygonMode = VK_POLYGON_MODE_FILL;
        raster.lineWidth = 1.0f;
        raster.cullMode = VK_CULL_MODE_NONE;
        raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

        VkPipelineMultisampleStateCreateInfo multisample{};
        multisample.sType =
            VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

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

        VkDynamicState dynamics[] = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR
        };

        VkPipelineDynamicStateCreateInfo dynamic{};
        dynamic.sType =
            VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamic.dynamicStateCount = 2;
        dynamic.pDynamicStates = dynamics;

        VkPipelineLayoutCreateInfo layout{};
        layout.sType =
            VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;

        if (vkCreatePipelineLayout(
                device, &layout, nullptr, &pipelineLayout) != VK_SUCCESS)
            throw std::runtime_error("Pipeline layout creation failed");

        VkGraphicsPipelineCreateInfo info{};
        info.sType =
            VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        info.stageCount = 2;
        info.pStages = stages;
        info.pVertexInputState = &vertexInput;
        info.pInputAssemblyState = &assembly;
        info.pViewportState = &viewport;
        info.pRasterizationState = &raster;
        info.pMultisampleState = &multisample;
        info.pColorBlendState = &blending;
        info.pDynamicState = &dynamic;
        info.layout = pipelineLayout;
        info.renderPass = renderPass;
        info.subpass = 0;

        VkResult result = vkCreateGraphicsPipelines(
            device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline
        );

        vkDestroyShaderModule(device, vert, nullptr);
        vkDestroyShaderModule(device, frag, nullptr);

        if (result != VK_SUCCESS)
            throw std::runtime_error("Graphics pipeline creation failed");
    }

    void createFramebuffers() {
        framebuffers.resize(imageViews.size());

        for (size_t i = 0; i < imageViews.size(); ++i) {
            VkFramebufferCreateInfo info{};
            info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            info.renderPass = renderPass;
            info.attachmentCount = 1;
            info.pAttachments = &imageViews[i];
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
        QueueFamilies queues = findQueues(gpu);

        VkCommandPoolCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        info.queueFamilyIndex = *queues.graphics;
        info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

        if (vkCreateCommandPool(
                device, &info, nullptr, &commandPool) != VK_SUCCESS)
            throw std::runtime_error("Command pool creation failed");
    }

    void createCommandBuffers() {
        commandBuffers.resize(framebuffers.size());

        VkCommandBufferAllocateInfo allocation{};
        allocation.sType =
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocation.commandPool = commandPool;
        allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocation.commandBufferCount =
            static_cast<uint32_t>(commandBuffers.size());

        if (vkAllocateCommandBuffers(
                device, &allocation, commandBuffers.data()) != VK_SUCCESS)
            throw std::runtime_error("Command buffer allocation failed");
    }

    void recordCommandBuffer(VkCommandBuffer command, uint32_t index) {
        VkCommandBufferBeginInfo begin{};
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

        if (vkBeginCommandBuffer(command, &begin) != VK_SUCCESS)
            throw std::runtime_error("Command buffer begin failed");

        VkClearValue clear{};
        clear.color = {{0.035f, 0.045f, 0.075f, 1.0f}};

        VkRenderPassBeginInfo render{};
        render.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        render.renderPass = renderPass;
        render.framebuffer = framebuffers[index];
        render.renderArea.offset = {0, 0};
        render.renderArea.extent = swapExtent;
        render.clearValueCount = 1;
        render.pClearValues = &clear;

        vkCmdBeginRenderPass(
            command, &render, VK_SUBPASS_CONTENTS_INLINE
        );

        vkCmdBindPipeline(
            command, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline
        );

        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = static_cast<float>(swapExtent.width);
        viewport.height = static_cast<float>(swapExtent.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;

        VkRect2D scissor{};
        scissor.offset = {0, 0};
        scissor.extent = swapExtent;

        vkCmdSetViewport(command, 0, 1, &viewport);
        vkCmdSetScissor(command, 0, 1, &scissor);

        vkCmdDraw(command, 3, 1, 0, 0);

        vkCmdEndRenderPass(command);

        if (vkEndCommandBuffer(command) != VK_SUCCESS)
            throw std::runtime_error("Command buffer recording failed");
    }

    void createSync() {
        VkSemaphoreCreateInfo semaphore{};
        semaphore.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fence{};
        fence.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fence.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
            if (vkCreateSemaphore(
                    device, &semaphore, nullptr, &imageAvailable[i])
                    != VK_SUCCESS ||
                vkCreateSemaphore(
                    device, &semaphore, nullptr, &renderFinished[i])
                    != VK_SUCCESS ||
                vkCreateFence(
                    device, &fence, nullptr, &inFlight[i])
                    != VK_SUCCESS)
                throw std::runtime_error("Synchronization creation failed");
        }
    }

    void createSwapchainResources() {
        createSwapchain();
        createImageViews();
        createRenderPass();
        createPipeline();
        createFramebuffers();
        createCommandBuffers();
        imagesInFlight.assign(swapImages.size(), VK_NULL_HANDLE);
    }

    void cleanupSwapchain() {
        if (!commandBuffers.empty()) {
            vkFreeCommandBuffers(
                device, commandPool,
                static_cast<uint32_t>(commandBuffers.size()),
                commandBuffers.data()
            );
            commandBuffers.clear();
        }

        for (auto framebuffer : framebuffers)
            vkDestroyFramebuffer(device, framebuffer, nullptr);
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

        if (swapchain != VK_NULL_HANDLE)
            vkDestroySwapchainKHR(device, swapchain, nullptr);
        swapchain = VK_NULL_HANDLE;
    }

    void recreateSwapchain() {
        int width = 0, height = 0;

        while (width == 0 || height == 0) {
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        }

        vkDeviceWaitIdle(device);
        cleanupSwapchain();
        createSwapchainResources();
        framebufferResized = false;
    }

    void drawFrame() {
        vkWaitForFences(
            device, 1, &inFlight[currentFrame], VK_TRUE, UINT64_MAX
        );

        uint32_t index = 0;
        VkResult result = vkAcquireNextImageKHR(
            device, swapchain, UINT64_MAX,
            imageAvailable[currentFrame], VK_NULL_HANDLE, &index
        );

        if (result == VK_ERROR_OUT_OF_DATE_KHR) {
            recreateSwapchain();
            return;
        }

        if (result != VK_SUCCESS &&
            result != VK_SUBOPTIMAL_KHR)
            throw std::runtime_error("Failed to acquire swapchain image");

        if (imagesInFlight[index] != VK_NULL_HANDLE)
            vkWaitForFences(
                device, 1, &imagesInFlight[index],
                VK_TRUE, UINT64_MAX
            );

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
            throw std::runtime_error("Graphics queue submission failed");

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

        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    void cleanup() {
        if (device != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(device);
            cleanupSwapchain();

            for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
                vkDestroySemaphore(device, imageAvailable[i], nullptr);
                vkDestroySemaphore(device, renderFinished[i], nullptr);
                vkDestroyFence(device, inFlight[i], nullptr);
            }

            if (commandPool != VK_NULL_HANDLE)
                vkDestroyCommandPool(device, commandPool, nullptr);

            vkDestroyDevice(device, nullptr);
        }

        if (surface != VK_NULL_HANDLE)
            vkDestroySurfaceKHR(instance, surface, nullptr);

        if (instance != VK_NULL_HANDLE)
            vkDestroyInstance(instance, nullptr);

        if (window)
            glfwDestroyWindow(window);

        glfwTerminate();
    }
};

int main() {
    try {
        CarFlare app;
        app.run();
    } catch (const std::exception& error) {
        std::cerr << "Car Flare error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
