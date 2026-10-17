
cmake_minimum_required(VERSION 3.20)
project(CarFlare LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Vulkan REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)

find_program(GLSLC glslc REQUIRED)

add_executable(CarFlare main.cpp)

target_link_libraries(CarFlare PRIVATE
    Vulkan::Vulkan
    glfw
    glm::glm
)

set(SHADER_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
file(MAKE_DIRECTORY "${SHADER_OUTPUT_DIR}")

foreach(SHADER cube.vert cube.frag)
    add_custom_command(
        OUTPUT "${SHADER_OUTPUT_DIR}/${SHADER}.spv"
        COMMAND "${GLSLC}"
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${SHADER}"
            -o "${SHADER_OUTPUT_DIR}/${SHADER}.spv"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${SHADER}"
        VERBATIM
    )
    list(APPEND SPIRV_OUTPUTS
        "${SHADER_OUTPUT_DIR}/${SHADER}.spv")
endforeach()

add_custom_target(Shaders DEPENDS ${SPIRV_OUTPUTS})
add_dependencies(CarFlare Shaders)

target_compile_definitions(CarFlare PRIVATE
    SHADER_DIR="${SHADER_OUTPUT_DIR}"
)
