
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

# Compile GLSL shaders to SPIR-V
set(SHADER_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
set(SHADER_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")

file(MAKE_DIRECTORY "${SHADER_OUTPUT_DIR}")

set(SHADER_FILES
    carflare.vert
    carflare.frag
)

set(SPIRV_FILES)

foreach(SHADER ${SHADER_FILES})
    set(SOURCE "${SHADER_SOURCE_DIR}/${SHADER}")
    set(OUTPUT "${SHADER_OUTPUT_DIR}/${SHADER}.spv")

    add_custom_command(
        OUTPUT "${OUTPUT}"
        COMMAND "${GLSLC}" "${SOURCE}" -o "${OUTPUT}"
        DEPENDS "${SOURCE}"
        COMMENT "Compiling ${SHADER} to SPIR-V"
        VERBATIM
    )

    list(APPEND SPIRV_FILES "${OUTPUT}")
endforeach()

add_custom_target(CarFlareShaders DEPENDS ${SPIRV_FILES})
add_dependencies(CarFlare CarFlareShaders)

# Tell the application where the compiled shaders are located
target_compile_definitions(CarFlare PRIVATE
    SHADER_DIR="${SHADER_OUTPUT_DIR}"
)
