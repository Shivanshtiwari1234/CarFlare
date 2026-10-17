
#version 450

layout(location = 0) in vec3 inPosition;

layout(location = 0) out vec3 worldPosition;

layout(push_constant) uniform PushData {
    mat4 mvp;
    vec4 color;
} pushData;

void main() {
    gl_Position = pushData.mvp * vec4(inPosition, 1.0);
    worldPosition = inPosition;
}
