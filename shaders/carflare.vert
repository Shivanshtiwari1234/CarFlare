#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 inTexCoord;

layout(location = 0) out vec2 fragTexCoord;
layout(location = 1) out vec3 localPosition;

layout(push_constant) uniform PushData {
    mat4 mvp;
} pushData;

void main() {
    gl_Position = pushData.mvp * vec4(inPosition, 1.0);

    fragTexCoord = inTexCoord;
    localPosition = inPosition;
}
