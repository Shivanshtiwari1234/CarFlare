
#version 450

layout(location = 0) in vec3 worldPosition;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform PushData {
    mat4 mvp;
    vec4 color;
} pushData;

void main() {
    // Subtle face shading gives cube-based objects more depth.
    float variation = 0.94 + 0.06 * abs(normalize(worldPosition + vec3(0.001)).y);
    outColor = vec4(pushData.color.rgb * variation, pushData.color.a);
}
