#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D carTexture;

layout(push_constant) uniform PushData {
    mat4 mvp;
    vec4 color;
    vec4 params; // params.x: use car texture
} pushData;

void main() {
    if (pushData.params.x > 0.5) {
        outColor = texture(carTexture, fragTexCoord) * pushData.color;
    } else {
        float variation = 0.94 + 0.06 * abs(normalize(fragColor + vec3(0.001)).y);
        outColor = vec4(pushData.color.rgb * fragColor * variation, pushData.color.a);
    }
}
