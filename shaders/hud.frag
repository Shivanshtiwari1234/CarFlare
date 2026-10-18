#version 450
layout(location=0) in vec2 uv;
layout(location=0) out vec4 outColor;
layout(set=0,binding=0) uniform sampler2D fontAtlas;
void main() { float a = texture(fontAtlas, uv).a; outColor = vec4(1.0, 0.92, 0.35, a); }
