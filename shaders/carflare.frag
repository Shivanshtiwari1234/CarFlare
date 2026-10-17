
#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;
layout(location = 2) in vec3 localPosition;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D carTexture;

layout(push_constant) uniform PushData {
    mat4 mvp;
    vec4 color;
    vec4 params; // params.x: use car texture
} pushData;

void main() {
    vec4 base = pushData.params.x > 0.5
        ? texture(carTexture, fragTexCoord) * pushData.color
        : vec4(pushData.color.rgb * fragColor, pushData.color.a);

    // Calculate a face normal from screen-space derivatives.
    vec3 dx = dFdx(localPosition);
    vec3 dy = dFdy(localPosition);
    vec3 crossNormal = cross(dx, dy);

    float normalLength2 = dot(crossNormal, crossNormal);

    vec3 normal = normalLength2 > 1e-10
        ? crossNormal * inversesqrt(normalLength2)
        : vec3(0.0, 1.0, 0.0);

    if (!gl_FrontFacing) {
        normal = -normal;
    }

    // Directional sunlight.
    vec3 sunDir = normalize(vec3(-0.42, 0.82, 0.36));
    float ndotl = max(dot(normal, sunDir), 0.0);

    // Neutral ambient fill prevents shadows from becoming too dark.
    float hemi = clamp(0.5 + 0.5 * normal.y, 0.0, 1.0);

    vec3 ambient = mix(
        vec3(0.30, 0.30, 0.32),
        vec3(0.52, 0.55, 0.58),
        hemi
    );

    vec3 sunColor = vec3(1.0, 0.97, 0.92);
    vec3 diffuse = sunColor * ndotl * 1.25;

    // Blinn-Phong specular highlights.
    vec3 viewDir = normalize(vec3(0.0, 0.6, 1.0));
    vec3 halfDir = normalize(sunDir + viewDir);

    float specPower = mix(
        32.0,
        80.0,
        clamp(base.r * 0.6 + base.g * 0.3, 0.0, 1.0)
    );

    float specular = pow(
        max(dot(normal, halfDir), 0.0),
        specPower
    ) * ndotl;

    float metallic = pushData.params.x > 0.5 ? 0.22 : 0.06;

    // Combine material color, ambient light, sunlight, and highlights.
    vec3 lit = base.rgb * (ambient + diffuse)
             + sunColor * specular * metallic;

    // Softer tone mapping to preserve dark surface details.
    lit = max(lit, vec3(0.0));
    lit = lit / (vec3(1.0) + lit * 0.10);
    lit = pow(lit, vec3(0.90));

    outColor = vec4(clamp(lit, 0.0, 1.0), base.a);
}
