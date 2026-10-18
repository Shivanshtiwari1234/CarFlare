#version 450

layout(location = 0) in vec2 fragTexCoord;
layout(location = 1) in vec3 localPosition;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D carTexture;

layout(push_constant) uniform PushData {
    layout(offset = 64) vec4 color;
    layout(offset = 80) vec4 params;     // params.x: use car texture (>0.5)
} pushData;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
const float PI     = 3.14159265359;
const float INV_PI = 0.31830988618;

// Soft daylight – lower intensity keeps near-field from blowing out
const vec3  SUN_DIR       = normalize(vec3(-0.35, 0.88, 0.30));
const vec3  SUN_COLOR     = vec3(1.00, 0.98, 0.94);
const float SUN_INTENSITY = 2.4;

// Hemisphere ambient (sky + ground)
const vec3 SKY_ZENITH  = vec3(0.55, 0.65, 0.82);
const vec3 SKY_HORIZON = vec3(0.72, 0.78, 0.85);
const vec3 GROUND_ALB  = vec3(0.30, 0.29, 0.26);

// ---------------------------------------------------------------------------
// GGX / Smith / Schlick (Cook-Torrance)
// ---------------------------------------------------------------------------
float D_GGX(float NdotH, float roughness) {
    float a  = max(roughness * roughness, 0.002);
    float a2 = a * a;
    float d  = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

float G_SchlickGGX(float NdotX, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) * 0.125;
    return NdotX / (NdotX * (1.0 - k) + k);
}

float G_Smith(float NdotV, float NdotL, float roughness) {
    return G_SchlickGGX(NdotV, roughness) * G_SchlickGGX(NdotL, roughness);
}

vec3 F_Schlick(float cosTheta, vec3 F0) {
    float f = pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
    return F0 + (1.0 - F0) * f;
}

// Soft hemisphere ambient – strong enough that flat ground stays evenly lit
vec3 ambientIrradiance(vec3 N) {
    float hemi = clamp(N.y * 0.5 + 0.5, 0.0, 1.0);
    vec3 sky = mix(SKY_HORIZON, SKY_ZENITH, hemi * hemi);
    return mix(GROUND_ALB, sky, hemi) * 0.85;
}

// ACES filmic (Narkowicz)
vec3 ACESFilm(vec3 x) {
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

// Avoid undefined normalize(0) results at the camera position or when a
// half-vector collapses (view direction opposite the sun).
vec3 safeNormalize(vec3 value, vec3 fallback) {
    float lengthSquared = dot(value, value);
    return lengthSquared > 1e-12 ? value * inversesqrt(lengthSquared) : fallback;
}

// ---------------------------------------------------------------------------
void main() {
    // Albedo – ignore bright vertex colours on procedural cubes
    vec4 base = pushData.params.x > 0.5
        ? texture(carTexture, fragTexCoord) * pushData.color
        : pushData.color;
    vec3 albedo = clamp(base.rgb, vec3(0.0), vec3(1.0));

    // Model-space derivatives remain stable because every mesh face is flat.
    // Keeping this independent of the transform lets the complete interface
    // fit in Vulkan's guaranteed push-constant capacity.
    vec3 dx = dFdx(localPosition);
    vec3 dy = dFdy(localPosition);
    vec3 faceN = cross(dx, dy);
    float len2 = dot(faceN, faceN);
    vec3 N = len2 > 1e-12 ? faceN * inversesqrt(len2) : vec3(0.0, 1.0, 0.0);
    if (!gl_FrontFacing) N = -N;
    // Force clean up-vector on nearly horizontal faces (road / grass)
    if (abs(N.y) > 0.80)
        N = vec3(0.0, N.y < 0.0 ? -1.0 : 1.0, 0.0);

    // Materials
    float metallic  = pushData.params.x > 0.5 ? 0.78 : 0.02;
    float roughness = pushData.params.x > 0.5 ? 0.28 : 0.72;
    float luma = dot(albedo, vec3(0.2126, 0.7152, 0.0722));
    roughness = clamp(roughness + (1.0 - luma) * 0.06, 0.04, 1.0);

    // Lighting vectors
    // The game camera looks down the negative Z axis; this stable view vector
    // avoids requiring another matrix in the shared push-constant block.
    vec3 V = safeNormalize(vec3(0.0, 0.42, 0.91), N);
    vec3 L = SUN_DIR;
    vec3 H = safeNormalize(V + L, N);

    float NdotL = max(dot(N, L), 0.0);
    float NdotV = max(dot(N, V), 0.001);
    float NdotH = max(dot(N, H), 0.0);
    float VdotH = max(dot(V, H), 0.0);

    // Specular (Cook-Torrance)
    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    vec3 F  = F_Schlick(VdotH, F0);
    float D = D_GGX(NdotH, roughness);
    float G = G_Smith(NdotV, NdotL, roughness);
    vec3 specular = (D * G * F) / max(4.0 * NdotV * NdotL, 1e-4);

    // Diffuse (energy-conserving Lambertian)
    vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);
    vec3 diffuse = kD * albedo * INV_PI;

    // Direct + ambient
    vec3 direct  = (diffuse + specular) * SUN_COLOR * SUN_INTENSITY * NdotL;
    vec3 ambient = ambientIrradiance(N) * albedo * (1.0 - metallic * 0.55);
    // Small specular ambient for metals
    ambient += ambientIrradiance(reflect(-V, N)) * F0 * (1.0 - roughness) * 0.18;

    vec3 color = direct + ambient;

    // Exposure + tone map
    color *= 0.90;
    color = ACESFilm(color);

    outColor = vec4(color, base.a);
}
