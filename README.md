# CarFlare — Endless Drive

A small Vulkan driving game using GLFW, GLM, libpng, and GLSL shaders.

## Build

Dependencies: a C++20 compiler, CMake, Ninja (optional), Vulkan SDK/loader and headers, GLFW, GLM, libpng, and `glslc`.

From this directory:

```bash
cmake -S . -B build -G Ninja
cmake --build build
./build/CarFlare
```

Run from the project root so `assets/car.obj` and `assets/colormap.png` resolve.

## Controls

- **W** — accelerate
- **S** — brake or reverse
- **A / D** — steer
- **Esc** — close the window

Click **PLAY** in the start screen.

## Current driving and rendering systems

- Time-step-based acceleration, braking, rolling resistance, and speed-dependent aerodynamic drag.
- A bicycle-model-inspired steering response, yaw, lateral inertia, tyre-grip damping, and limited body roll/pitch cues.
- Reduced grip and speed on the roadside, plus a compliant boundary response at the guardrail.
- Directional sunlight, hemispheric ambient fill, diffuse shading, restrained specular highlights, and a mild tone-mapping shoulder in the fragment shader.
- UV-textured OBJ car with a built-in procedural fallback.

These are game-oriented approximations, not a validated automotive simulation. This revision does **not** yet implement shadow mapping, dynamic weather, physically based material maps, full suspension/wheel rotation, collision meshes, or a full rigid-body solver.

## Layout

- `main.cpp` — application entry point.
- `carflare.hpp` — public application interface.
- `carflare.cpp` — Vulkan renderer and driving implementation.
- `shaders/carflare.vert`, `shaders/carflare.frag` — vertex and fragment shaders.
- `assets/` — car mesh and texture.
