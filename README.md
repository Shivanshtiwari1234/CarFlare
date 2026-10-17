CarFlare — Endless Drive update

Replace your existing main.cpp with this main.cpp and replace:
  shaders/cube.vert
  shaders/cube.frag

CMakeLists.txt can remain unchanged.

Build from the project root:
  cmake -S . -B build -G Ninja
  cmake --build build
  ./build/CarFlare

Controls:
  W = accelerate
  S = brake / slow down
  A = steer left
  D = steer right
  Esc = close the window

This keeps the existing Vulkan setup and uses colored cube meshes for the car,
road, lane markers, and roadside posts. The lane markers and posts loop to
simulate an endless road.
## Textured OBJ car

The renderer loads `assets/car.obj` UV coordinates and samples `assets/colormap.png` through a Vulkan combined-image sampler. `assets/race.mtl` references that image as `map_Kd colormap.png` (the loader uses the corresponding PNG path directly). PNG decoding uses libpng, so install the libpng development package and make sure CMake can find it.

The car texture is loaded from a path relative to the process working directory. Run CarFlare from the project root so `assets/car.obj` and `assets/colormap.png` resolve correctly.
