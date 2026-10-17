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