# Photon Stage 2

Stage 2 is the real-time part of the Photon rendering engine. It takes as an input a .photon file generated in Stage 1 and allows for rendering the scene it describes in real time.

Currently, this stage can be run using a command-line interface (see [main.cu](./main.cu)). Afterwards, an OpenGL window is created in [renderer.cu](./Renderer.cu) using SDL2; the scene is loaded and pushed to CUDA where it is rendered in [cudaRenderer.cu](./cudaRenderer.cu).
