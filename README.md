# CUDA Game of Life Viewer (CUDA + OpenGL Interop)

This project implements a high‑performance **Conway’s Game of Life** simulator using:

- **CUDA kernels** for the simulation step  
- **CUDA + OpenGL Interoperability** for real‑time rendering  
- **OpenGL** for visualization and zoom/pan camera  
- **GLFW** + **GLEW** for the windowing and graphics context  

It is designed for **large grids** (e.g., 1024×1024) with real-time interaction, camera movement, zooming, and GPU‑accelerated overlay/detection features.

---

# Table of Contents
1. [Overview](#overview)
2. [Main Code Structure](#main-code-structure)
3. [Simulation Kernels](#simulation-kernels)
4. [CUDA/OpenGL Interop Explained](#cudaopengl-interop-explained)
5. [Main Snippets](#main-snippets)
6. [Build Instructions](#build-instructions)
7. [Controls](#controls)

---

# Overview

The application simulates Conway’s Game of Life entirely on the GPU.  
Each frame:

1. **CUDA kernel updates the grid**  
2. **(Optional) CUDA kernel detects patterns / overlays colors**  
3. **A CUDA kernel writes RGB pixels into an OpenGL PBO**  
4. **OpenGL uploads the PBO into a texture using `glTexSubImage2D`**  
5. **The texture is rendered to the screen**  
6. **User camera input adjusts zoom and panning**

This avoids expensive CPU–GPU transfers and enables real‑time performance at large resolutions.

---

# Main Code Structure

### Files

| File | Purpose |
|------|---------|
| `main.cpp` | OpenGL setup, camera, input, CUDA/GL interop, main loop |
| `cuda_kernels.cu` | Simulation kernels, detection kernel, RGB conversion kernel |
| `shaders/` (optional) | Any GLSL shaders if the project is extended |

---

# Simulation Kernels

The simulation logic is implemented entirely in CUDA inside `cuda_kernels.cu`.

### Core Life Kernel (Conceptual)

```cpp
__global__ void step_kernel(unsigned char* current, unsigned char* next,
                            int rows, int cols)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= cols || y >= rows) return;

    int idx = y * cols + x;

    int count = 0;
    // Example neighbor loop
    for (int dy = -1; dy <= 1; dy++)
    for (int dx = -1; dx <= 1; dx++) {
        if (dx == 0 && dy == 0) continue;
        int nx = x + dx;
        int ny = y + dy;
        if (0 <= nx && nx < cols && 0 <= ny && ny < rows)
            count += (current[ny * cols + nx] > 0);
    }

    unsigned char v = current[idx];
    next[idx] = (v ? (count == 2 || count == 3) : (count == 3)) ? 255 : 0;
}
```

### Detection Kernel (Example)

This kernel adds overlay colors depending on pattern detections.

```cpp
__global__ void detection_kernel(unsigned char* grid,
                                 unsigned char* overlay,
                                 int rows, int cols)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= cols || y >= rows) return;

    int idx = y * cols + x;

    if (grid[idx] == 255) {
        overlay[idx] = 128; // mark something
    } else {
        overlay[idx] = 0;
    }
}
```

### RGB Writing Kernel (CUDA → OpenGL)

This kernel takes the simulation grid and writes RGB pixels into the OpenGL PBO:

```cpp
__global__ void write_rgb_kernel(unsigned char* pbo,
                                 unsigned char* grid,
                                 unsigned char* overlay,
                                 int rows, int cols,
                                 bool overlay_flag)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= cols || y >= rows) return;

    int idx = y * cols + x;
    int p = 3 * idx;

    unsigned char g = grid[idx];
    if (overlay_flag)
        g = max(g, overlay[idx]);

    pbo[p + 0] = g;
    pbo[p + 1] = g;
    pbo[p + 2] = g;
}
```

---

# CUDA/OpenGL Interop Explained

This application uses **pixel buffer objects (PBOs)** combined with CUDA graphics resources to avoid CPU copies.

## How Interop Works

### 1. OpenGL allocates a Pixel Buffer Object (PBO)

The PBO is created and initialized inside `create_texture_and_pbo_register()`.  
This buffer will later be registered with CUDA so kernels can write texture data directly into it.

```cpp
// Create a Pixel Buffer Object (PBO)
glGenBuffers(1, &glPBO);
glBindBuffer(GL_PIXEL_UNPACK_BUFFER, glPBO);

// Allocate GPU memory for the PBO (RGB image: rows × cols × 3 bytes)
glBufferData(
    GL_PIXEL_UNPACK_BUFFER,
    rows * cols * 3,
    nullptr,                // no initial data
    GL_DYNAMIC_DRAW         // optimized for frequent updates
);
```
### 2. CUDA registers the PBO

Inside `create_texture_and_pbo_register()`, the Pixel Buffer Object (PBO) created by OpenGL is registered with CUDA: 

```cpp
cudaGraphicsGLRegisterBuffer(&cudaPBOResource, glPBO,
                             cudaGraphicsRegisterFlagsWriteDiscard);
```

Registering the PBO allows CUDA to map this OpenGL buffer, write into it directly, and then unmap it so OpenGL can use the updated pixel data efficiently—without copying through CPU memory.

---

## 3. Each frame, CUDA maps the buffer

Inside `update_texture_using_pbo_and_kernels()`, the PBO registered with CUDA is mapped so kernels can access it:


```cpp
cudaGraphicsMapResources(1, &cudaPBOResource);
cudaGraphicsResourceGetMappedPointer(&pboDevPtr, &pboSize, cudaPBOResource);
```

After this call, `pboDevPtr` becomes a raw **device pointer** that refers to the memory inside the OpenGL Pixel Buffer Object.
Your CUDA kernels can now write pixel data directly into the PBO, avoiding any CPU-side copies.

---

## 4. CUDA Writes Pixels Into the Mapped PBO

This step also occurs inside `update_texture_using_pbo_and_kernels()`:

```cpp
write_rgb_from_grid_cu(pboDevPtr, ROWS, COLS, pitch, overlay_flag, stream);
```

At this stage:

* pboDevPtr is a device pointer to the mapped OpenGL PBO.
* CUDA kernels write pixel data directly into this buffer.
* No copies to the CPU are needed — everything stays on the GPU.

This is the key benefit of CUDA–OpenGL interoperability:
CUDA generates the image, and OpenGL displays it, without any GPU–CPU transfer.


## 5. CUDA Unmaps the PBO (Making It Usable by OpenGL Again)

Still inside `update_texture_using_pbo_and_kernels()`, after the kernels finish writing into the PBO:

```cpp
cudaGraphicsUnmapResources(1, &cudaPBOResource);
```

Unmapping performs the following:

* Synchronizes CUDA and OpenGL so OpenGL can safely use the updated pixel data.

* Releases CUDA’s access to the mapped PBO.

* Makes the PBO available for OpenGL operations (such as uploading it to a texture).

After this unmap, OpenGL can immediately read the PBO contents and display the image. This ensures a seamless pipeline:
 
 CUDA writes → unmap → OpenGL renders
all without moving data through the CPU.

---

## 6. OpenGL Uploads the PBO Data Into the Texture

After CUDA unmaps the PBO, OpenGL can safely read its contents.  
This also happens inside `update_texture_using_pbo_and_kernels()`:

```cpp
glBindTexture(GL_TEXTURE_2D, glTexID);

glBindBuffer(GL_PIXEL_UNPACK_BUFFER, glPBO);
glTexSubImage2D(
    GL_TEXTURE_2D,
    0,              // mipmap level
    0, 0,           // x, y offset
    COLS, ROWS,     // width, height
    GL_RGB,         // pixel format
    GL_UNSIGNED_BYTE,
    nullptr         // data comes from the currently bound PBO
);
```

Explanation:

* glBindBuffer(GL_PIXEL_UNPACK_BUFFER, glPBO);
Tells OpenGL that pixel data should come from the PBO, not from CPU memory.
* glTexSubImage2D(..., nullptr);
Passing nullptr means:
“Read pixel data from the bound PBO.”

This operation copies the pixel data (written by CUDA) into the OpenGL texture.

Once the texture is updated, the rendering code (`render_texture_with_camera()`) draws it on the screen.

---

# Main Snippets

### Device Count Check

```cpp
int devCount = 0;
CHECK_CUDA(cudaGetDeviceCount(&devCount));
if (devCount == 0) {
    fprintf(stderr, "No CUDA devices found
");
    return -1;
}
CHECK_CUDA(cudaSetDevice(0));
```

### Simulation Step

```cpp
if (is_running) {
    step_simulation_cu(stream);
    if (overlay_detection)
        run_detection_cu(stream);
}
```

### Texture Update

```cpp
update_texture_using_pbo_and_kernels(stream);
```

### Camera Movement

Inside the `clampCameraOffsets()` function, camera panning is scaled so that
movement remains consistent regardless of zoom level or window size:

```cpp
cameraOffsetX -= dx * (COLS / w) / cameraZoom;
cameraOffsetY -= dy * (ROWS / h) / cameraZoom;
```

Explanation:

* dx and dy are the mouse drag distances in screen pixels.
* (COLS / w) and (ROWS / h) convert these pixel movements into
simulation-space units, based on the size of the compute grid.
* Dividing by cameraZoom ensures that panning speed feels the same
whether the user is zoomed in or zoomed out.

This allows the camera to move intuitively and proportionally over the simulated texture.


### Zoom

Zooming is handled inside `scroll_callback()`:

```cpp
if (yoffset > 0)
    cameraZoom *= 1.1f;   // zoom in
else
    cameraZoom /= 1.1f;   // zoom out
```

Explanation:

* yoffset is the scroll direction provided by GLFW.
* A factor of 1.1 produces smooth, incremental zoom steps.
* Zooming in multiplies cameraZoom, making the simulation appear larger.
* Zooming out divides cameraZoom, showing a wider region of the simulation.

This keeps the zoom behavior simple, responsive, and frame-independent.


---

# Build Instructions

```bash
nvcc main.cpp cuda_kernels.cu -o gol     -lglfw -lGLEW -lGL -ldl -lpthread
```

or Simply:

```bash
make
```

Requires:

- NVIDIA GPU  
- CUDA Toolkit  
- GLEW  
- GLFW  
- OpenGL  

---

# Controls

| Key | Action |
|-----|--------|
| SPACE | Start/stop simulation |
| . (dot) | Step single frame |
| C | Pattern based grid* |
| R | Randomize grid |
| D | Toggle detection overlay |
| Arrow Keys | Pan camera |
| Z/X | Zoom in/out |
| Scroll | Smooth zoom |
| Right Mouse Drag | Pan |
| ESC | Exit |
---



*In the pattern based grid some known patters are shown for display among them are:  
The Bee-hive, The Block, The Pulsar (period 2), The Blinker, The Toad,  The Glider, and The Light-weight spaceship.



# Summary

This project demonstrates efficient real‑time GPU simulation using:

- CUDA compute kernels  
- CUDA/OpenGL PBO interop  
- Real‑time interactive visualization  

The architecture avoids CPU bottlenecks and is suitable for  
large-scale cellular automata and other grid-based GPU simulations.

