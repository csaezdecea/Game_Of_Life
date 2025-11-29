// main.cpp
// Split version of the Game of Life viewer (main program)
// Compile with:
// nvcc main.cpp cuda_kernels.cu -o gol -lglfw -lGLEW -lGL -ldl -lpthread

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <iostream>
#include <algorithm>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <thread>
#include <chrono>

// CUDA prototypes
extern "C" void upload_glider_templates();
extern "C" void initialize_simulation_cu(bool randomize);
extern "C" void cleanup_simulation_cu();
extern "C" void step_simulation_cu(cudaStream_t stream);
extern "C" void run_detection_cu(cudaStream_t stream);
extern "C" void write_rgb_from_grid_cu(unsigned char* pboDevPtr, int rows, int cols,
                                       unsigned char overlay_flag, cudaStream_t stream);

#define CHECK_CUDA(call) do {                                 \
    cudaError_t e = (call);                                   \
    if (e != cudaSuccess) {                                   \
        fprintf(stderr, "CUDA Error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
} while(0)

static const int ROWS = 1024;
static const int COLS = 1024;

// GL/CUDA variables
static struct cudaGraphicsResource* cudaPBOResource = nullptr;
static GLuint glTexture = 0;
static GLuint glPBO = 0;

// Simulation state
static bool is_running = true;
static bool overlay_detection = true;

// Camera
float previousZoom = 1.0f; 
static float cameraZoom = 1.0f;
static float cameraOffsetX = 0.0f;
static float cameraOffsetY = 0.0f;
static bool rightMouseDown = false;
static double lastMouseX = 0.0, lastMouseY = 0.0;

// Timing
static double lastFrameTime = 0.0;
static double targetFPS = 10.0;



/* Clamp camera so visible area (taking zoom into account) remains fully inside the texture [0..COLS] x [0..ROWS]. 
Visible width in texture units = COLS / cameraZoom, Visible height = ROWS / cameraZoom. 
So allowed offsets: 0 .. COLS - visibleWidth, etc. */
void clampCameraOffsets() {
    // Visible region size in texture (world) coordinates
    float visibleWidth  = (float)COLS / cameraZoom;
    float visibleHeight = (float)ROWS / cameraZoom;

    // Maximum allowed top-left coordinates so the visible rect fits inside the grid
    float maxX = (float)COLS - visibleWidth;
    float maxY = (float)ROWS - visibleHeight;

    if (maxX < 0.0f) maxX = 0.0f; // view larger than grid -> lock to 0
    if (maxY < 0.0f) maxY = 0.0f;

    if (cameraOffsetX < 0.0f) cameraOffsetX = 0.0f;
    if (cameraOffsetY < 0.0f) cameraOffsetY = 0.0f;

    if (cameraOffsetX > maxX) cameraOffsetX = maxX;
    if (cameraOffsetY > maxY) cameraOffsetY = maxY;
}


/* Mouse movement (right button drag) */
void cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
    if (rightMouseDown) {
        int w, h; 
        glfwGetFramebufferSize(window, &w, &h);

        // ---- PUT IT HERE ----
        float minZoom = std::min(
            (float)w / COLS,
            (float)w / ROWS
        );
        cameraZoom = std::max(cameraZoom, minZoom);
        // ----------------------

        double dx = xpos - lastMouseX;
        double dy = ypos - lastMouseY;

        // Translate mouse movement (screen) to texture pixels, taking zoom into account.
        // The factors (COLS / w) and (ROWS / h) convert screen pixels to texture units.
        cameraOffsetX -= (float)(dx * (COLS / (double)w)) / cameraZoom;
        cameraOffsetY -= (float)(dy * (ROWS / (double)h)) / cameraZoom;

        clampCameraOffsets();
    }
    lastMouseX = xpos;
    lastMouseY = ypos;
}


void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        rightMouseDown = (action == GLFW_PRESS);
        if (rightMouseDown) glfwGetCursorPos(window, &lastMouseX, &lastMouseY);
    }
}


void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    if (yoffset > 0) cameraZoom *= 1.1f;
    else if (yoffset < 0) cameraZoom /= 1.1f;

    cameraZoom = std::clamp(cameraZoom, 0.02f, 40.0f);

    // After zoom, clamp to keep view inside grid
    clampCameraOffsets();
}


void ZommOutKey(GLFWwindow* window) {
     float oldZoom = cameraZoom;
    cameraZoom = std::max(cameraZoom / 1.1f, 0.02f);  // zoom out
    // Only adjust window if zoom actually decreased
    if (cameraZoom < oldZoom)
    {
        int newW = (int)(COLS * cameraZoom);
        int newH = (int)(ROWS * cameraZoom);

        newW = std::max(1, newW);
        newH = std::max(1, newH);

        // Get current window size
        int ww, wh;
        glfwGetWindowSize(window, &ww, &wh);

        // If simulation area < current screen → shrink window
        if (newW < ww || newH < wh)
            glfwSetWindowSize(window, newW, newH);
    }

}


/* KEYBOARD – including fixed arrow camera movement */
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (action == GLFW_PRESS || action == GLFW_REPEAT)
    {
        const float panSpeed = 30.0f / cameraZoom;

        switch (key) {

            case GLFW_KEY_SPACE:
                is_running = !is_running;
                break;

            case GLFW_KEY_PERIOD:
                if (!is_running) {
                    cudaStream_t s = 0;
                    step_simulation_cu(s);
                    if (overlay_detection) run_detection_cu(s);
                }
                break;

            case GLFW_KEY_D: overlay_detection = !overlay_detection; break;
            case GLFW_KEY_C: initialize_simulation_cu(false); break;
            case GLFW_KEY_R: initialize_simulation_cu(true); break;
            case GLFW_KEY_ESCAPE: glfwSetWindowShouldClose(window, GLFW_TRUE); break;

            // panning
            case GLFW_KEY_LEFT:  cameraOffsetX -= panSpeed; break;
            case GLFW_KEY_RIGHT: cameraOffsetX += panSpeed; break;
            case GLFW_KEY_UP:    cameraOffsetY -= panSpeed; break;
            case GLFW_KEY_DOWN:  cameraOffsetY += panSpeed; break;

            // zoom
            case GLFW_KEY_Z: cameraZoom *= 1.1f; break;  // zoom IN
            case GLFW_KEY_X: ZommOutKey(window); break;  // zoom OUT
            //case GLFW_KEY_X: cameraZoom /= 1.1f; break;  // zoom OUT
            
            case GLFW_KEY_MINUS:
                targetFPS = std::max(targetFPS / 1.5, 1.0);
                printf("Target FPS: %.2f\n", targetFPS);
                break;

            case GLFW_KEY_EQUAL:
                targetFPS = std::min(targetFPS * 1.5, 240.0);
                printf("Target FPS: %.2f\n", targetFPS);
                break;
        }

        clampCameraOffsets();
    }
}






/*Creates the OpenGL texture and Pixel Buffer Object (PBO) that will store pixel data on the 
GPU. It initializes the PBO with the correct size and format, then registers it with CUDA so 
CUDA kernels can write directly into it. This function sets up all GPU resources required for
CUDA–OpenGL interop.*/
GLuint create_texture_and_pbo_register(int width, int height) {

    glGenTextures(1, &glTexture);
    glBindTexture(GL_TEXTURE_2D, glTexture);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    // clamp to edge to avoid sampling outside the texture
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, width, height,
                 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);

    glBindTexture(GL_TEXTURE_2D, 0);

    glGenBuffers(1, &glPBO);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, glPBO);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, (size_t)width * (size_t)height * 3 * sizeof(unsigned char), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    CHECK_CUDA(cudaGraphicsGLRegisterBuffer(
        &cudaPBOResource, glPBO, cudaGraphicsRegisterFlagsWriteDiscard));

    const GLubyte* renderer = glGetString(GL_RENDERER);
    const GLubyte* version  = glGetString(GL_VERSION);
    printf("OpenGL renderer: %s\n", renderer ? (const char*)renderer : "NULL");
    printf("OpenGL version:  %s\n", version ? (const char*)version : "NULL");

    return glTexture;
}


/*Maps the OpenGL Pixel Buffer Object so CUDA can access it, obtains a device pointer to the 
PBO, and runs CUDA kernels to write RGB pixel data into it. After synchronizing the CUDA stream,
it unmaps the resource and updates the OpenGL texture using the newly written PBO data.
This function performs the full CUDA→OpenGL interop step that transfers computed GPU pixels 
directly to the rendered texture without copying through the CPU. */
void update_texture_using_pbo_and_kernels(cudaStream_t stream) {
     // 1. Make the OpenGL PBO available to CUDA
    CHECK_CUDA(cudaGraphicsMapResources(1, &cudaPBOResource, 0));

    // 2. Get a raw CUDA device pointer to the mapped PBO memory
    void* pboDevPtr = nullptr;
    size_t pboSize = 0;
    CHECK_CUDA(cudaGraphicsResourceGetMappedPointer(&pboDevPtr, &pboSize, cudaPBOResource));

    // 3. Run CUDA kernels to fill the PBO with pixel data
    unsigned char overlay_flag = overlay_detection ? 1 : 0;
    write_rgb_from_grid_cu((unsigned char*)pboDevPtr, ROWS, COLS, overlay_flag, stream);

    // 4. Wait for CUDA to finish writing before OpenGL uses the buffer
    CHECK_CUDA(cudaStreamSynchronize(stream));
    
    // 5. Unmap the resource so OpenGL can use it again
    CHECK_CUDA(cudaGraphicsUnmapResources(1, &cudaPBOResource, 0));

    // 6. Upload the PBO into the OpenGL texture
    glBindTexture(GL_TEXTURE_2D, glTexture);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, glPBO);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1); // ensure no alignment padding issues
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, COLS, ROWS, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4); // optional: restore default if you like
   
     // 7. Cleanup / restore OpenGL bindings
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
}


void render_texture_with_camera() {

    glClear(GL_COLOR_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0, COLS, ROWS, 0, -1, 1);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glScalef(cameraZoom, cameraZoom, 1.0f);
    glTranslatef(-cameraOffsetX, -cameraOffsetY, 0.0f);

    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, glTexture);

    glBegin(GL_QUADS);
        glTexCoord2f(0, 0); glVertex2f(0, 0);
        glTexCoord2f(1, 0); glVertex2f(COLS, 0);
        glTexCoord2f(1, 1); glVertex2f(COLS, ROWS);
        glTexCoord2f(0, 1); glVertex2f(0, ROWS);
    glEnd();

    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_TEXTURE_2D);

    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
}

// ------------------------------------------------------
int main() {

    if (!glfwInit()) return -1;

    GLFWwindow* window = glfwCreateWindow(COLS/2, ROWS/2,
        "Game of Life CUDA", NULL, NULL);

    if (!window) { fprintf(stderr, "Failed to create window\n"); return -1; }

    glfwMakeContextCurrent(window);
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) { fprintf(stderr, "Failed to init GLEW\n"); return -1; }
    //Prompt Info!    
    

     // --- CUDA DEVICE CHECK ---
    int devCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&devCount));

    if (devCount == 0) {
        fprintf(stderr, "No CUDA devices found\n");
        return -1;
    }

    // Select device 0 (safe now)
    CHECK_CUDA(cudaSetDevice(0));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("Using GPU 0: %s (%d SMs)\n", prop.name, prop.multiProcessorCount);

    upload_glider_templates();
    initialize_simulation_cu(false);
    create_texture_and_pbo_register(COLS, ROWS);
    glfwSetKeyCallback(window, key_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    lastFrameTime = glfwGetTime();

    // ensure initial offsets are valid
    clampCameraOffsets();
    
    while (!glfwWindowShouldClose(window)) {

        glfwPollEvents();
        if (is_running) {
            step_simulation_cu(stream);
            if (overlay_detection) run_detection_cu(stream);
        }

        update_texture_using_pbo_and_kernels(stream);

        // ----------------------- ADD HERE -----------------------
        double now = glfwGetTime();
        double dt  = now - lastFrameTime;
        double targetDt = 1.0 / targetFPS;

        if (dt < targetDt) {
            std::this_thread::sleep_for(
                std::chrono::duration<double>(targetDt - dt)
            );
        }

        lastFrameTime = glfwGetTime();
        // --------------------------------------------------------

        int ww, wh;
        glfwGetFramebufferSize(window, &ww, &wh);

        clampCameraOffsets();

        //
        // Render
        //
        glViewport(0, 0, ww, wh);
        render_texture_with_camera();
        glfwSwapBuffers(window);
    }

    CHECK_CUDA(cudaStreamDestroy(stream));
    if (cudaPBOResource) cudaGraphicsUnregisterResource(cudaPBOResource);
    if (glPBO) glDeleteBuffers(1, &glPBO);
    if (glTexture) glDeleteTextures(1, &glTexture);

    cleanup_simulation_cu();
    glfwTerminate();
    return 0;
}
