#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cmath>

// Add this after includes for easy error checking
#define CHECK_CUDA_CU(call) do {                             \
    cudaError_t e = (call);                                  \
    if (e != cudaSuccess) {                                  \
        fprintf(stderr, "CUDA Error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE);                                  \
    }                                                        \
} while(0)

#define ROWS 1024
#define COLS 1024
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

static unsigned char* d_grid = nullptr;
static unsigned char* d_grid_next = nullptr;
static unsigned char* d_detection = nullptr;

#define IDX(i,j) ((i)*COLS + (j))

__global__ void step_kernel(const unsigned char* grid, unsigned char* grid_next) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ROWS || j >= COLS) return;

    // Count live neighbors
    int count = 0;
    for (int di = -1; di <= 1; ++di)
        for (int dj = -1; dj <= 1; ++dj)
            if (!(di == 0 && dj == 0)) {
                int ni = (i + di + ROWS) % ROWS;
                int nj = (j + dj + COLS) % COLS;
                if (grid[IDX(ni,nj)] > 0) ++count;
            }

    unsigned char v = grid[IDX(i,j)];
    // Apply Game of Life rules
    if (v) {
        if (count == 2 || count == 3) grid_next[IDX(i,j)] = 255;
        else grid_next[IDX(i,j)] = 0;
    } else {
    if (count == 3) grid_next[IDX(i,j)] = 255;
    else grid_next[IDX(i,j)] = 0;
    }
}

__global__ void detection_kernel(const unsigned char* grid, unsigned char* detection) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ROWS || j >= COLS) return;
    if (grid[IDX(i,j)] > 0) detection[IDX(i,j)] = 1;
    else detection[IDX(i,j)] = 0;
}

__global__ void write_rgb_kernel(const unsigned char* grid, const unsigned char* detection, unsigned char* rgb, unsigned char overlay) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ROWS || j >= COLS) return;

    int idx = IDX(i,j);
    int rgb_idx = 3*idx;

    unsigned char cell = grid[idx]; // 0 or 255

    if (overlay && detection[idx] > 0) {
        // Example overlay: detected cells drawn white
        rgb[rgb_idx + 0] = 255;
        rgb[rgb_idx + 1] = 255;
        rgb[rgb_idx + 2] = 255;
    } else {
        // monochrome white/black based on cell
        rgb[rgb_idx + 0] = cell;
        rgb[rgb_idx + 1] = cell;
        rgb[rgb_idx + 2] = cell;
    }
}


// -----------------------------------------------------------------------------
// CUDA buffer allocation
// -----------------------------------------------------------------------------
void allocate_simulation_buffers() {
    size_t size = (size_t)ROWS * (size_t)COLS * sizeof(unsigned char);

    if (!d_grid) {
        CHECK_CUDA_CU(cudaMalloc(&d_grid, size));
        CHECK_CUDA_CU(cudaMemset(d_grid, 0, size));
    }
    if (!d_grid_next) {
        CHECK_CUDA_CU(cudaMalloc(&d_grid_next, size));
        CHECK_CUDA_CU(cudaMemset(d_grid_next, 0, size));
    }
    if (!d_detection) {
        CHECK_CUDA_CU(cudaMalloc(&d_detection, size));
        CHECK_CUDA_CU(cudaMemset(d_detection, 0, size));
    }
}

// -----------------------------------------------------------------------------
// Random initialization
// -----------------------------------------------------------------------------
void init_random_grid(unsigned char* g) {
    for (int i = 0; i < ROWS * COLS; ++i)
        g[i] = (rand() % 2) * 255;
}

// -----------------------------------------------------------------------------
// Generic pattern writer
// -----------------------------------------------------------------------------
void set_cells(unsigned char* g,
               const int pattern[][2],
               int count,
               int cx, int cy)
{
    for (int k = 0; k < count; ++k) {
        int x = cx + pattern[k][0];
        int y = cy + pattern[k][1];
        // wrap around (or clamp) — here we wrap to keep patterns within the toroidal grid
        x = (x % COLS + COLS) % COLS;
        y = (y % ROWS + ROWS) % ROWS;
        g[IDX(y, x)] = 255;
    }
}

// -----------------------------------------------------------------------------
// Pattern implementations
// -----------------------------------------------------------------------------
void add_glider(unsigned char* g, int cx, int cy) {
    const int glider[][2] = {
        {0,1},{1,2},{2,0},{2,1},{2,2}
    };
    set_cells(g, glider, 5, cx, cy);
}

void add_lwss(unsigned char* g, int cx, int cy) {
    const int lwss[][2] = {
        {1,0},{2,0},{3,0},{4,0},
        {0,1},{4,1},{4,2},
        {0,3},{3,3}
    };
    set_cells(g, lwss, 9, cx, cy);
}

void add_blinker(unsigned char* g, int cx, int cy) {
    const int blinker[][2] = {{0,0},{1,0},{2,0}};
    set_cells(g, blinker, 3, cx, cy);
}

void add_toad(unsigned char* g, int cx, int cy) {
    const int toad[][2] = {
        {1,0},{2,0},{3,0},
        {0,1},{1,1},{2,1}
    };
    set_cells(g, toad, 6, cx, cy);
}

void add_beacon(unsigned char* g, int cx, int cy) {
    const int beacon[][2] = {
        {0,0},{1,0},{0,1},{1,1},
        {2,2},{3,2},{2,3},{3,3}
    };
    set_cells(g, beacon, 8, cx, cy);
}

void add_block(unsigned char* g, int cx, int cy) {
    const int block[][2] = {{0,0},{1,0},{0,1},{1,1}};
    set_cells(g, block, 4, cx, cy);
}

void add_beehive(unsigned char* g, int cx, int cy) {
    const int beehive[][2] = {
        {1,0},{2,0},
        {0,1},{3,1},
        {1,2},{2,2}
    };
    set_cells(g, beehive, 6, cx, cy);
}

// Pulsar must stay custom because it is symmetric and large
void add_pulsar(unsigned char* g, int cx, int cy) {
    int offsets[] = {2,3,4,8,9,10};
    for (int a = 0; a < (int)(sizeof(offsets)/sizeof(offsets[0])); ++a) {
        int dx = offsets[a];
        for (int b = 0; b < (int)(sizeof(offsets)/sizeof(offsets[0])); ++b) {
            int dy = offsets[b];
            if ((dx<5 && dy<5) || (dx>5 && dy>5)) continue;
            // compute wrapped coords for each of the 4 symmetric points
            int x1 = (cx - dx) % COLS; if (x1 < 0) x1 += COLS;
            int y1 = (cy - dy) % ROWS; if (y1 < 0) y1 += ROWS;
            int x2 = (cx - dx) % COLS; if (x2 < 0) x2 += COLS;
            int y2 = (cy + dy) % ROWS; if (y2 < 0) y2 += ROWS;
            int x3 = (cx + dx) % COLS; if (x3 < 0) x3 += COLS;
            int y3 = (cy - dy) % ROWS; if (y3 < 0) y3 += ROWS;
            int x4 = (cx + dx) % COLS; if (x4 < 0) x4 += COLS;
            int y4 = (cy + dy) % ROWS; if (y4 < 0) y4 += ROWS;

            g[IDX(y1, x1)] = 255;
            g[IDX(y2, x2)] = 255;
            g[IDX(y3, x3)] = 255;
            g[IDX(y4, x4)] = 255;
        }
    }
}

// -----------------------------------------------------------------------------
// Initialize all patterns
// -----------------------------------------------------------------------------
void init_patterns(unsigned char* g) {
    memset(g, 0, ROWS * COLS);

    add_pulsar (g, COLS/6, ROWS/6);
    add_glider (g, 5*COLS/6, ROWS/6);
    add_lwss   (g, COLS/6, ROWS/3);
    add_blinker(g, 5*COLS/6, ROWS/2);
    add_toad   (g, COLS/6, 5*ROWS/6);
    add_beacon (g, 5*COLS/6, 5*ROWS/6);
    add_block  (g, COLS/3, 5*ROWS/6);
    add_beehive(g, COLS/3, ROWS/6);
}

// -----------------------------------------------------------------------------
// Main public initializer
// -----------------------------------------------------------------------------
extern "C" void initialize_simulation_cu(bool randomize) {
    allocate_simulation_buffers();

    size_t size = (size_t)ROWS * (size_t)COLS * sizeof(unsigned char);
    unsigned char* h_grid = new unsigned char[ROWS * COLS];

    if (randomize) {
        for (int i = 0; i < ROWS * COLS; ++i)
            h_grid[i] = (rand() % 2) ? 255 : 0;
    } else {
        init_patterns(h_grid);
    }

    // copy initial grid to device
    CHECK_CUDA_CU(cudaMemcpy(d_grid, h_grid, size, cudaMemcpyHostToDevice));

    // zero the next and detection buffers on device
    CHECK_CUDA_CU(cudaMemset(d_grid_next, 0, size));
    CHECK_CUDA_CU(cudaMemset(d_detection, 0, size));

    delete[] h_grid;
}


extern "C" void cleanup_simulation_cu() {
    if (d_grid)        { CHECK_CUDA_CU(cudaFree(d_grid));       d_grid = nullptr; }
    if (d_grid_next)   { CHECK_CUDA_CU(cudaFree(d_grid_next));  d_grid_next = nullptr; }
    if (d_detection)   { CHECK_CUDA_CU(cudaFree(d_detection));  d_detection = nullptr; }
}

extern "C" void step_simulation_cu(cudaStream_t stream) {
    dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid((COLS + block.x - 1)/block.x, (ROWS + block.y - 1)/block.y);

    // Launch kernel to compute grid_next from grid
    step_kernel<<<grid, block, 0, stream>>>(d_grid, d_grid_next);
    CHECK_CUDA_CU(cudaGetLastError());

    // Swap pointers (fast)
    unsigned char* tmp = d_grid;
    d_grid = d_grid_next;
    d_grid_next = tmp;
}

extern "C" void run_detection_cu(cudaStream_t stream) {
    dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid((COLS + block.x - 1)/block.x, (ROWS + block.y - 1)/block.y);
    detection_kernel<<<grid, block, 0, stream>>>(d_grid, d_detection);
}

extern "C" void write_rgb_from_grid_cu(unsigned char* pboDevPtr, int rows, int cols, unsigned char overlay_flag, cudaStream_t stream) {
    dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid((cols + block.x - 1)/block.x, (rows + block.y - 1)/block.y);

    write_rgb_kernel<<<grid, block, 0, stream>>>(d_grid, d_detection, pboDevPtr, overlay_flag);
    CHECK_CUDA_CU(cudaGetLastError());
}

extern "C" void upload_glider_templates() { /* not used */ }
