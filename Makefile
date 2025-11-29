# ============================================
# Makefile for CUDA + OpenGL Game of Life
# ============================================

# Compiler settings
CXX := g++
NVCC := nvcc

# Flags
CXXFLAGS := -O2 -std=c++17
NVCCFLAGS := -O2 -arch=sm_70

# Include paths (if you have extra headers)
INCLUDES := -Iinclude

# Libraries
LIBS := -lglfw -lGLEW -lGL -ldl -lpthread

# Sources
SRC_CPP := src/main.cpp
SRC_CU  := src/cuda_kernels.cu

OBJ_CPP := $(SRC_CPP:.cpp=.o)
OBJ_CU  := $(SRC_CU:.cu=.o)

# Output binary
TARGET := visualizer

# Default target
all: $(TARGET)

# Compile C++ source
%.o: %.cpp
	$(NVCC) -c $(CXXFLAGS) $(INCLUDES) $< -o $@

# Compile CUDA source
%.o: %.cu
	$(NVCC) -c $(NVCCFLAGS) $(INCLUDES) $< -o $@

# Link everything
$(TARGET): $(OBJ_CPP) $(OBJ_CU)
	$(NVCC) $(OBJ_CPP) $(OBJ_CU) -o $(TARGET) $(LIBS)

# Clean build
clean:
	rm -f $(TARGET) src/*.o

# Phony targets
.PHONY: all clean
