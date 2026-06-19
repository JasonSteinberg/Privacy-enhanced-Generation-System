# Compilation configuration for PeGS CUDA Engine
# Requires NVIDIA CUDA Toolkit (nvcc)

NVCC = nvcc
CFLAGS = -O3 --shared -Xcompiler -fPIC
TARGET = liblsh.so
SRC = lsh_kernel.cu

all: $(TARGET)

$(TARGET): $(SRC)
	@echo "Compiling CUDA LSH Engine..."
	$(NVCC) $(CFLAGS) -o $(TARGET) $(SRC)
	@echo "Build complete: $(TARGET)"

clean:
	rm -f $(TARGET)

.PHONY: all clean
