package pegs

import (
	"errors"
	"fmt"
	"runtime"
	"sync"

	"github.com/ebitengine/purego"
)

// CUDA_LSH_Match matches the signature of the exported C/CUDA function.
type CUDA_LSH_Match func(
	batchContexts *uint16,
	batchSize int,
	contextLen int,
	neighborIndices *int32,
	neighborCounts *int32,
	maxNeighbors int,
) int

var (
	mu           sync.RWMutex
	lshMatchFunc CUDA_LSH_Match
	setDatasetFunc func(data *uint16, numRows, numCols int) int
	libHandle    uintptr
)

// InitCUDAEngine dynamically binds the shared library symbols at runtime.
func InitCUDAEngine(soPath string) error {
	mu.Lock()
	defer mu.Unlock()

	if libHandle != 0 {
		return nil // Already initialized
	}

	handle, err := purego.Dlopen(soPath, purego.RTLD_NOW|purego.RTLD_GLOBAL)
	if err != nil {
		return fmt.Errorf("failed to load shared object %s: %w", soPath, err)
	}
	libHandle = handle

	purego.RegisterLibFunc(&lshMatchFunc, libHandle, "cuda_lsh_match")
	purego.RegisterLibFunc(&setDatasetFunc, libHandle, "cuda_set_dataset")

	if lshMatchFunc == nil || setDatasetFunc == nil {
		purego.Dlclose(libHandle)
		libHandle = 0
		return errors.New("failed to bind symbols: cuda_lsh_match or cuda_set_dataset")
	}

	return nil
}

// SetCUDAData uploads the reference dataset to the GPU.
func SetCUDAData(data []uint16, numRows, numCols int) error {
	mu.RLock()
	fn := setDatasetFunc
	mu.RUnlock()

	if fn == nil {
		return errors.New("CUDA engine not initialized")
	}

	if ret := fn(&data[0], numRows, numCols); ret != 0 {
		return fmt.Errorf("failed to upload dataset to GPU, exit code: %d", ret)
	}
	return nil
}

// CloseCUDAEngine releases the shared library handle.
func CloseCUDAEngine() error {
	mu.Lock()
	defer mu.Unlock()

	if libHandle == 0 {
		return nil
	}

	if err := purego.Dlclose(libHandle); err != nil {
		return fmt.Errorf("failed to close shared object: %w", err)
	}
	libHandle = 0
	lshMatchFunc = nil
	return nil
}

// FindBatchNeighbors passes pinned Go slices directly to the unmanaged CUDA driver.
func FindBatchNeighbors(contexts []uint16, batchSize, contextLen, maxNeighbors int) ([]int32, []int32, error) {
	mu.RLock()
	fn := lshMatchFunc
	mu.RUnlock()

	if fn == nil {
		return nil, nil, errors.New("CUDA engine not initialized")
	}

	// Pre-allocate output buffers
	indices := make([]int32, batchSize*maxNeighbors)
	counts := make([]int32, batchSize)

	// Lock execution to a single OS thread to satisfy CUDA context thread-affinity requirements
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	// Direct zero-copy pointer passing of contiguous flat memory arrays
	ret := fn(
		&contexts[0],
		batchSize,
		contextLen,
		&indices[0],
		&counts[0],
		maxNeighbors,
	)

	if ret != 0 {
		return nil, nil, fmt.Errorf("CUDA LSH kernel execution failed with exit code: %d", ret)
	}

	return indices, counts, nil
}
