#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <curand_kernel.h>

/**
 * Enhanced CUDA LSH Kernel for matching similar patient records.
 */

static uint16_t* d_dataset = NULL;
static int datasetRows = 0;
static int datasetCols = 0;

extern "C" {

int cuda_set_dataset(const uint16_t* h_data, int numRows, int numCols) {
    if (d_dataset) cudaFree(d_dataset);
    datasetRows = numRows;
    datasetCols = numCols;
    cudaMalloc(&d_dataset, (size_t)numRows * numCols * sizeof(uint16_t));
    return cudaMemcpy(d_dataset, h_data, (size_t)numRows * numCols * sizeof(uint16_t), cudaMemcpyHostToDevice);
}

__global__ void lsh_match_kernel(
    const uint16_t* batchContexts,
    int batchSize,
    int contextLen,
    const uint16_t* dataset,
    int dRows,
    int32_t* neighborIndices,
    int32_t* neighborCounts,
    int maxNeighbors
) {
    int bIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bIdx >= batchSize) return;

    const uint16_t* query = &batchContexts[bIdx * contextLen];
    int count = 0;

    // Simple heuristic: search a random subset of the dataset to find neighbors.
    // In a real LSH, we would use hash buckets, but for this experiment,
    // a sampled search is often sufficient for empirical distribution estimation.
    curandState state;
    curand_init(1337, bIdx, 0, &state);

    for (int i = 0; i < 500; i++) { // Sample 500 random rows
        int targetRow = curand(&state) % dRows;
        const uint16_t* target = &dataset[targetRow * contextLen];

        // Compute Manhattan distance
        int dist = 0;
        for (int c = 0; c < contextLen; c++) {
            int d = (int)query[c] - (int)target[c];
            dist += (d < 0) ? -d : d;
            if (dist > 5) break; // Early exit for dissimilar rows
        }

        if (dist <= 5) {
            if (count < maxNeighbors) {
                neighborIndices[bIdx * maxNeighbors + count] = targetRow;
                count++;
            }
        }
    }
    neighborCounts[bIdx] = count;
}

int cuda_lsh_match(
    uint16_t* batchContexts,
    int batchSize,
    int contextLen,
    int32_t* neighborIndices,
    int32_t* neighborCounts,
    int maxNeighbors
) {
    if (!d_dataset) return -1;

    uint16_t *d_batch;
    int32_t *d_indices, *d_counts;

    cudaMalloc(&d_batch, batchSize * contextLen * sizeof(uint16_t));
    cudaMalloc(&d_indices, batchSize * maxNeighbors * sizeof(int32_t));
    cudaMalloc(&d_counts, batchSize * sizeof(int32_t));

    cudaMemcpy(d_batch, batchContexts, batchSize * contextLen * sizeof(uint16_t), cudaMemcpyHostToDevice);

    int threadsPerBlock = 128;
    int blocksPerGrid = (batchSize + threadsPerBlock - 1) / threadsPerBlock;
    lsh_match_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_batch, batchSize, contextLen, d_dataset, datasetRows, d_indices, d_counts, maxNeighbors
    );

    cudaMemcpy(neighborIndices, d_indices, batchSize * maxNeighbors * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(neighborCounts, d_counts, batchSize * sizeof(int32_t), cudaMemcpyDeviceToHost);

    cudaFree(d_batch);
    cudaFree(d_indices);
    cudaFree(d_counts);

    return 0;
}

}
