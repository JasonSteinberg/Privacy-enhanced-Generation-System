package pegs

import (
	"fmt"
	"math/rand"
	"testing"
	"time"
)

const (
	NumRows     = 10000000
	NumCols     = 10
	Cardinality = 5
)

// TestHarness_10M_Generation verifies the high-throughput generation pipeline.
func TestHarness_10M_Generation(t *testing.T) {
	fmt.Printf("[TestHarness] Initializing Contiguous Memory Buffers for %d rows...\n", NumRows)

	// Pre-allocate flat contiguous buffer to bypass GC allocation overhead
	dataBuffer := make([]uint16, NumRows*NumCols)

	// WARNING: Simulated data generation for testing purposes
	rng := rand.New(rand.NewSource(42))
	for i := 0; i < len(dataBuffer); i++ {
		rowIdx := i / NumCols
		if i%NumCols == 0 {
			dataBuffer[i] = uint16(rowIdx % Cardinality)
		} else {
			prevVal := dataBuffer[i-1]
			if rng.Float64() < 0.85 {
				dataBuffer[i] = prevVal
			} else {
				dataBuffer[i] = uint16(rng.Intn(Cardinality))
			}
		}
	}

	fmt.Println("[TestHarness] Data allocation complete. Launching concurrent worker pool...")
	startTime := time.Now()

	numWorkers := 8
	synthBuffer := CoordinateParaPeGS(dataBuffer, NumRows, NumCols, Cardinality, numWorkers, 1.5, 0.1)

	duration := time.Since(startTime)
	throughput := float64(NumRows) / duration.Seconds()

	fmt.Printf("[TestHarness] Generation complete. Processed 10M rows in %.2f seconds.\n", duration.Seconds())
	fmt.Printf("[TestHarness] Pipeline throughput: %.2f rows/sec\n", throughput)

	// Run empirical utility checks
	VerifyUtilityAndPrivacy(synthBuffer, dataBuffer, NumRows, NumCols, Cardinality)
}
