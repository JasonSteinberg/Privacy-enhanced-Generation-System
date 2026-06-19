package main

import (
	"flag"
	"fmt"
	"math/rand"
	"os"
	"time"

	"deep/pkg/pegs"
)

func main() {
	var (
		numRows     = flag.Int("rows", 15000000, "Number of rows to generate")
		numCols     = flag.Int("cols", 350, "Number of columns per row")
		cardinality = flag.Int("card", 10, "Cardinality of categorical tokens (max deviation units)")
		numWorkers  = flag.Int("workers", 32, "Number of parallel worker goroutines")
		entropy     = flag.Float64("entropy", 2.0, "Target entropy for l-diversity")
		epsilon     = flag.Float64("epsilon", 0.5, "Epsilon for differential privacy")
		soPath      = flag.String("cuda-so", "./liblsh.so", "Path to CUDA LSH shared library")
		dataPath    = flag.String("data-path", "", "Path to real data (binary uint16 format). If empty, simulated data is used.")
		outputPath  = flag.String("output-path", "", "Path to save synthetic data (binary uint16 format).")
	)
	flag.Parse()

	fmt.Printf("[PeGS] Initializing CUDA Engine from %s...\n", *soPath)
	tCuda := time.Now()
	if err := pegs.InitCUDAEngine(*soPath); err != nil {
		fmt.Printf("\n[PeGS] ⚠️  CUDA LSH ENGINE NOT FOUND: %v\n", err)
		fmt.Printf("[PeGS] INFO: To enable hardware acceleration, ensure liblsh.so is compiled and in your path.\n")
		fmt.Printf("[PeGS] INFO: Falling back to CPU-based Heuristic Empirical Sampler.\n\n")
	} else {
		defer func() {
			if err := pegs.CloseCUDAEngine(); err != nil {
				fmt.Printf("[PeGS] Warning: Error closing CUDA engine: %v\n", err)
			}
		}()
		fmt.Printf("[PeGS] CUDA Initialization took %v\n", time.Since(tCuda))
	}

	fmt.Printf("[PeGS] Initializing Contiguous Memory Buffers for %d rows...\n", *numRows)
	tMem := time.Now()
	// Pre-allocate flat contiguous buffer to bypass GC allocation overhead
	bufferSize := int64(*numRows) * int64(*numCols) * 2
	dataBuffer := make([]uint16, (*numRows)*(*numCols))
	memDuration := time.Since(tMem)
	fmt.Printf("[PeGS] Memory allocation took %v (%.2f GB/s)\n", memDuration, float64(bufferSize)/memDuration.Seconds()/1e9)

	if *dataPath != "" {
		fmt.Printf("[PeGS] Loading real data from %s...\n", *dataPath)
		tLoad := time.Now()
		content, err := os.ReadFile(*dataPath)
		if err != nil {
			fmt.Printf("[PeGS] Error reading data file: %v. Falling back to simulation.\n", err)
			generateSimulatedData(dataBuffer, *numCols, *cardinality)
		} else {
			loadDuration := time.Since(tLoad)
			expectedBytes := (*numRows) * (*numCols) * 2
			if len(content) < expectedBytes {
				fmt.Printf("[PeGS] Warning: Data file smaller than expected (%d < %d bytes). Only partial data loaded.\n", len(content), expectedBytes)
			}
			// Use unsafe or just a better copy loop?
			// For Go 1.21, we can't easily use unsafe to cast []byte to []uint16 without more boilerplate.
			// But we can at least optimize the copy loop.
			n := len(dataBuffer)
			if len(content)/2 < n {
				n = len(content) / 2
			}
			for i := 0; i < n; i++ {
				dataBuffer[i] = uint16(content[i*2]) | (uint16(content[i*2+1]) << 8)
			}
			fmt.Printf("[PeGS] Data loading took %v (%.2f MB/s)\n", loadDuration, float64(len(content))/loadDuration.Seconds()/1e6)
		}
	} else {
		fmt.Println("[PeGS] WARNING: No data path provided. Using simulated data generation.")
		tSim := time.Now()
		generateSimulatedData(dataBuffer, *numCols, *cardinality)
		simDuration := time.Since(tSim)
		fmt.Printf("[PeGS] Simulated data generation took %v (%.2f M records/s)\n", simDuration, float64(*numRows)/simDuration.Seconds()/1e6)
	}

	fmt.Println("[PeGS] Uploading data to CUDA Engine...")
	tUpload := time.Now()
	if err := pegs.SetCUDAData(dataBuffer, *numRows, *numCols); err != nil {
		fmt.Printf("[PeGS] Warning: Failed to upload data to GPU: %v\n", err)
	} else {
		uploadDuration := time.Since(tUpload)
		fmt.Printf("[PeGS] GPU Upload took %v (%.2f GB/s)\n", uploadDuration, float64(bufferSize)/uploadDuration.Seconds()/1e9)
	}

	fmt.Println("[PeGS] Launching concurrent worker pool...")

	startTime := time.Now()

	synthBuffer := pegs.CoordinateParaPeGS(dataBuffer, *numRows, *numCols, *cardinality, *numWorkers, *entropy, *epsilon)

	duration := time.Since(startTime)
	throughput := float64(*numRows) / duration.Seconds()

	fmt.Printf("\n[PeGS] --- Performance Summary ---\n")
	fmt.Printf("[PeGS] Total generation time: %.2f seconds\n", duration.Seconds())
	fmt.Printf("[PeGS] Global pipeline throughput: %.2f rows/sec\n", throughput)

	lshSeconds := float64(pegs.TotalLSHNano) / 1e9
	sampleSeconds := float64(pegs.TotalSamplingNano) / 1e9

	// numGibbsIters = 2 is hardcoded in CoordinateParaPeGS
	totalLSHRecords := float64(*numRows) * 2
	totalTokens := totalLSHRecords * float64(*numCols)

	if lshSeconds > 0 {
		fmt.Printf("[PeGS] LSH Engine throughput: %.2f records/sec (Total: %.2f sec)\n", totalLSHRecords/lshSeconds, lshSeconds)
	}
	if sampleSeconds > 0 {
		fmt.Printf("[PeGS] Sampling throughput: %.2f tokens/sec (Total: %.2f sec)\n", totalTokens/sampleSeconds, sampleSeconds)
	}
	fmt.Printf("[PeGS] ---------------------------\n\n")

	// Run empirical utility checks
	tVerify := time.Now()
	pegs.VerifyUtilityAndPrivacy(synthBuffer, dataBuffer, *numRows, *numCols, *cardinality)
	fmt.Printf("[PeGS] Validation checks took %v\n", time.Since(tVerify))

	if *outputPath != "" {
		fmt.Printf("[PeGS] Saving synthetic data to %s...\n", *outputPath)
		tSave := time.Now()
		outBytes := make([]byte, len(synthBuffer)*2)
		for i, val := range synthBuffer {
			outBytes[i*2] = byte(val & 0xFF)
			outBytes[i*2+1] = byte(val >> 8)
		}
		if err := os.WriteFile(*outputPath, outBytes, 0644); err != nil {
			fmt.Printf("[PeGS] Error writing output file: %v\n", err)
		} else {
			saveDuration := time.Since(tSave)
			fmt.Printf("[PeGS] Successfully saved %d rows in %v (%.2f MB/s).\n", *numRows, saveDuration, float64(len(outBytes))/saveDuration.Seconds()/1e6)
		}
	}
}

func generateSimulatedData(buffer []uint16, numCols, cardinality int) {
	const numPeople = 5000000
	const avgRecordsPerPerson = 3
	rng := rand.New(rand.NewSource(42))

	// Track current row in buffer
	currentRow := 0
	numRows := len(buffer) / numCols

	for p := 0; p < numPeople && currentRow < numRows; p++ {
		// Determine number of records for this person (Poisson-ish distribution around 3)
		nRecords := 1 + rng.Intn(5) // Simple random between 1 and 5 records
		if currentRow+nRecords > numRows {
			nRecords = numRows - currentRow
		}

		// Person-level baseline deviations (sparse)
		baseline := make([]uint16, numCols)
		for c := 0; c < numCols; c++ {
			if rng.Float64() < 0.05 { // 5% chance of a baseline deviation
				baseline[c] = uint16(1 + rng.Intn(cardinality-1))
			} else {
				baseline[c] = 0
			}
		}

		for r := 0; r < nRecords; r++ {
			rowStart := currentRow * numCols
			for c := 0; c < numCols; c++ {
				// Record is baseline + some noise
				val := baseline[c]
				if rng.Float64() < 0.1 { // 10% chance of temporary deviation shift
					if val > 0 && rng.Float64() < 0.5 {
						val--
					} else if int(val) < cardinality-1 {
						val++
					}
				}
				buffer[rowStart+c] = val
			}
			currentRow++
		}
	}

	// Fill any remaining rows with sparse noise if needed
	for currentRow < numRows {
		rowStart := currentRow * numCols
		for c := 0; c < numCols; c++ {
			if rng.Float64() < 0.01 {
				buffer[rowStart+c] = uint16(1 + rng.Intn(cardinality-1))
			} else {
				buffer[rowStart+c] = 0
			}
		}
		currentRow++
	}
}
