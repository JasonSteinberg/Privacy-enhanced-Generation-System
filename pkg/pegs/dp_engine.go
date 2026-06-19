package pegs

import "math"

// ComputeDPDistribution scales empirical frequencies safely under differential privacy.
func ComputeDPDistribution(peCounts []float64, epsilon float64) []float64 {
	card := len(peCounts)
	// Normalize probabilities first if they are raw counts
	sum := 0.0
	for _, p := range peCounts {
		sum += p
	}

	scaledWeights := make([]float64, card)

	// Identify maximum scaled value to stabilize the exponent calculations
	maxZ := -math.MaxFloat64
	for i := 0; i < card; i++ {
		p := peCounts[i]
		if sum > 0 {
			p /= sum
		}
		scaledWeights[i] = epsilon * p
		if scaledWeights[i] > maxZ {
			maxZ = scaledWeights[i]
		}
	}

	// Apply Log-Sum-Exp subtraction: log(Sum(exp(z_i))) = maxZ + log(Sum(exp(z_i - maxZ)))
	sumExp := 0.0
	for i := 0; i < card; i++ {
		sumExp += math.Exp(scaledWeights[i] - maxZ)
	}
	logSumExp := maxZ + math.Log(sumExp)

	// Compute stable output probabilities
	q := make([]float64, card)
	qSum := 0.0
	for i := 0; i < card; i++ {
		q[i] = math.Exp(scaledWeights[i] - logSumExp)
		qSum += q[i]
	}

	// Final renormalization for floating point precision
	if qSum > 0 && math.Abs(qSum-1.0) > 1e-9 {
		for i := 0; i < card; i++ {
			q[i] /= qSum
		}
	}

	return q
}
