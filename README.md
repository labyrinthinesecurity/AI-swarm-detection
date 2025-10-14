# Hierarchical Magnetization Analysis for Swarm Detection

## Overview

This repository implements a novel approach to detecting coordinated swarm behavior in peer-to-peer (P2P) networks using concepts from statistical physics. The method applies Ising model spin dynamics to identify spatial patterns that distinguish legitimate distributed traffic from coordinated botnets or malicious swarms.

Rather than relying on rule-based heuristics, the system maps behavioral network metrics onto a physical "spin field" and uses hierarchical spatial correlation analysis to detect emergent coordination patterns.

## Background

Traditional botnet detection methods struggle with decentralized swarms because:
- Individual nodes may appear normal in isolation
- Coordination is distributed across spatial and behavioral dimensions
- Adversaries can evade signature-based detection

This approach treats the network as a 2D lattice where each peer has a "spin" derived from traffic statistics. Coordinated swarms manifest as regions of coherent spin alignment, detectable through multi-scale magnetization analysis.

## Pipeline Architecture

The detection pipeline consists of four hierarchical tiers:

### Tier 1: Behavioral Metrics to Anomaly Scores

**Input**: Per-peer traffic statistics
- `alpha` (α): Heavy-tail exponent of inter-arrival times
- `kurtosis` (κ): Peakedness of traffic distribution  
- `hurst` (H): Long-range dependence (Hurst exponent)

**Functions**:
```python
compute_anomaly_score(alpha, kurtosis, hurst)
```

Combines the three metrics into a composite anomaly score ∈ [0,1]:

Legitimate traffic: α ∈ [2.5, 3.5], κ ∈ [3, 6], H ∈ [0.5, 0.6] → score ≈ 0
Swarm traffic: α ∈ [1.5, 2.0], κ ∈ [9, 15], H ∈ [0.65, 0.8] → score ≈ 1

Weighting: 30% alpha + 50% kurtosis + 20% hurst (kurtosis is the strongest indicator)

```python
anomaly_to_spin(anomaly_score)
```
Maps anomaly score to spin value ∈ [-1, +1]:

Score 0.0 → spin +1 (normal/legitimate)
Score 0.5 → spin 0 (neutral)
Score 1.0 → spin -1 (anomalous)

### Tier 2: Spatial Correlation (Moran Balls)

**Input**: Individual peer spins + spatial positions

**Function**:
```python
generate_moran_balls(peers, ambient_field_spin=1)
```
For each peer, computes a spatially-weighted local spin field:
local_spin_i = Σⱼ w(rᵢ, rⱼ) · spin_j / Σⱼ w(rᵢ, rⱼ)
where w(rᵢ, rⱼ) = exp(-d²/2σ²) is a Gaussian spatial weight.

Insight:
- Isolated anomaly: One bad peer among good neighbors → local_spin diluted toward +1
- Coordinated swarm: Cluster of bad peers → local_spin reinforced toward -1

Each Moran ball contains:
- local_spin: spatially-averaged spin value
- alignment: deviation from expected normal behavior (ambient_field_spin)
- position: spatial coordinates

**Helper Function**:
```python
weight_func(pos_i, pos_j)
```
Gaussian spatial weight with cutoff (σ = domain_size/15, d_max = domain_size/2)

### Tier 3: Hierarchical Coarse-Graining (Kadanoff Blocks)

**Input**: Moran balls with local spin fields

**Function**:
```python
kadanoff_aggregation(moran_balls, levels=8, initial_block_size=12.5)
```

Implements renormalization group transformation:
- Divide space into grid blocks of size block_size
- Compute block-level magnetization
- Coarse-grain: replace each block with single representative
- Double block size and repeat

At each level:
```python
spatial_blocking(balls, block_size)
```
Groups balls into spatial grid cells

```python
compute_block_magnetizations(blocks, ambient_field_spin=1.0)
```
Computes magnetization M ∈ [0,1] measuring:
- Deviation: How far block spins deviate from ambient field
- Coherence: How similar blocks are to each other (inverse coefficient of variation)
- Combined: M = deviation × coherence (requires both to be high)

**Output**: List of levels with:
- n_points: number of spatial points at this scale
- n_blocks: number of blocks formed
- magnetization: M value at this scale
- spin_variance: variance of spins within level
- spin_range: (min, max) spin values

### Tier 4: Classification and Detection

**Function**:
```python
tier3_summary(levels)
```
Analyzes magnetization evolution across scales and produces classification.

Computed Metrics:
Primary M (max): Maximum magnetization across all scales
Indicates strongest coherent signal

Fine-scale M (L0): Magnetization at finest resolution
Baseline local coordination level

Decay slope: Linear fit of log(M) vs. level
Positive slope: coordination strengthens with scale (hierarchical swarm)
Near-zero slope: persistent coordination (stable swarm)
Negative slope: coordination fades with scale (decaying clusters)



## Results Output Format
```
======================================================================
Tier 3: Hierarchical Magnetization Analysis
======================================================================
Level  N_pts   N_blks  BlkSize    M_block
----------------------------------------------------------------------
0      100     51      12.5       0.126
       │██████
1      51      16      25.0       0.127
       │██████
2      16      4       50.0       0.128
       │██████
3      4       1       100.0      0.129
       │██████
======================================================================
Swarm Detection:
  Primary |M| (max):        0.129
  Fine-scale |M| (L0):      0.126
  Decay slope (log):        0.008
  Classification:           LOW
  Interpretation:           Legitimate / Random traffic
======================================================================
```

## Usage
Basic Example
```python
from swarm import generate_peers, generate_moran_balls, kadanoff_aggregation, tier3_summary
```

### Generate synthetic peer population
```
peers = generate_peers(
    n_peers=100,
    domain_size=(100, 100),
    scenario="Legitimate",  # or "Swarm", "Mixed", "Directional"
    seed=42
)
```

### Display results and classification
```
results = tier3_summary(levels)
print(f"Swarm probability: {results['swarm_probability']}")
```

### Analyzing Real Data
To use with real network data, provide peers as a list of dictionaries:
```
pythonpeers = [
    {
        'alpha': 2.8,        # Heavy-tail exponent
        'kurtosis': 5.2,     # Traffic distribution peakedness
        'hurst': 0.56,       # Long-range dependence
        'position': (x, y)   # Spatial coordinates (e.g., IP geolocation)
    },
    # ... more peers
]
```

## Scenarios Tested
The code includes four test scenarios:

Legitimate P2P: Normal distributed traffic (α≈3, κ≈5, H≈0.55)

Expected: LOW classification, M < 0.2


Coordinated Swarm: Botnet-like behavior (α≈1.8, κ≈11, H≈0.7)

Expected: HIGH classification, M > 0.4, positive slope


Directional Swarm: Spatial gradient (normal on left, swarm on right)

Expected: MEDIUM-HIGH, directional magnetization pattern


Mixed Population: 50/50 legitimate and swarm peers

Expected: MEDIUM, intermediate M values
