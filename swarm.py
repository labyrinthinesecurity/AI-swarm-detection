#!/usr/bin/env python3
"""
Hierarchical AI Swarm Detection for Local Wireless P2P Networks

APPLICATION DOMAIN:
- WiFi Direct file sharing (Android Nearby Share, Apple AirDrop)
- Bluetooth mesh networks (IoT, smart home devices)
- Emergency ad-hoc networks (disaster response)
- Vehicular networks (V2V communication)

THREAT MODEL:
- Coordinated LLM digital assistants on devices performing local attacks (AI swarms)
- Malicious mesh nodes corrupting data propagation
- AI orchestrated Privacy attacks via device fingerprinting clusters
- AI Physical surveillance through coordinated beaconing

WHY SPATIAL ANALYSIS MATTERS:
Unlike internet P2P, wireless range constraints mean:
- Position IS the network topology
- Physical clustering indicates coordinated deployment
- Spatial correlation reveals non-random device patterns
"""

import numpy as np
from collections import defaultdict
import math
import random

domain_size=(100,100)


# ============================================================================
#  Tier 1: Anomaly Score Computation At Device/Local level
# ============================================================================

def generate_peers(n_peers, domain_size=domain_size, scenario="Legitimate", seed=None):
    """
    Generate realistic peer populations with behavioral metrics.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    peers = []

    for i in range(n_peers):
        pos_x = random.uniform(0, domain_size[0])
        pos_y = random.uniform(0, domain_size[1])

        # Generate realistic behavioral metrics based on scenario
        if scenario == "Legitimate":
            # Normal P2P: α~3, κ~5, H~0.55
            alpha = np.random.normal(3.0, 0.3)
            kurtosis = np.random.normal(5.0, 1.0)
            hurst = np.random.normal(0.55, 0.05)

        elif scenario == "Swarm":
            # Swarm behavior: α~1.8, κ~11, H~0.7
            alpha = np.random.normal(1.8, 0.2)
            kurtosis = np.random.normal(11.0, 1.5)
            hurst = np.random.normal(0.70, 0.05)

        elif scenario == "Mixed":
            # 50/50 mix
            if random.random() < 0.5:
                alpha = np.random.normal(3.0, 0.3)
                kurtosis = np.random.normal(5.0, 1.0)
                hurst = np.random.normal(0.55, 0.05)
            else:
                alpha = np.random.normal(1.8, 0.2)
                kurtosis = np.random.normal(11.0, 1.5)
                hurst = np.random.normal(0.70, 0.05)

        elif scenario == "Directional":
            # Spatial gradient: anomalous on one side
            if pos_x < 50:
                alpha = np.random.normal(3.0, 0.3)
                kurtosis = np.random.normal(5.0, 1.0)
                hurst = np.random.normal(0.55, 0.05)
            else:
                alpha = np.random.normal(1.8, 0.2)
                kurtosis = np.random.normal(11.0, 1.5)
                hurst = np.random.normal(0.70, 0.05)
        else:
            raise ValueError(f"Unknown scenario: {scenario}")

        # Clip to reasonable ranges
        alpha = np.clip(alpha, 1.0, 4.0)
        kurtosis = np.clip(kurtosis, 3.0, 20.0)
        hurst = np.clip(hurst, 0.4, 0.9)

        peers.append({
            'alpha': float(alpha),
            'kurtosis': float(kurtosis),
            'hurst': float(hurst),
            'position': (pos_x, pos_y)
        })

    return peers

def compute_anomaly_score(alpha, kurtosis, hurst):
    """
    Compute composite anomaly score from behavioral metrics.
    Returns value in [0, 1] where higher = more anomalous.
    
    Reference thresholds (from empirical P2P analysis):
    - Legitimate: α ∈ [2.5, 3.5], κ ∈ [3, 6], H ∈ [0.5, 0.6]
    - Swarm: α ∈ [1.5, 2.0], κ ∈ [9, 15], H ∈ [0.65, 0.8]
    """
    # Normalize each metric to [0,1] anomaly scale
    alpha_anom = max(0, (3.0 - alpha) / 1.5)  # Lower α = more anomalous
    kurtosis_anom = max(0, (kurtosis - 6.0) / 9.0)  # Higher κ = more anomalous  
    hurst_anom = max(0, (hurst - 0.6) / 0.2)  # Higher H = more anomalous
    
    # Clip to [0, 1]
    alpha_anom = np.clip(alpha_anom, 0, 1)
    kurtosis_anom = np.clip(kurtosis_anom, 0, 1)
    hurst_anom = np.clip(hurst_anom, 0, 1)
    
    # Weighted combination (kurtosis is strongest indicator)
    score = 0.3 * alpha_anom + 0.5 * kurtosis_anom + 0.2 * hurst_anom
    
    return float(np.clip(score, 0, 1))


def anomaly_to_spin(anomaly_score, threshold=0.5):
    """
    Continuous spin in [-1, +1]. Increased gain for better separation.
    Map: anomaly_score 0 -> +1, 0.5 -> 0, 1 -> -1.
    """
    gain = 4.0
    return float(np.tanh((0.5 - anomaly_score) * gain))

# ============================================================================
#  Tier 2: Moran Ball Generation
# ============================================================================

def weight_func(pos_i, pos_j):
    """
    Gaussian spatial weight with cutoff.
    Reduced sigma and dmax so local averaging preserves neighborhood differences.
    """
    sigma=domain_size[0]/15
    dmax=domain_size[0]/2
    dx = pos_i[0] - pos_j[0]
    dy = pos_i[1] - pos_j[1]
    dist2 = dx*dx + dy*dy
    dist = math.sqrt(dist2)

    if dist > dmax:
        return 0.0

    return math.exp(-dist2 / (2 * sigma**2))

def generate_moran_balls(peers, ambient_field_spin=1, weight_func_custom=None):
    """
    Compute local spin field from peer anomaly patterns.
    
    Args:
        peers: List of dicts with 'position', 'alpha', 'kurtosis', 'hurst'
        ambient_field_spin: Expected spin for normal behavior (+1)
        weight_func_custom: Optional custom spatial weight function
    
    Returns:
        List of Moran balls with 'local_spin' and 'alignment'
    """
    if weight_func_custom is None:
        weight_func_custom = weight_func
    
    # First: compute anomaly scores and spins for all peers
    for peer in peers:
        anom_score = compute_anomaly_score(
            peer['alpha'], 
            peer['kurtosis'], 
            peer['hurst']
        )
        peer['anomaly_score'] = anom_score
        peer['spin'] = anomaly_to_spin(anom_score)
    
    moran_balls = []
    
    # Second: compute spatially-weighted local fields
    for peer in peers:
        w_sum = 0.0
        weighted_spin_sum = 0.0
        
        for other in peers:
            w = weight_func_custom(peer['position'], other['position'])
            w_sum += w
            weighted_spin_sum += w * other['spin']
        
        if w_sum > 0:
            raw_local = weighted_spin_sum / w_sum
        else:
            raw_local = peer['spin']
        
        # Smooth saturation to [-1, 1]
        local_spin = np.tanh(raw_local)

        alignment = local_spin - ambient_field_spin
        
        moran_balls.append({
            'position': peer['position'],
            'local_spin': float(local_spin),
            'alignment': float(alignment),
            'anomaly_score': peer['anomaly_score'],
            'spin': peer['spin']
        })
    
    return moran_balls

# ============================================================================
#  Tier 3 : Hierarchical Coarse-Graining
# ============================================================================

def spatial_blocking(balls, block_size):
    """Group balls into spatial grid blocks."""
    blocks = defaultdict(list)
    for ball in balls:
        x, y = ball['position']
        cell_x = int(x // block_size)
        cell_y = int(y // block_size)
        blocks[(cell_x, cell_y)].append(ball)
    return list(blocks.values())


def compute_block_magnetizations(blocks, ambient_field_spin=1.0):
    """
    Compute per-block magnetization measuring both deviation from the ambient
    field and inter-block coherence, with population weighting.

    Each block contributes proportionally to its number of contained points.
    The final magnetization lies in [0, 1]:
        0  -> perfectly normal / incoherent
        1  -> maximally anomalous and spatially coherent
    """

    # ------------------------------------------------------------------
    # 1. Guard clauses
    # ------------------------------------------------------------------
    if len(blocks) == 0:
        return [], 0.0

    # Filter out empty blocks just in case
    valid_blocks = [block for block in blocks if len(block) > 0]
    if len(valid_blocks) == 0:
        return [], 0.0

    # ------------------------------------------------------------------
    # 2. Compute mean spin per block and population weights
    # ------------------------------------------------------------------
    block_sizes = np.array([len(block) for block in valid_blocks], dtype=float)
    block_mags = np.array([np.mean([b['local_spin'] for b in block])
                           for block in valid_blocks], dtype=float)
    weights = block_sizes / np.sum(block_sizes)  # normalized weights (sum = 1)

    # ------------------------------------------------------------------
    # 3. Deviation from ambient (weighted)
    #    |block_spin - ambient| / 2 maps deviation to [0,1]
    # ------------------------------------------------------------------
    deviation = float(np.sum(weights * np.abs(block_mags - ambient_field_spin)) / 2.0)

    # ------------------------------------------------------------------
    # 4. Inter-block coherence (weighted coefficient of variation)
    #    Low cv => blocks are similar => high coherence
    # ------------------------------------------------------------------
    weighted_mean = np.sum(weights * np.abs(block_mags)) + 1e-9
    weighted_mean_raw = np.sum(weights * block_mags)
    weighted_var = np.sum(weights * (block_mags - weighted_mean_raw) ** 2)
    weighted_std = math.sqrt(weighted_var)
    cv = weighted_std / weighted_mean
    coherence = 1.0 / (1.0 + cv)  # maps [0,∞) → (0,1]

    # ------------------------------------------------------------------
    # 5. Combined metric: deviation × coherence
    #    Requires both to be high for strong magnetization
    # ------------------------------------------------------------------
    magnetization = float(np.clip(deviation * coherence, 0.0, 1.0))

    # ------------------------------------------------------------------
    # 6. Return both block list (for diagnostics) and scalar metric
    # ------------------------------------------------------------------
    return list(block_mags), magnetization

def kadanoff_aggregation(moran_balls, levels=8, initial_block_size=12.5, ambient_field_spin=1.0):
    """
    Hierarchical coarse-graining of spin field.

    ambient_field_spin: expected normal spin (default +1.0). Passed to compute_block_magnetizations.
    """
    current_balls = [
        {
            'position': tuple(b['position']),
            'local_spin': float(b['local_spin'])
        }
        for b in moran_balls
    ]

    result = []
    block_size = initial_block_size

    for lvl in range(levels):
        n_points = len(current_balls)
        if n_points == 0:
            break

        # Create spatial blocks at current scale
        blocks = spatial_blocking(current_balls, block_size)

        if len(blocks) == 0:
            break
        block_counts = [len(block) for block in blocks]
        min_b, max_b, med_b = min(block_counts), max(block_counts), np.median(block_counts)
        print(f"  [L{lvl}] ... BlockCount(min/med/max)={min_b}/{med_b}/{max_b}")

        # DIAGNOSTIC: Let's see what's happening
        all_spins = [b['local_spin'] for b in current_balls]
        spin_variance = np.var(all_spins)
        spin_range = (np.min(all_spins), np.max(all_spins))

        # Compute block statistics (now relative to ambient_field_spin)
        block_mags, magnetization = compute_block_magnetizations(blocks, ambient_field_spin=ambient_field_spin)

        global_mean = np.mean(all_spins)

        print(f"  [L{lvl}] N={n_points}, Blocks={len(blocks)}, "
              f"Spin range=[{spin_range[0]:.3f}, {spin_range[1]:.3f}], "
              f"Var={spin_variance:.4f}, M={magnetization:.3f}")

        result.append({
            'level': lvl,
            'n_points': n_points,
            'n_blocks': len(blocks),
            'block_size': block_size,
            'magnetization': magnetization,
            'global_spin': float(global_mean),
            'spin_variance': float(spin_variance),
            'spin_range': spin_range,
            'block_mags': block_mags
        })

        # Coarse-grain to next level
        next_balls = []
        for block in blocks:
            if len(block) == 0:
                continue

            xs = [b['position'][0] for b in block]
            ys = [b['position'][1] for b in block]
            avg_pos = (np.mean(xs), np.mean(ys))
            avg_spin = np.mean([b['local_spin'] for b in block])

            next_balls.append({
                'position': avg_pos,
                'local_spin': np.clip(avg_spin, -1.0, 1.0)
            })

        # If we collapse to a single coarse point, compute its magnetization relative to ambient.
        if len(next_balls) <= 1:
            if len(next_balls) == 1:
                final_spin = next_balls[0]['local_spin']
                final_mag = float(abs(final_spin - ambient_field_spin) / 2.0)
                result.append({
                    'level': lvl + 1,
                    'n_points': 1,
                    'n_blocks': 1,
                    'block_size': block_size * 2,
                    'magnetization': final_mag,
                    'global_spin': final_spin,
                    'spin_variance': 0.0,
                    'spin_range': (final_spin, final_spin),
                    'block_mags': [final_spin]
                })
            break

        current_balls = next_balls
        block_size *= 2.0

    return result


# ============================================================================
#  Visualization
# ============================================================================
def tier3_summary(levels):
    """Display hierarchical magnetization results."""
    if not levels:
        print("No levels to display")
        return

    print("\n" + "="*70)
    print("Tier 3: Hierarchical Magnetization Analysis")
    print("="*70)
    print(f"{'Level':<6} {'N_pts':<7} {'N_blks':<7} {'BlkSize':<10} {'M_block':<10}")
    print("-"*70)

    for lvl in levels:
        level_num = lvl['level']
        n_pts = lvl['n_points']
        n_blks = lvl['n_blocks']
        blk_sz = lvl['block_size']
        M = lvl['magnetization']

        bar_len = int(M * 50)
        bar = "█" * bar_len

        print(f"{level_num:<6} {n_pts:<7} {n_blks:<7} {blk_sz:<10.1f} {M:<10.3f}")
        print(f"       │{bar}")

    print("="*70)

    # Use non-degenerate levels (n_points > 1) for global metrics; fallback if none.
    mags_all = [lvl['magnetization'] for lvl in levels]
    mags_non_degenerate = [lvl['magnetization'] for lvl in levels if lvl.get('n_points', 0) > 1]

    primary_M = max(mags_non_degenerate) if mags_non_degenerate else (max(mags_all) if mags_all else 0.0)
    secondary_M = levels[0]['magnetization'] if levels else 0.0

    if len(mags_non_degenerate) >= 2:
        x = np.arange(len(mags_non_degenerate))
        log_mags = np.log(np.array(mags_non_degenerate) + 1e-6)
        slope = np.polyfit(x, log_mags, 1)[0]
    else:
        slope = 0.0

    # Classification
    # ===============================================================
    # Classification logic using three metrics: primary, secondary, slope
    # ===============================================================
    #
    # Interpretations:
    #  - Legitimate: low M at all scales, slope ~0
    #  - Random/Distributed: low primary, low slope, maybe slightly >0 secondary
    #  - Directional/decaying: medium primary, negative slope (structure fades)
    #  - Persistent swarm: high primary, small or positive slope (structure persists)
    #  - Hierarchical swarm: high primary, strongly positive slope (structure strengthens)
    #

    if primary_M < 0.25 and secondary_M < 0.25:
      swarm_prob = "LOW"
      interp = "Legitimate / Random traffic"
    elif primary_M < 0.35:
      # weak global coherence
      swarm_prob = "MEDIUM"
      interp = "Weak clustering / distributed anomalies"
    else:
      # primary_M >= 0.35: check slope for scale behavior
      if slope > 0.10:
        swarm_prob = "HIGH"
        interp = "Hierarchical/directional swarm (coherence strengthens with scale)"
      elif slope > -0.05:
        swarm_prob = "HIGH"
        interp = "Persistent coordinated swarm"
      else:
        swarm_prob = "MEDIUM–HIGH"
        interp = "Decaying hierarchical clusters (coherent but fading)"

    print(f"\nSwarm Detection:")
    print(f"  Primary |M| (max):        {primary_M:.3f}")
    print(f"  Fine-scale |M| (L0):      {secondary_M:.3f}")
    print(f"  Decay slope (log):        {slope:.3f}")
    print(f"  Classification:           {swarm_prob}")
    print(f"  Interpretation:           {interp}")
    print("="*70 + "\n")

    return {
        'primary_M': primary_M,
        'secondary_M': secondary_M,
        'slope': slope,
        'swarm_probability': swarm_prob
    }


# ============================================================================
#  Main
# ============================================================================

if __name__ == "__main__":
    
    scenarios = [
        ("Legitimate P2P", 42, "Legitimate", 100),
        ("Coordinated Swarm", 43, "Swarm", 100),
        ("Directional Swarm", 44, "Directional", 100),
        ("Mixed Population", 45, "Mixed", 100),
    ]
    
    for name, seed, scenario, n_peers in scenarios:
        print(f"\n{'='*70}")
        print(f"SCENARIO: {name}")
        print(f"{'='*70}")
        
        # Generate peers with scenario-based behavioral metrics
        peers = generate_peers(n_peers, domain_size=(100, 100), 
                              scenario=scenario, seed=seed)
        
        # Compute local spin fields from anomaly patterns
        moran_balls = generate_moran_balls(peers, ambient_field_spin=1)
        
        # Run hierarchical analysis
        levels = kadanoff_aggregation(moran_balls, levels=8, initial_block_size=12.5)
        
        # Display results
        tier3_summary(levels)
