#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
import functools
from collections import defaultdict
import math
import random
import argparse

print = functools.partial(print, flush=True)

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

# ============================================================================
#  Parse Command-Line Arguments
# ============================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Swarm Detection with Configurable Behavioral Parameters',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Legitimate (normal) behavioral parameters
    parser.add_argument('--alpha_legit', type=float, default=3.0,
                        help='Mean alpha for legitimate peers')
    parser.add_argument('--alpha_legit_std', type=float, default=0.3,
                        help='Std dev of alpha for legitimate peers')
    parser.add_argument('--kurtosis_legit', type=float, default=5.0,
                        help='Mean kurtosis for legitimate peers')
    parser.add_argument('--kurtosis_legit_std', type=float, default=1.0,
                        help='Std dev of kurtosis for legitimate peers')
    parser.add_argument('--hurst_legit', type=float, default=0.55,
                        help='Mean Hurst exponent for legitimate peers')
    parser.add_argument('--hurst_legit_std', type=float, default=0.05,
                        help='Std dev of Hurst exponent for legitimate peers')
    
    # Anomalous (swarm) behavioral parameters
    parser.add_argument('--alpha_anom', type=float, default=1.8,
                        help='Mean alpha for anomalous peers')
    parser.add_argument('--alpha_anom_std', type=float, default=0.2,
                        help='Std dev of alpha for anomalous peers')
    parser.add_argument('--kurtosis_anom', type=float, default=11.0,
                        help='Mean kurtosis for anomalous peers')
    parser.add_argument('--kurtosis_anom_std', type=float, default=1.5,
                        help='Std dev of kurtosis for anomalous peers')
    parser.add_argument('--hurst_anom', type=float, default=0.70,
                        help='Mean Hurst exponent for anomalous peers')
    parser.add_argument('--hurst_anom_std', type=float, default=0.05,
                        help='Std dev of Hurst exponent for anomalous peers')
    
    # Weighting coefficients for anomaly score
    parser.add_argument('--c0', type=float, default=0.3,
                        help='Weight coefficient for alpha (c0)')
    parser.add_argument('--c1', type=float, default=0.5,
                        help='Weight coefficient for kurtosis (c1), Hurst gets (1-c0-c1)')
    
    # Simulation parameters
    parser.add_argument('--N', type=int, default=1000,
                        help='Number of peers')
    parser.add_argument('--domain_size', type=int, default=75,
                        help='Domain size (square: domain_size x domain_size)')
    parser.add_argument('--scenario', type=str, default='Directional',
                        help='Swarm scenario (RingSwarm, Directional, Mixed, etc.)')
    parser.add_argument('--n_seeds', type=int, default=100,
                        help='Number of random seeds for averaging')
    parser.add_argument('--ambient', type=float, default=0.75,
                        help='Ambient field spin value')
    
    return parser.parse_args()

# ============================================================================
#  Tier 1: Anomaly Score Computation At Device/Local level
# ============================================================================
# ============================================================================
#  Centralized Behavior Parameters (will be overridden by command-line args)
# ============================================================================

BEHAVIOR_PARAMS = {
    'legitimate': {
        'alpha': (3.0, 0.3),      # (mean, std) - defaults
        'kurtosis': (5.0, 1.0),
        'hurst': (0.55, 0.05)
    },
    'anomalous': {
        'alpha': (1.8, 0.2),
        'kurtosis': (11.0, 1.5),
        'hurst': (0.70, 0.05)
    }
}

# Global variables for weighting coefficients and legitimate centers
# Will be set from command-line args or by wrapper scripts
C0 = 0.3
C1 = 0.5
ALPHA_LEGIT_CENTER = 3.0
KURTOSIS_LEGIT_CENTER = 5.0
HURST_LEGIT_CENTER = 0.55
ALPHA_LEGIT_STD = 0.3
KURTOSIS_LEGIT_STD = 1.0
HURST_LEGIT_STD = 0.05

# Global variables for simulation parameters
# Will be set from command-line args or by wrapper scripts
masterseed = 0  # Default value, will be overridden
ambient = 0.75  # Default ambient field spin value

'''
BEHAVIOR_PARAMS = {
    'legitimate': {
        'alpha': (3.0, 1.0),      # (mean, std)
        'kurtosis': (5.0, 2.0),
        'hurst': (0.55, 0.05)
    },
    'anomalous': {
        'alpha': (1.8, 0.9),
        'kurtosis': (7.0, 2.5),
        'hurst': (0.70, 0.05)
    }
}
'''

def sample_behavior(rng, behavior_type='legitimate'):
    """
    Sample behavioral metrics for a peer.
    
    Args:
        rng: numpy RandomState object
        behavior_type: 'legitimate' or 'anomalous'
    
    Returns:
        dict with keys: alpha, kurtosis, hurst
    """
    params = BEHAVIOR_PARAMS[behavior_type]
    return {
        'alpha': rng.normal(*params['alpha']),
        'kurtosis': rng.normal(*params['kurtosis']),
        'hurst': rng.normal(*params['hurst'])
    }

def generate_peers(n_peers, domain_size=(200,200), scenario="Legitimate", seed=None):
    """
    Generate peer populations with behavioral metrics.
    Supports ring swarm scenarios with partial coverage.
    """
    rng = np.random.RandomState(seed) if seed is not None else np.random
    print("rng seeded with", seed)
    peers = []
    legitimate_count = 0
    illegitimate_count = 0
    center_x, center_y = domain_size[0]/2, domain_size[1]/2

    for i in range(n_peers):
        # Initialize with default legitimate behavior
        is_anomalous = False
        pos_x = rng.uniform(0, domain_size[0])
        pos_y = rng.uniform(0, domain_size[1])

        if scenario.startswith("Triangles"):
            # Determine spatial coverage fraction
            coverage_map = {
                "Triangles 10%": 0.10,
                "Triangles 25%": 0.25,
                "Triangles": 0.50,
                "Triangles 75%": 0.75
            }
            coverage = coverage_map.get(scenario)
            if coverage is None:
                raise ValueError(f"Unknown Triangles scenario: {scenario}")

            # Calculate triangle boundaries based on coverage
            height, width = domain_size
            triangle_size = math.sqrt(coverage * width * height)

            # Top-left triangle
            in_top_left = (pos_x <= triangle_size and
                          pos_y <= triangle_size and
                          pos_x + pos_y <= triangle_size)

            # Bottom-right triangle
            in_bottom_right = (pos_x >= width - triangle_size and
                              pos_y >= height - triangle_size and
                              (width - pos_x) + (height - pos_y) <= triangle_size)

            is_anomalous = in_top_left or in_bottom_right

        elif scenario.startswith("RingSwarm"):
            # Determine coverage fraction
            coverage_map = {
                "RingSwarm 10%": 0.10,
                "RingSwarm 25%": 0.25,
                "RingSwarm": 0.50,
                "RingSwarm 75%": 0.75
            }
            coverage = coverage_map.get(scenario)
            if coverage is None:
                raise ValueError(f"Unknown RingSwarm scenario: {scenario}")

            # Calculate radius based on coverage
            max_radius = min(domain_size) / 2
            domain_area = domain_size[0] * domain_size[1]
            target_area = coverage * domain_area
            target_radius = math.sqrt(target_area / math.pi)
            radius = min(target_radius, max_radius)

            dist = math.sqrt((pos_x - center_x)**2 + (pos_y - center_y)**2)
            is_anomalous = dist <= radius

            # Clip to domain
            pos_x = np.clip(pos_x, 0, domain_size[0])
            pos_y = np.clip(pos_y, 0, domain_size[1])

        elif scenario == "Legitimate":
            is_anomalous = False

        elif scenario == "Swarm":
            is_anomalous = True

        elif scenario == "Mixed":
            is_anomalous = rng.random() < 0.5

        elif scenario.startswith("Directional"):
            # Spatial gradient scenarios
            directional_map = {
                "Directional": (0.50, 'x'),           # 50% split on x-axis
                "Directional 75%": (0.25, 'y'),       # 75% anomalous (top 75%)
                "Directional 33%": (0.67, 'y'),       # 33% anomalous (top 33%)
                "Directional 25%": (0.75, 'y')        # 25% anomalous (top 25%)
            }

            config = directional_map.get(scenario)
            if config is None:
                raise ValueError(f"Unknown Directional scenario: {scenario}")

            threshold, axis = config
            if axis == 'x':
                is_anomalous = pos_x >= domain_size[0] * threshold
            else:  # axis == 'y'
                is_anomalous = pos_y >= domain_size[1] * threshold

        else:
            raise ValueError(f"Unknown scenario: {scenario}")

        # Sample behavior based on classification
        if is_anomalous:
            illegitimate_count += 1
            behavior = sample_behavior(rng, 'anomalous')
        else:
            legitimate_count += 1
            behavior = sample_behavior(rng, 'legitimate')

        peers.append({
            'alpha': float(behavior['alpha']),
            'kurtosis': float(behavior['kurtosis']),
            'hurst': float(behavior['hurst']),
            'position': (pos_x, pos_y)
        })

    print(f"Generated {n_peers} peers: {legitimate_count} legitimate, {illegitimate_count} illegitimate")
    return peers

def compute_anomaly_score(alpha, kurtosis, hurst):
    """
    Compute anomaly score using global parameters set from command-line args.
    Uses 3-sigma distances from legitimate centers.
    """
    global C0, C1, ALPHA_LEGIT_CENTER, KURTOSIS_LEGIT_CENTER, HURST_LEGIT_CENTER
    global ALPHA_LEGIT_STD, KURTOSIS_LEGIT_STD, HURST_LEGIT_STD
    
    # Distance from legitimate centers, using 3 sigmas at denominator to capture 99.7%
    alpha_dev = abs(ALPHA_LEGIT_CENTER - alpha) / (3.0 * ALPHA_LEGIT_STD)
    kurtosis_dev = abs(KURTOSIS_LEGIT_CENTER - kurtosis) / (3.0 * KURTOSIS_LEGIT_STD)
    hurst_dev = abs(HURST_LEGIT_CENTER - hurst) / (3.0 * HURST_LEGIT_STD)
    
    # Clip and combine with coefficients c0, c1, (1-c0-c1)
    c2 = 1.0 - C0 - C1
    score = C0 * min(1, alpha_dev) + C1 * min(1, kurtosis_dev) + c2 * min(1, hurst_dev)
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

def weight_func(pos_i, pos_j, sigma, twosigma2, dmax, dmax2):
    """
    Gaussian spatial weight with cutoff.
    Reduced sigma and dmax so local averaging preserves neighborhood differences.
    """
    #sigma=100/15  #sigma=6.66667 i.e 10*(2/3) for n_peers=100 domain_size=(100,100)
    #sigma=tnnd * (2/3)    # sigma=6.6667 for n_peers=100 domain_size=(100,100)
    #dmax=2 * sigma  # dmax=50 for domain_size=(100,100)
    #dmax= 50
    dx = pos_i[0] - pos_j[0]
    dy = pos_i[1] - pos_j[1]
    dist2 = dx*dx + dy*dy

    dmax2 = 9.0 * sigma*sigma

    if dist2 > dmax2:
        return 0.0

    return math.exp(-dist2 / (twosigma2))

def generate_moran_balls(peers=None, ambient_field_spin=0.75, sigma=100/15, dmax=50, weight_func_custom=None):
    """
    Compute local spin field from peer anomaly patterns.
    
    Args:
        peers: List of dicts with 'position', 'alpha', 'kurtosis', 'hurst'
        ambient_field_spin: Expected spin for normal behavior (+1)
        weight_func_custom: Optional custom spatial weight function
    
    Returns:
        List of Moran balls with 'local_spin' and 'alignment'
    """
    print("gen moran balls sigma",sigma,"dmax",dmax)
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
    dmax2 = dmax*dmax
    twosigma2 = 2 * sigma**2

    # Second: compute spatially-weighted local fields
    for i,peer in enumerate(peers):
        w_sum = 0.0
        weighted_spin_sum = 0.0
    
        for j,other in enumerate(peers):
            if i==j:
                #weighted_spin_sum += ambient_field_spin
                continue
            w = weight_func_custom(peer['position'], other['position'], sigma, twosigma2, dmax, dmax2)
            w_sum += w
            weighted_spin_sum += w * other['spin']
        
        if w_sum > 0:
         raw_local = weighted_spin_sum / w_sum
        else:
        #raw_local = peer['spin']
         raw_local = ambient_field_spin
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


def compute_block_magnetizations(blocks, ambient_field_spin=0.75):
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

def kadanoff_aggregation(moran_balls, levels=8, initial_block_size=12.5, ambient_field_spin=0.75,verbose=True):
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
        if verbose:
            print(f"  [L{lvl}] ... BlockCount(min/med/max)={min_b}/{med_b}/{max_b}")

        all_spins = [b['local_spin'] for b in current_balls]
        spin_variance = np.var(all_spins)
        spin_range = (np.min(all_spins), np.max(all_spins))

        # Compute block statistics (relative to ambient_field_spin)
        block_mags, magnetization = compute_block_magnetizations(blocks, ambient_field_spin=ambient_field_spin)

        global_mean = np.mean(all_spins)

        if verbose:
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
def tier3_summary(levels,verbose):
    """Display hierarchical magnetization results."""
    if not levels:
        print("No levels to display")
        return

    if verbose:
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

        if verbose:
            print(f"{level_num:<6} {n_pts:<7} {n_blks:<7} {blk_sz:<10.1f} {M:<10.3f}")
            print(f"       │{bar}")

    if verbose:
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
    # Classification logic using 3 metrics: primary, secondary, slope
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

import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

def typical_nn_distance(n_peers, domain_size):
    # Approximate nearest-neighbour scale ~ 1/sqrt(density)
    area = domain_size[0] * domain_size[1]
    rho = n_peers / area
    return 1.0 / np.sqrt(rho)

def smooth_and_peak_sigma(sigma_vals, M_vals, window=7, poly=2):
    """
    Smooth M(σ) with Savitzky-Golay and return sigma at max dM/dσ
    """
    if len(sigma_vals) < window:
        # fallback: raw gradient
        dM = np.gradient(M_vals, sigma_vals)
        dM_excluded = dM[1:-1]
        idx = np.nanargmax(dM_excluded)
        # Map back to the original sigma_vals array
        return float(sigma_vals[idx + 1]), dM
    M_s = savgol_filter(M_vals, window_length=window, polyorder=poly)
    dM = np.gradient(M_s, sigma_vals)
    dM_excluded = dM[1:-1]
    idx = np.nanargmax(dM_excluded)

    # Map back to the original sigma_vals array
    return float(sigma_vals[idx + 1]), dM

def estimate_sigma_c_bootstrap(all_M_per_seed, sigma_values, n_boot=200, window=7, poly=2):
    """
    all_M_per_seed: (n_seeds, n_sigmas)
    Returns mean, std, 95% CI of sigma_c
    """
    n_seeds = all_M_per_seed.shape[0]
    sigma_cs = []
    for _ in range(n_boot):
        ids = np.random.choice(n_seeds, size=n_seeds, replace=True)
        M_boot = all_M_per_seed[ids].mean(axis=0)
        sigma_c, _ = smooth_and_peak_sigma(sigma_values, M_boot, window=window, poly=poly)
        sigma_cs.append(sigma_c)
    arr = np.array(sigma_cs)
    return arr.mean(), arr.std(), np.percentile(arr, 2.5), np.percentile(arr, 97.5)

def find_sigma_critical(
    N,
    domain_size=(125,125),
    swarm_scenario=None,
    block_size=12.5,
    n_seeds=30,
    toroidal=False,
    dmax_factor=2.0,
    smoothing_window=31,
    smoothing_poly=3,
):
    """
    Returns detailed results per N: bootstrap mean/std/CI for sigma_c, xi, and M(σ) curves.
    """
    print("N:",N)
    print(domain_size)
    print("seeds:",n_seeds)
    n_boot = max(100, n_seeds * 5)
    n_boot = 100

    results = {}
    area = domain_size[0] * domain_size[1]

    #min_s, max_s, npts = sigma_grid_abs
    #sigma_values = np.linspace(min_s, max_s, npts)
    #sigma_values = np.linspace(0.001, 0.3, 250)
    sigma_values = np.concatenate([
        np.linspace(0.05, 0.35, 40, endpoint=False),
        np.linspace(0.35, 0.65, 100, endpoint=False),  # Dense in peak
        np.linspace(0.65, 1.00, 40)
    ])
    if domain_size[0]==300 and N==4000:
        sigma_values = np.linspace(0.35, 0.95, 130) # like 150, 1000 peers
    elif domain_size[0]==150 and N==1000:
        sigma_values = np.linspace(0.35, 0.95, 130)
    elif domain_size[0]==150 and N==4000:
        sigma_values = np.linspace(0.15, 0.60, 100) # like 75, 1000 peers
    elif domain_size[0]==125 and N==1000:
        sigma_values = np.linspace(0.3, 0.85, 120)
    elif domain_size[0]==100 and N==1000:
        sigma_values = np.linspace(0.25, 0.70, 100)
    elif domain_size[0]==75 and N==1000:
        sigma_values = np.linspace(0.15, 0.60, 100)
    print(sigma_values)
    #sigma_values = np.logspace(np.log10(min_s), np.log10(max_s), npts)

    if True:
        print(f"\nTesting N = {N}")
        rho = N / area
        sqrt_rho = np.sqrt(rho)

        # collect M(σ) for each seed
        all_M = []
        seeds = list(range(masterseed+N,masterseed+N+n_seeds))  # deterministic seed list

        for seed in seeds:
            # Generate peers once per seed (use your generate_peers)
            peers = generate_peers(
                n_peers=N,
                domain_size=domain_size,
                scenario=swarm_scenario,
                seed=seed
            )

            Ms = []
            for sigma in sigma_values:
                print("sigma:",sigma)
                dmax = dmax_factor * sigma
                moran_balls = generate_moran_balls(
                    peers=peers,
                    ambient_field_spin=ambient,
                    sigma=sigma,
                    dmax=dmax
                )
                levels = kadanoff_aggregation(
                    moran_balls,
                    levels=8,
                    initial_block_size=block_size,
                    ambient_field_spin=ambient,
                    verbose=False
                )
                # choose primary magnetization consistently: use max across nontrivial levels
                mags = [lvl['magnetization'] for lvl in levels]# if lvl.get('n',0) > 1]
                M_primary = float(np.max(mags)) if len(mags)>0 else 0.0
                Ms.append(M_primary)
            all_M.append(Ms)

        all_M = np.array(all_M)  # shape (n_seeds, n_sigmas)
        M_mean = all_M.mean(axis=0)
        M_std = all_M.std(axis=0)

        # Smooth and find sigma_c with bootstrap
        sigma_c_mean, sigma_c_std, sigma_c_lo, sigma_c_hi = estimate_sigma_c_bootstrap(
            all_M, sigma_values, n_boot=n_boot, window=smoothing_window, poly=smoothing_poly
        )

        # Compute xi distribution via bootstrap samples of sigma_c * sqrt(rho)
        theta_mean = sigma_c_mean * sqrt_rho
        theta_std = sigma_c_std * sqrt_rho
        theta_lo = sigma_c_lo * sqrt_rho
        theta_hi = sigma_c_hi * sqrt_rho

        results[N] = {
            'rho': rho,
            'sqrt_rho': sqrt_rho,
            'sigma_values': sigma_values,
            'M_mean': M_mean,
            'M_std': M_std,
            'sigma_c_mean': sigma_c_mean,
            'sigma_c_std': sigma_c_std,
            'sigma_c_ci': (sigma_c_lo, sigma_c_hi),
            'theta_mean': theta_mean,
            'theta_std': theta_std,
            'theta_ci': (theta_lo, theta_hi),
            'all_M': all_M
        }

        print(f"  rho={rho:.6f}, sqrt_rho={sqrt_rho:.4f}")
        print(f"  sigma_c (mean ± std) = {sigma_c_mean:.4f} ± {sigma_c_std:.4f}")
        print(f"  sigma_c 95% CI = [{sigma_c_lo:.4f}, {sigma_c_hi:.4f}]")
        print(f"  theta_c = sigma_c * sqrt(rho) = {theta_mean:.4f} ± {theta_std:.4f}, CI=[{theta_lo:.4f},{theta_hi:.4f}]")

    var_per_sigma = np.var(results[N]['all_M'], axis=0)
    print("Variance across seeds (per sigma) - min/median/max:", var_per_sigma.min(), np.median(var_per_sigma), var_per_sigma.max())

    if np.allclose(var_per_sigma, 0.0):
        print("ALL SEEDS IDENTICAL: peer generation or downstream pipeline is deterministic across seeds.")
    else:
        print("Seeds vary. Some variability present.")
    return results

def visualize_susceptibility_curve(results, N, save_path=None):
    """
    Visualize the susceptibility curve M(σ) and χ(σ) = dM/dσ for a given N value.
    
    Shows two subplots:
    1. Mean M(σ) curve with 95% confidence interval band
    2. Susceptibility χ(σ) = dM/dσ showing the peak at sigma_c
    
    Args:
        results: dict returned by find_sigma_critical
        N: which N value to visualize
        save_path: optional path to save the figure (e.g., 'susceptibility_N500.png')
    """
    import seaborn as sns
    from scipy.signal import savgol_filter
    
    if N not in results:
        raise ValueError(f"N={N} not found in results. Available: {list(results.keys())}")
    
    data = results[N]
    sigma_values = data['sigma_values']
    M_mean = data['M_mean']
    M_std = data['M_std']
    sigma_c_mean = data['sigma_c_mean']
    sigma_c_lo, sigma_c_hi = data['sigma_c_ci']
    
    # Compute 95% CI for M
    M_ci_lower = M_mean - 1.96 * M_std
    M_ci_upper = M_mean + 1.96 * M_std
    
    # Compute susceptibility χ = dM/dσ with smoothing
    window = 7 if len(sigma_values) >= 7 else len(sigma_values)
    if window % 2 == 0:
        window -= 1
    M_smoothed = savgol_filter(M_mean, window_length=window, polyorder=2)
    chi = np.gradient(M_smoothed, sigma_values)
    
    # Compute CI for chi via bootstrap from all_M
    all_M = data['all_M']  # shape (n_seeds, n_sigmas)
    chi_per_seed = []
    for M_seed in all_M:
        M_s = savgol_filter(M_seed, window_length=window, polyorder=2)
        chi_s = np.gradient(M_s, sigma_values)
        chi_per_seed.append(chi_s)
    chi_per_seed = np.array(chi_per_seed)
    chi_std = chi_per_seed.std(axis=0)
    chi_ci_lower = chi - 1.96 * chi_std
    chi_ci_upper = chi + 1.96 * chi_std
    
    # Set seaborn style
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # ========== Top panel: M(σ) ==========
    ax1.plot(sigma_values, M_mean, linewidth=2.5, color='#2E86AB', label='Mean M(σ)')
    ax1.fill_between(sigma_values, M_ci_lower, M_ci_upper, 
                     alpha=0.3, color='#2E86AB', label='95% CI')
    
    M_at_sigma_c = np.interp(sigma_c_mean, sigma_values, M_mean)
    ax1.axvline(sigma_c_mean, color='#A23B72', linestyle='--', linewidth=2, 
               label=f'σc = {sigma_c_mean:.4f}')
    ax1.errorbar(sigma_c_mean, M_at_sigma_c, 
                xerr=[[sigma_c_mean - sigma_c_lo], [sigma_c_hi - sigma_c_mean]],
                fmt='o', color='#A23B72', markersize=8, capsize=5, capthick=2,
                label=f'σc 95% CI [{sigma_c_lo:.4f}, {sigma_c_hi:.4f}]')
    
    ax1.set_ylabel('Magnetization |M|', fontsize=13)
    ax1.set_title(f'Magnetization and Susceptibility for N={N} (ρ={data["rho"]:.4f})', 
                 fontsize=15, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # ========== Bottom panel: χ(σ) = dM/dσ ==========
    ax2.plot(sigma_values, chi, linewidth=2.5, color='#F18F01', label='χ(σ) = dM/dσ')
    ax2.fill_between(sigma_values, chi_ci_lower, chi_ci_upper, 
                     alpha=0.3, color='#F18F01', label='95% CI')
    
    # Mark the peak
    chi_at_sigma_c = np.interp(sigma_c_mean, sigma_values, chi)
    ax2.axvline(sigma_c_mean, color='#A23B72', linestyle='--', linewidth=2, 
               label=f'Peak at σc = {sigma_c_mean:.4f}')
    ax2.plot(sigma_c_mean, chi_at_sigma_c, 'o', color='#A23B72', markersize=8)
    
    ax2.set_xlabel('Coarse-graining scale σ', fontsize=13)
    ax2.set_ylabel('Susceptibility χ = dM/dσ', fontsize=13)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()
    
    return fig, (ax1, ax2)

def save_results(results, N, filepath):
    """
    Save results[N] to a file using numpy.
    
    Args:
        results: dict returned by find_sigma_critical
        N: which N value to save
        filepath: path to save the results (e.g., 'results_N500.npz')
    """
    if N not in results:
        raise ValueError(f"N={N} not found in results. Available: {list(results.keys())}")
    
    data = results[N]
    np.savez(filepath, **data)
    print(f"Results for N={N} saved to {filepath}")


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()
    
    # Set global behavioral parameters from args
    BEHAVIOR_PARAMS['legitimate']['alpha'] = (args.alpha_legit, args.alpha_legit_std)
    BEHAVIOR_PARAMS['legitimate']['kurtosis'] = (args.kurtosis_legit, args.kurtosis_legit_std)
    BEHAVIOR_PARAMS['legitimate']['hurst'] = (args.hurst_legit, args.hurst_legit_std)
    
    BEHAVIOR_PARAMS['anomalous']['alpha'] = (args.alpha_anom, args.alpha_anom_std)
    BEHAVIOR_PARAMS['anomalous']['kurtosis'] = (args.kurtosis_anom, args.kurtosis_anom_std)
    BEHAVIOR_PARAMS['anomalous']['hurst'] = (args.hurst_anom, args.hurst_anom_std)
    
    # Set global weighting coefficients
    C0 = args.c0
    C1 = args.c1
    
    # Set legitimate centers and stds for anomaly score computation
    ALPHA_LEGIT_CENTER = args.alpha_legit
    KURTOSIS_LEGIT_CENTER = args.kurtosis_legit
    HURST_LEGIT_CENTER = args.hurst_legit
    ALPHA_LEGIT_STD = args.alpha_legit_std
    KURTOSIS_LEGIT_STD = args.kurtosis_legit_std
    HURST_LEGIT_STD = args.hurst_legit_std
    
    # Validate c0 + c1 <= 1
    if C0 + C1 > 1.0:
        raise ValueError(f"c0 + c1 must be <= 1.0, got c0={C0}, c1={C1}, sum={C0+C1}")
    
    print("="*70)
    print("BEHAVIORAL PARAMETER CONFIGURATION")
    print("="*70)
    print("\nLegitimate peers:")
    print(f"  Alpha:    μ={args.alpha_legit:.3f}, σ={args.alpha_legit_std:.3f}")
    print(f"  Kurtosis: μ={args.kurtosis_legit:.3f}, σ={args.kurtosis_legit_std:.3f}")
    print(f"  Hurst:    μ={args.hurst_legit:.3f}, σ={args.hurst_legit_std:.3f}")
    
    print("\nAnomalous peers:")
    print(f"  Alpha:    μ={args.alpha_anom:.3f}, σ={args.alpha_anom_std:.3f}")
    print(f"  Kurtosis: μ={args.kurtosis_anom:.3f}, σ={args.kurtosis_anom_std:.3f}")
    print(f"  Hurst:    μ={args.hurst_anom:.3f}, σ={args.hurst_anom_std:.3f}")
    
    print("\nWeighting coefficients:")
    print(f"  c0 (alpha):    {C0:.3f}")
    print(f"  c1 (kurtosis): {C1:.3f}")
    print(f"  c2 (hurst):    {1.0-C0-C1:.3f}")
    
    print("\nSeparation metrics:")
    print(f"  Δα = {abs(args.alpha_legit - args.alpha_anom):.3f}")
    print(f"  Δκ = {abs(args.kurtosis_legit - args.kurtosis_anom):.3f}")
    print(f"  ΔH = {abs(args.hurst_legit - args.hurst_anom):.3f}")
    print("="*70)
    
    # Set simulation parameters
    ambient = args.ambient
    N = args.N
    scenario = args.scenario
    domain_size = (args.domain_size, args.domain_size)
    n_seeds = args.n_seeds
    
    masterseed = int(hash(scenario) % 1E09)
    print(f"\nSimulation parameters:")
    print(f"  N = {N}")
    print(f"  Domain = {domain_size[0]}×{domain_size[1]}")
    print(f"  Scenario = {scenario}")
    print(f"  Seeds = {n_seeds}")
    print(f"  Ambient = {ambient}")
    print(f"  Master seed = {masterseed}")
    print("="*70 + "\n")
    
    # Run simulation
    res = find_sigma_critical(N, n_seeds=n_seeds, domain_size=domain_size, swarm_scenario=scenario)
    
    # Generate output filename with parameter hash for uniqueness
    param_str = f"a{args.alpha_legit:.2f}_{args.alpha_anom:.2f}_" \
                f"k{args.kurtosis_legit:.2f}_{args.kurtosis_anom:.2f}_" \
                f"h{args.hurst_legit:.3f}_{args.hurst_anom:.3f}_" \
                f"c{C0:.2f}_{C1:.2f}"
    
    output_base = f'results_3SIGMAS_{scenario}_{N}_{domain_size[0]}_{param_str}'
    
    # Save results and visualize
    save_results(res, N, f'{output_base}.npz')
    visualize_susceptibility_curve(res, N, save_path=f'{output_base}.png')
    
    print("\n" + "="*70)
    print("SIMULATION COMPLETE")
    print("="*70)
