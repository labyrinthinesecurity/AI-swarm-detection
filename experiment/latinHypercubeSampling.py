import csv
import os
import sys
from datetime import datetime

# Import the susceptibility module to access its functions and modify its globals
import susceptibility as susc

def run_parameter_sweep(scenarios, output_csv, process_id="main"):
    """
    Run find_sigma_critical for multiple parameter scenarios with crash recovery.
    
    Args:
        scenarios: list of dicts, each containing the 14 behavioral params + simulation params
        output_csv: path to CSV file for storing results
        process_id: identifier for this process (for logging)
    """
    
    # Define CSV columns
    csv_columns = [
        'scenario_id', 'timestamp',
        # Legitimate params
        'alpha_legit', 'alpha_legit_std',
        'kurtosis_legit', 'kurtosis_legit_std',
        'hurst_legit', 'hurst_legit_std',
        # Anomalous params
        'alpha_anom', 'alpha_anom_std',
        'kurtosis_anom', 'kurtosis_anom_std',
        'hurst_anom', 'hurst_anom_std',
        # Weights
        'c0', 'c1',
        # Simulation params
        'N', 'domain_size', 'scenario', 'n_seeds', 'ambient',
        # Separation metrics (computed)
        'delta_alpha', 'delta_kurtosis', 'delta_hurst',
        # Results
        'rho', 'sqrt_rho',
        'sigma_c_mean', 'sigma_c_std', 'sigma_c_lo', 'sigma_c_hi',
        'theta_c_mean', 'theta_c_std', 'theta_c_lo', 'theta_c_hi',
        # Metadata
        'status', 'error_msg'
    ]
    
    # Check which scenarios are already completed
    completed_ids = set()
    if os.path.exists(output_csv):
        print(f"[{process_id}] Found existing CSV: {output_csv}")
        with open(output_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['status'] == 'completed':
                    completed_ids.add(int(row['scenario_id']))
        print(f"[{process_id}] Already completed: {len(completed_ids)} scenarios")
    else:
        # Create new CSV with header
        print(f"[{process_id}] Creating new CSV: {output_csv}")
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_columns)
            writer.writeheader()
    
    # Process scenarios
    total = len(scenarios)
    remaining = [s for s in scenarios if s['scenario_id'] not in completed_ids]
    
    print(f"[{process_id}] Total scenarios: {total}")
    print(f"[{process_id}] Remaining: {len(remaining)}")
    print(f"[{process_id}] Starting sweep...\n")
    
    for idx, params in enumerate(remaining, 1):
        scenario_id = params['scenario_id']
        
        print("="*70)
        print(f"[{process_id}] Scenario {scenario_id}/{total} (Remaining: {len(remaining)-idx+1})")
        print("="*70)
        
        # Prepare row data
        row = {
            'scenario_id': scenario_id,
            'timestamp': datetime.now().isoformat(),
            'alpha_legit': params['alpha_legit'],
            'alpha_legit_std': params['alpha_legit_std'],
            'kurtosis_legit': params['kurtosis_legit'],
            'kurtosis_legit_std': params['kurtosis_legit_std'],
            'hurst_legit': params['hurst_legit'],
            'hurst_legit_std': params['hurst_legit_std'],
            'alpha_anom': params['alpha_anom'],
            'alpha_anom_std': params['alpha_anom_std'],
            'kurtosis_anom': params['kurtosis_anom'],
            'kurtosis_anom_std': params['kurtosis_anom_std'],
            'hurst_anom': params['hurst_anom'],
            'hurst_anom_std': params['hurst_anom_std'],
            'c0': params['c0'],
            'c1': params['c1'],
            'N': params['N'],
            'domain_size': params['domain_size'],
            'scenario': params['scenario'],
            'n_seeds': params['n_seeds'],
            'ambient': params['ambient'],
            'delta_alpha': abs(params['alpha_legit'] - params['alpha_anom']),
            'delta_kurtosis': abs(params['kurtosis_legit'] - params['kurtosis_anom']),
            'delta_hurst': abs(params['hurst_legit'] - params['hurst_anom']),
            'status': 'running',
            'error_msg': ''
        }
        
        try:
            # Set global parameters in the susceptibility module
            susc.BEHAVIOR_PARAMS['legitimate']['alpha'] = (params['alpha_legit'], params['alpha_legit_std'])
            susc.BEHAVIOR_PARAMS['legitimate']['kurtosis'] = (params['kurtosis_legit'], params['kurtosis_legit_std'])
            susc.BEHAVIOR_PARAMS['legitimate']['hurst'] = (params['hurst_legit'], params['hurst_legit_std'])

            susc.BEHAVIOR_PARAMS['anomalous']['alpha'] = (params['alpha_anom'], params['alpha_anom_std'])
            susc.BEHAVIOR_PARAMS['anomalous']['kurtosis'] = (params['kurtosis_anom'], params['kurtosis_anom_std'])
            susc.BEHAVIOR_PARAMS['anomalous']['hurst'] = (params['hurst_anom'], params['hurst_anom_std'])

            susc.C0 = params['c0']
            susc.C1 = params['c1']

            susc.ALPHA_LEGIT_CENTER = params['alpha_legit']
            susc.KURTOSIS_LEGIT_CENTER = params['kurtosis_legit']
            susc.HURST_LEGIT_CENTER = params['hurst_legit']
            susc.ALPHA_LEGIT_STD = params['alpha_legit_std']
            susc.KURTOSIS_LEGIT_STD = params['kurtosis_legit_std']
            susc.HURST_LEGIT_STD = params['hurst_legit_std']

            # Set masterseed and ambient (required by find_sigma_critical)
            susc.masterseed = int(hash(params['scenario']) % 1E09)
            susc.ambient = params['ambient']

            print(f"Δα={row['delta_alpha']:.3f}, Δκ={row['delta_kurtosis']:.3f}, ΔH={row['delta_hurst']:.3f}")

            # Run simulation
            domain_size = (params['domain_size'], params['domain_size'])
            res = susc.find_sigma_critical(
                N=params['N'],
                n_seeds=params['n_seeds'],
                domain_size=domain_size,
                swarm_scenario=params['scenario']
            )
            
            # Extract results
            N = params['N']
            if N in res:
                r = res[N]
                row['rho'] = r['rho']
                row['sqrt_rho'] = r['sqrt_rho']
                row['sigma_c_mean'] = r['sigma_c_mean']
                row['sigma_c_std'] = r['sigma_c_std']
                row['sigma_c_lo'] = r['sigma_c_ci'][0]
                row['sigma_c_hi'] = r['sigma_c_ci'][1]
                row['theta_c_mean'] = r['theta_mean']
                row['theta_c_std'] = r['theta_std']
                row['theta_c_lo'] = r['theta_ci'][0]
                row['theta_c_hi'] = r['theta_ci'][1]
                row['status'] = 'completed'
                
                print(f"✓ theta_c = {row['theta_c_mean']:.4f} ± {row['theta_c_std']:.4f}")
            else:
                row['status'] = 'failed'
                row['error_msg'] = f'N={N} not in results'
                print(f"✗ Failed: N not in results")
                
        except Exception as e:
            row['status'] = 'failed'
            row['error_msg'] = str(e)
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
        
        # Append to CSV immediately (crash recovery)
        with open(output_csv, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_columns)
            writer.writerow(row)
        
        print(f"[{process_id}] Saved to {output_csv}\n")
    
    print("="*70)
    print(f"[{process_id}] SWEEP COMPLETE")
    print(f"[{process_id}] Results saved to: {output_csv}")
    print("="*70)


def generate_lhs_scenarios(n_samples=500, seed=42):
    """
    Generate Latin Hypercube Sampling scenarios for 14-dimensional parameter space.
    
    Returns:
        list of dicts with all 14 parameters + simulation settings
    """
    from scipy.stats import qmc
    
    # Define parameter bounds [min, max]
    bounds = {
        'alpha_legit': [2.0, 4.0],
        'alpha_legit_std': [0.2, 0.8],
        'kurtosis_legit': [2.0, 8.0],
        'kurtosis_legit_std': [0.5, 2.5],
        'hurst_legit': [0.4, 0.7],
        'hurst_legit_std': [0.03, 0.12],
        'alpha_anom': [1.2, 4.0],  # Overlaps with legit
        'alpha_anom_std': [0.2, 0.8],
        'kurtosis_anom': [2.0, 20.0],  # Overlaps with legit
        'kurtosis_anom_std': [0.5, 4.0],
        'hurst_anom': [0.4, 0.85],  # Overlaps with legit
        'hurst_anom_std': [0.03, 0.15],
        'c0': [0.05, 0.6],
        'c1': [0.05, 0.6]
    }
    
    param_names = list(bounds.keys())
    lower_bounds = [bounds[p][0] for p in param_names]
    upper_bounds = [bounds[p][1] for p in param_names]
    
    # Generate LHS samples
    sampler = qmc.LatinHypercube(d=14, seed=seed)
    samples = sampler.random(n=n_samples)
    samples_scaled = qmc.scale(samples, lower_bounds, upper_bounds)
    
    # Convert to scenario dicts
    scenarios = []
    for i, sample in enumerate(samples_scaled):
        params = dict(zip(param_names, sample))
        
        # Ensure c0 + c1 <= 1
        if params['c0'] + params['c1'] > 1.0:
            # Rescale to satisfy constraint
            total = params['c0'] + params['c1']
            params['c0'] = params['c0'] / total * 0.95
            params['c1'] = params['c1'] / total * 0.95
        
        # Add simulation parameters (fixed for all scenarios)
        params['scenario_id'] = i
        params['N'] = 562  # 75x75 domain at rho=0.1
        params['domain_size'] = 75
        params['scenario'] = 'RingSwarm'
        params['n_seeds'] = 50  # Reduced for speed
        params['ambient'] = 0.75
        
        scenarios.append(params)
    
    return scenarios


# Example usage:
if __name__ == "__main__":
    import sys
    import subprocess
    import time

    # Check if this is a subprocess call with odd/even argument
    if len(sys.argv) >= 2 and sys.argv[1].lower() in ['odd', 'even']:
        mode = sys.argv[1].lower()

        # Generate scenarios
        print("Generating LHS scenarios...")
        all_scenarios = generate_lhs_scenarios(n_samples=500, seed=42)

        # Split into odd/even
        if mode == 'odd':
            scenarios = [s for s in all_scenarios if s['scenario_id'] % 2 == 1]
            output_csv = 'results_odd.csv'
            process_id = 'ODD'
        elif mode == 'even':
            scenarios = [s for s in all_scenarios if s['scenario_id'] % 2 == 0]
            output_csv = 'results_even.csv'
            process_id = 'EVEN'

        print(f"Process: {process_id}")
        print(f"Scenarios: {len(scenarios)}")
        print(f"Output: {output_csv}\n")

        # Run sweep
        run_parameter_sweep(scenarios, output_csv, process_id)

    else:
        # Master process: launch both odd and even subprocesses
        print("="*70)
        print("LAUNCHING PARALLEL ODD/EVEN SWEEP PROCESSES")
        print("="*70)

        # Launch odd process
        print("\n[MASTER] Starting ODD process...")
        odd_process = subprocess.Popen(
            [sys.executable, __file__, 'odd'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        # Launch even process
        print("[MASTER] Starting EVEN process...")
        even_process = subprocess.Popen(
            [sys.executable, __file__, 'even'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        print("[MASTER] Both processes launched. Monitoring outputs...\n")
        print("="*70)

        # Monitor both processes
        processes = {
            'ODD': odd_process,
            'EVEN': even_process
        }

        # Wait for both to complete and stream their output
        while processes:
            for name, proc in list(processes.items()):
                if proc.poll() is not None:
                    # Process finished
                    remaining_output = proc.stdout.read()
                    if remaining_output:
                        for line in remaining_output.splitlines():
                            print(f"[{name}] {line}")

                    returncode = proc.returncode
                    print(f"\n[MASTER] {name} process completed with return code {returncode}")
                    del processes[name]
                else:
                    # Read available output
                    line = proc.stdout.readline()
                    if line:
                        print(f"[{name}] {line.rstrip()}")

            time.sleep(0.1)

        print("\n" + "="*70)
        print("[MASTER] ALL PROCESSES COMPLETE")
        print("="*70)
        print("\nResults saved to:")
        print("  - results_odd.csv")
        print("  - results_even.csv")
