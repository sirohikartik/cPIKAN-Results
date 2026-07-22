import os
import re
from scipy import stats
import numpy as np

def extract_latencies(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
        # Find the line starting with "Individual Times:"
        match = re.search(r"Individual Times: ([\d\.,\s]+)", content)
        if match:
            times_str = match.group(1).strip()
            # Split by comma and convert to float
            times = [float(t) for t in times_str.split(',') if t.strip()]
            return times
    return []

def main():
    configs = [
        "diffusion_32",
        "diffusion_64",
        "diffusion_128",
        "reaction_diffusion_32"
    ]
    
    results = {}
    
    for config in configs:
        baseline_file = f"logs/ablation_{config}_baseline.txt"
        scalar_file = f"logs/ablation_{config}_scalar_recurrence.txt"
        
        if not os.path.exists(baseline_file) or not os.path.exists(scalar_file):
            print(f"Skipping {config}: one or more files not found.")
            continue
            
        baseline_latencies = extract_latencies(baseline_file)
        scalar_latencies = extract_latencies(scalar_file)
        
        if len(baseline_latencies) == 0 or len(scalar_latencies) == 0:
            print(f"Skipping {config}: could not extract latencies.")
            continue
            
        # Perform independent t-test
        t_stat, p_val = stats.ttest_ind(baseline_latencies, scalar_latencies)
        
        results[config] = {
            'baseline_mean': np.mean(baseline_latencies),
            'scalar_mean': np.mean(scalar_latencies),
            'p_value': p_val,
            't_stat': t_stat,
            'baseline_std': np.std(baseline_latencies),
            'scalar_std': np.std(scalar_latencies)
        }

    print(f"{'Config':<25} | {'Baseline Mean':<15} | {'Scalar Mean':<15} | {'p-value':<10} | {'Significant?'}")
    print("-" * 85)
    for config, data in results.items():
        sig = "Yes" if data['p_value'] < 0.05 else "No"
        print(f"{config:<25} | {data['baseline_mean']:<15.6f} | {data['scalar_mean']:<15.6f} | {data['p_value']:<10.6f} | {sig}")

if __name__ == "__main__":
    main()
