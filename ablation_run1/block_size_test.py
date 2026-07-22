import os
import re
from scipy import stats
import numpy as np
from itertools import combinations

def extract_latencies(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
        match = re.search(r"Individual Times: ([\d\.,\s]+)", content)
        if match:
            times_str = match.group(1).strip()
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
    block_sizes = ["64", "128", "256", "1024"]
    
    for config in configs:
        print(f"\n{'='*60}")
        print(f"Analyzing Block Sizes for: {config}")
        print(f"{'='*60}")
        
        data = {}
        for size in block_sizes:
            file_path = f"logs/ablation_{config}_block_size_{size}.txt"
            if os.path.exists(file_path):
                latencies = extract_latencies(file_path)
                if latencies:
                    data[size] = latencies
        
        if not data:
            print(f"No data found for {config}")
            continue

        # Calculate means and identify the best (min latency)
        means = {size: np.mean(lats) for size, lats in data.items()}
        best_size = min(means, key=means.get)
        best_mean = means[best_size]
        
        print(f"Means: { {k: f'{v:.6f}' for k,v in means.items()} }")
        print(f"Best candidate: Block Size {best_size} (Mean: {best_mean:.6f})")
        print("-" * 60)
        print(f"{'Block Size':<15} | {'Mean':<15} | {'p-value (vs best)':<20} | {'Significant?'}")
        print("-" * 60)
        
        for size in block_sizes:
            if size not in data:
                continue
                
            if size == best_size:
                print(f"{size:<15} | {means[size]:<15.6f} | {'-':<20} | {'Optimal'}")
                continue
            
            # T-test against the best one
            t_stat, p_val = stats.ttest_ind(data[size], data[best_size])
            # We use a Bonferroni correction roughly by using 0.05 / (number of comparisons)
            # but let's just stick to 0.05 for simplicity unless told otherwise, 
            # or maybe 0.01 to be more conservative.
            sig = "Yes" if p_val < 0.05 else "No"
            print(f"{size:<15} | {means[size]:<15.6f} | {p_val:<20.6f} | {sig}")

if __name__ == "__main__":
    main()
