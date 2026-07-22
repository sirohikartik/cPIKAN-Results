import os
import numpy as np
import time
import argparse

def cheby_kan_layer(x_in, W_by_degree):
    """
    x_in : [batch, in_dim]
    W_by_degree : [deg_max, in_dim, out_dim]
    """
    # Tanh
    x = np.tanh(x_in)
    
    # Chebyshev recurrence
    deg_max = W_by_degree.shape[0]
    T_prev2 = np.ones_like(x)              # T0
    T_prev1 = x                            # T1
    
    # T0 and T1 matmuls
    out = T_prev2 @ W_by_degree[0] + T_prev1 @ W_by_degree[1]
    
    for d in range(2, deg_max):
        T_cur = 2 * x * T_prev1 - T_prev2
        out += T_cur @ W_by_degree[d]
        T_prev2, T_prev1 = T_prev1, T_cur
        
    return out

def load_weights(weights_dir):
    weights = []
    # Find all layerX_W.npy files and sort them by X
    layer_files = sorted([f for f in os.listdir(weights_dir) if f.startswith("layer") and f.endswith("_W.npy")],
                         key=lambda x: int(x.split('_')[0].replace("layer", "")))
    
    for path_name in layer_files:
        path = os.path.join(weights_dir, path_name)
        W = np.load(path).astype(np.float32)
        # [in_dim, out_dim, deg_max] -> [deg_max, in_dim, out_dim]
        W_by_degree = np.ascontiguousarray(W.transpose(2, 0, 1))
        weights.append(W_by_degree)
    return weights

def forward(x, weights):
    h = x.astype(np.float32)
    for W in weights:
        h = cheby_kan_layer(h, W)
    return h

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Path to trained weights directory")
    parser.add_argument("--M", type=int, default=1001*1001, help="Number of inference points")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--reps", type=int, default=20, help="Benchmark iterations")
    parser.add_argument("--output", help="Path to save final predictions")
    args = parser.parse_args()

    weights = load_weights(args.weights)

    # Generate input data
    n = int(np.sqrt(args.M))
    if n * n == args.M:
        x = np.linspace(-1, 1, n)
        y = np.linspace(-1, 1, n)
        xv, yv = np.meshgrid(x, y)
        xy = np.stack([xv.ravel(), yv.ravel()], axis=1).astype(np.float32)
    else:
        xy = np.linspace(0, 1, args.M * 2).reshape(args.M, 2).astype(np.float32)

    # Warmup
    for _ in range(args.warmup):
        _ = forward(xy, weights)

    # Measurement
    total_times = []
    last_out = None
    for _ in range(args.reps):
        t_start = time.perf_counter()
        last_out = forward(xy, weights)
        t_end = time.perf_counter()
        total_times.append(t_end - t_start)

    mean_total = np.mean(total_times)
    std_total = np.std(total_times)

    print(f"\n--- Results ---")
    print(f"Overall Pipeline Time: {mean_total:.6f}s +/- {std_total:.6f}s")
    print(f"Throughput: {args.M / mean_total:.2f} points/sec")
    print(f"Latency per point: {(mean_total / args.M) * 1e6:.4f} us")

    if args.output:
        last_out.astype(np.float32).tofile(args.output)

if __name__ == "__main__":
    main()
