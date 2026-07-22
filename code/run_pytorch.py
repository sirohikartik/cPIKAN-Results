import os
import torch
import numpy as np
import time
import argparse

class ChebyKANLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, deg_max):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(in_dim, out_dim, deg_max), requires_grad=False)
        self.deg_max = deg_max

    def forward(self, x):
        x_act = torch.tanh(x)
        out = torch.matmul(torch.ones_like(x_act), self.weight[:, :, 0]) + \
              torch.matmul(x_act, self.weight[:, :, 1])
        T_prev2 = torch.ones_like(x_act)
        T_prev1 = x_act
        for d in range(2, self.deg_max + 1):
            T_cur = 2 * x_act * T_prev1 - T_prev2
            out += torch.matmul(T_cur, self.weight[:, :, d])
            T_prev2, T_prev1 = T_prev1, T_cur
        return out

class ChebyKAN(torch.nn.Module):
    def __init__(self, layers_cfg, degree=5):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for i in range(len(layers_cfg)-1):
            self.layers.append(ChebyKANLayer(layers_cfg[i], layers_cfg[i+1], degree))

    def forward(self, x):
        h = x
        for layer in self.layers:
            h = layer(h)
        return h

def load_weights_torch(model, weights_dir):
    layer_files = sorted([f for f in os.listdir(weights_dir) if f.startswith("layer") and f.endswith("_W.npy")],
                         key=lambda x: int(x.split('_')[0].replace("layer", "")))
    for i, path_name in enumerate(layer_files):
        W = np.load(os.path.join(weights_dir, path_name))
        model.layers[i].weight.data = torch.from_numpy(W).float()
        model.layers[i].deg_max = W.shape[-1] - 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--M", type=int, default=1001*1001)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--mode", type=str, default="eager", choices=["eager", "script", "compile"], help="PyTorch mode")
    parser.add_argument("--output", help="Path to save final predictions")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)

    first_w = np.load(os.path.join(args.weights, "layer0_W.npy"))
    in_dim, out_dim, deg_plus_one = first_w.shape
    layer_files = sorted([f for f in os.listdir(args.weights) if f.startswith("layer") and f.endswith("_W.npy")],
                         key=lambda x: int(x.split('_')[0].replace("layer", "")))
    dims = [in_dim]
    for f in layer_files:
        w = np.load(os.path.join(args.weights, f))
        dims.append(w.shape[1])
    
    model = ChebyKAN(dims, degree=deg_plus_one-1).eval()
    load_weights_torch(model, args.weights)

    if args.mode == "script":
        model = torch.jit.script(model)
    elif args.mode == "compile":
        model = torch.compile(model)

    n = int(np.sqrt(args.M))
    if n * n == args.M:
        x = np.linspace(-1, 1, n)
        y = np.linspace(-1, 1, n)
        xv, yv = np.meshgrid(x, y)
        xy = torch.tensor(np.stack([xv.ravel(), yv.ravel()], axis=1), dtype=torch.float32)
    else:
        xy = torch.linspace(0, 1, args.M * 2).reshape(args.M, 2)

    BATCH_SIZE = 1024 * 16
    def run_full_forward(input_tensor):
        results = []
        for i in range(0, input_tensor.shape[0], BATCH_SIZE):
            batch = input_tensor[i : i + BATCH_SIZE]
            results.append(model(batch))
        return torch.cat(results, dim=0)

    with torch.no_grad():
        for _ in range(args.warmup):
            _ = run_full_forward(xy)
        total_times = []
        last_out = None
        for _ in range(args.reps):
            t0 = time.perf_counter()
            last_out = run_full_forward(xy)
            t1 = time.perf_counter()
            total_times.append(t1 - t0)

    mean_total = np.mean(total_times)
    std_total = np.std(total_times)
    print(f"\n--- Results ({args.mode}) ---")
    print(f"Overall Pipeline Time: {mean_total:.6f}s +/- {std_total:.6f}s")
    print(f"Throughput: {args.M / mean_total:.2f} points/sec")
    print(f"Latency per point: {(mean_total / args.M) * 1e6:.4f} us")

    if args.output and last_out is not None:
        last_out.numpy().astype(np.float32).tofile(args.output)

if __name__ == "__main__":
    main()
