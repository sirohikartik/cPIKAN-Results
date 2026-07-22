import os
import subprocess
import re
import numpy as np
import json
import sys
import time
import random
import psutil
from typing import List, Tuple, Dict

# ----------------------------------------------------------------------------
# CONFIG -- edit these to trim runtime if needed before the deadline.
# Each config below produces one subprocess call to ./ablate per (pde, width).
# Total calls per full pass = len(WIDTHS_TO_RUN) * len(ABLATION_CONFIGS).
# With cooldown, each call costs ~180s regardless of its own runtime, so
# estimate total wall time as roughly (num_calls * 180s) at minimum.
# ----------------------------------------------------------------------------

ABLATE_BINARY = "./ablate"                    # normal build, full -O3 auto-vectorization
ABLATE_BINARY_SCALAR_SAFE = "./ablate_scalar_safe"  # -fno-tree-vectorize -fno-slp-vectorize build
                                                      # ONLY used for --recurrence scalar, so that
                                                      # ablation is guaranteed truly scalar. Using this
                                                      # build for anything else (including baseline)
                                                      # would unfairly de-optimize unrelated loops like
                                                      # simd_tanh's, which has no explicit intrinsics and
                                                      # normally relies on the auto-vectorizer.
M = 1001 * 1001
WARMUP, REPS = 5, 15
BASELINE_THREADS = 8
COOLDOWN_SECONDS = 180

# (pde, width) combos to run every ablation config against.
WIDTHS_TO_RUN = [("diffusion", "32"), ("diffusion", "64"), ("diffusion", "128"), ("reaction_diffusion", "32")]

# Block sizes to sweep (512 is the original engine's default, already
# covered by the "baseline" config below -- no need to repeat it here).
BLOCK_SIZES_TO_SWEEP = [64, 128, 256, 1024]

# Thread counts to sweep (8 is the default, already covered by "baseline").
THREAD_COUNTS_TO_SWEEP = [1, 2, 4, 6]


def build_ablation_configs() -> List[Dict]:
    """
    Builds the full list of ablation configurations. Each is a single-flag
    change from the baseline (all optimizations on), except "baseline"
    itself, which anchors all the deltas.
    """
    configs = [
        {"label": "baseline", "recurrence": "simd", "matmul": "blas", "block_size": 512, "threads": BASELINE_THREADS},
        {"label": "scalar_recurrence", "recurrence": "scalar", "matmul": "blas", "block_size": 512, "threads": BASELINE_THREADS},
        {"label": "naive_matmul", "recurrence": "simd", "matmul": "naive", "block_size": 512, "threads": BASELINE_THREADS},
    ]
    for bs in BLOCK_SIZES_TO_SWEEP:
        configs.append({"label": f"block_size_{bs}", "recurrence": "simd", "matmul": "blas", "block_size": bs, "threads": BASELINE_THREADS})
    for th in THREAD_COUNTS_TO_SWEEP:
        configs.append({"label": f"threads_{th}", "recurrence": "simd", "matmul": "blas", "block_size": 512, "threads": th})
    return configs


# ----------------------------------------------------------------------------
# Shared helpers (same as run_benchmarks_fixed.py)
# ----------------------------------------------------------------------------

def get_cpu_temp():
    """
    Attempt to get CPU temperature.
    On Linux, psutil.sensors_temperatures() usually works out of the box.
    On macOS (Apple Silicon), we use `macmon pipe -s 1`, which reads real
    numeric CPU/GPU temps via a private Apple API WITHOUT requiring sudo.
    Install with: brew install macmon
    """
    try:
        temps = psutil.sensors_temperatures()
        if temps:
            for name, entries in temps.items():
                if entries:
                    return f"{entries[0].current:.1f}C ({name})"
    except AttributeError:
        if sys.platform == "darwin":
            try:
                result = subprocess.run(
                    ["macmon", "pipe", "-s", "1"],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0 and result.stdout.strip():
                    data = json.loads(result.stdout.strip().splitlines()[-1])
                    cpu_temp = data.get("temp", {}).get("cpu_temp_avg")
                    if cpu_temp is not None:
                        return f"{cpu_temp:.1f}C (CPU avg)"
                    return "N/A (macmon ran but no cpu_temp_avg field found)"
                else:
                    return "N/A (macmon failed — is it installed? `brew install macmon`)"
            except FileNotFoundError:
                return "N/A (macmon not installed — run: brew install macmon)"
            except subprocess.TimeoutExpired:
                return "N/A (macmon timed out)"
            except (json.JSONDecodeError, IndexError):
                return "N/A (could not parse macmon JSON output)"
            except Exception as e:
                return f"N/A (macmon error: {e})"
    except Exception as e:
        return f"N/A (error: {e})"
    return "N/A (no sensors exposed on this OS without elevated perms)"


def get_system_load():
    """Returns current CPU utilization as a proxy for thermal state."""
    return psutil.cpu_percent(interval=0.1)


def parse_output(output: str):
    if not output: return None, None, None
    mean_match = re.search(r"Overall Pipeline Time(?: \(Estimated for M=\d+\))?: ([\d.]+)s \+/- ([\d.]+)s", output)
    times_match = re.search(r"Individual Times: ([\d.,\s]+)", output)
    if not mean_match: return None, None, None
    mean = float(mean_match.group(1))
    tool_std = float(mean_match.group(2))
    times = []
    if times_match:
        times = [float(t) for t in times_match.group(1).split(",") if t.strip()]
    return mean, tool_std, times


def get_exact_solution(pde, M):
    n = int(np.sqrt(M))
    if n * n == M:
        x = np.linspace(-1, 1, n)
        t = np.linspace(-1, 1, n)
        xv, tv = np.meshgrid(x, t)
        coords_x, coords_t = xv.ravel(), tv.ravel()
    else:
        coords_x = np.linspace(0, 1, M)
        coords_t = np.linspace(0, 1, M)
    if pde == "diffusion":
        D = 0.1
        u_exact = np.sin(np.pi * coords_x) * np.exp(-(np.pi**2) * D * coords_t)
    elif pde == "reaction_diffusion":
        u_exact = (np.sin(6.0 * coords_x)**3) * np.exp(-coords_t)
    else:
        raise ValueError(f"Unknown PDE: {pde}")
    return u_exact.astype(np.float32)


def postprocess_prediction(pred_path, pde, M, label):
    npy_path = os.path.splitext(pred_path)[0] + ".npy"

    if not os.path.exists(pred_path):
        print(f"[{label}] WARNING: prediction file not found at {pred_path}, skipping npy save + L2 calc.")
        return np.nan, None

    u_pred = np.fromfile(pred_path, dtype=np.float32).flatten()
    if u_pred.size == 0:
        print(f"[{label}] WARNING: prediction file at {pred_path} is empty, skipping npy save + L2 calc.")
        return np.nan, None

    np.save(npy_path, u_pred)

    u_exact = get_exact_solution(pde, M).flatten()
    min_len = min(len(u_pred), len(u_exact))
    norm_diff = np.linalg.norm(u_pred[:min_len] - u_exact[:min_len])
    norm_exact = np.linalg.norm(u_exact[:min_len])
    l2 = norm_diff / norm_exact if norm_exact > 1e-12 else np.nan

    print(f"[{label}] Saved predictions to {npy_path} | Relative L2 error: {l2:.6e}")
    return l2, npy_path


def run_benchmark(cmd: List[str], env: dict, log_path: str = None):
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w") as f:
            f.write(f"COMMAND: {' '.join(cmd)}\n")
            f.write(f"ENV OMP_NUM_THREADS: {env.get('OMP_NUM_THREADS')}\n")
            f.write("STDOUT:\n" + result.stdout + "\n\nSTDERR:\n" + result.stderr)

    if result.returncode != 0:
        print(f"CRITICAL ERROR: Command failed with return code {result.returncode}")
        print(f"STDERR: {result.stderr}")
        return None

    print(result.stdout)
    return result.stdout


def append_cpu_temp_log(log_path, start_load, end_load, start_temp, end_temp):
    if not log_path:
        return
    with open(log_path, "a") as f:
        f.write("\n\n--- CPU / THERMAL ---\n")
        f.write(f"Start CPU load: {start_load}%\n")
        f.write(f"End CPU load:   {end_load}%\n")
        f.write(f"Start CPU temp: {start_temp}\n")
        f.write(f"End CPU temp:   {end_temp}\n")


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------

def main():
    if not os.path.exists(ABLATE_BINARY):
        print(f"ERROR: {ABLATE_BINARY} not found. Build it first, e.g.:\n"
              f"  clang++ -O3 -fopenmp -std=c++17 ablate.cpp -o ablate -framework Accelerate")
        sys.exit(1)
    if not os.path.exists(ABLATE_BINARY_SCALAR_SAFE):
        print(f"ERROR: {ABLATE_BINARY_SCALAR_SAFE} not found. Build it first, e.g.:\n"
              f"  clang++ -O3 -fno-tree-vectorize -fno-slp-vectorize -fopenmp -std=c++17 "
              f"ablate.cpp -o ablate_scalar_safe -framework Accelerate")
        sys.exit(1)

    ablation_configs = build_ablation_configs()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_folder = f"run_ablation_{timestamp}"
    os.makedirs(run_folder, exist_ok=True)
    log_dir = os.path.join(run_folder, "logs")
    os.makedirs(log_dir, exist_ok=True)
    results_json = os.path.join(run_folder, "results_summary.json")

    n_calls = len(WIDTHS_TO_RUN) * len(ablation_configs)
    est_minutes = (n_calls * COOLDOWN_SECONDS) / 60.0
    print(f"All ablation results for this session will be saved in: {run_folder}")
    print(f"{len(WIDTHS_TO_RUN)} model/width combos x {len(ablation_configs)} ablation configs "
          f"= {n_calls} total runs.")
    print(f"Estimated minimum wall time (cooldown only, excludes actual run time): ~{est_minutes:.0f} min")

    widths = list(WIDTHS_TO_RUN)
    random.shuffle(widths)
    print(f"\nModel order for this session: {widths}")

    all_run_stats = []

    for pde, width in widths:
        weights_dir = f"results/cpikan_{pde}_weights/width{width}"

        configs_this_model = list(ablation_configs)
        random.shuffle(configs_this_model)
        print(f"\n>>> Testing {pde} width {width} "
              f"(Config order: {[c['label'] for c in configs_this_model]})")

        for cfg in configs_this_model:
            label = cfg["label"]
            output_bin = os.path.join(run_folder, f"pred_{pde}_{width}_{label}.bin")
            log_path = os.path.join(log_dir, f"ablation_{pde}_{width}_{label}.txt")

            # Only the scalar-recurrence ablation needs the vectorizer-disabled
            # build; every other config (including baseline) uses the normally
            # optimized binary so unrelated loops (e.g. simd_tanh) aren't
            # unfairly de-optimized and stay comparable to the main results.
            binary = ABLATE_BINARY_SCALAR_SAFE if cfg["recurrence"] == "scalar" else ABLATE_BINARY

            cmd = [
                binary,
                "--weights", weights_dir,
                "--M", str(M),
                "--warmup", str(WARMUP),
                "--reps", str(REPS),
                "--recurrence", cfg["recurrence"],
                "--matmul", cfg["matmul"],
                "--block-size", str(cfg["block_size"]),
                "--output", output_bin,
            ]

            # Thread count is controlled via env vars (OpenMP + Accelerate both
            # read these), not a CLI flag -- matches how the main script and
            # the original engine control threading.
            run_env = os.environ.copy()
            run_env["OMP_NUM_THREADS"] = str(cfg["threads"])
            run_env["VECLIB_MAXIMUM_THREADS"] = str(cfg["threads"])
            run_env["MKL_NUM_THREADS"] = str(cfg["threads"])

            start_load = get_system_load()
            start_temp = get_cpu_temp()

            out = run_benchmark(cmd, run_env, log_path)

            end_load = get_system_load()
            end_temp = get_cpu_temp()
            append_cpu_temp_log(log_path, start_load, end_load, start_temp, end_temp)

            if out is None:
                print(f"Skipping results for {label} due to crash. Check {log_path}")
                print(f"Cooling down for {COOLDOWN_SECONDS}s...")
                time.sleep(COOLDOWN_SECONDS)
                continue

            mean, tool_std, times = parse_output(out)
            if mean is None:
                print(f"Error: Could not parse output for {label}. Check {log_path}")
                print(f"Cooling down for {COOLDOWN_SECONDS}s...")
                time.sleep(COOLDOWN_SECONDS)
                continue

            std = np.std(times) if times else tool_std

            l2, npy_path = postprocess_prediction(output_bin, pde, M, label)

            res_entry = {
                "model": pde, "width": width,
                "ablation_label": label,
                "recurrence": cfg["recurrence"], "matmul": cfg["matmul"],
                "block_size": cfg["block_size"], "threads": cfg["threads"],
                "mean": float(mean), "std": float(std),
                "l2": float(l2) if l2 is not None and not np.isnan(l2) else None,
                "npy_path": npy_path,
                "start_load": float(start_load), "end_load": float(end_load),
                "start_temp": start_temp, "end_temp": end_temp
            }
            all_run_stats.append(res_entry)

            print(f"[{label}] Mean: {mean:.4f}s +/- {std:.4f}s | L2: {l2:.2e} | "
                  f"Load: {start_load}% -> {end_load}% | Temp: {start_temp} -> {end_temp}")

            try:
                with open(results_json, "w") as f:
                    json.dump(all_run_stats, f, indent=4)
            except TypeError as e:
                print(f"WARNING: could not write results_summary.json after [{label}]: {e}")

            print(f"Cooling down for {COOLDOWN_SECONDS}s...")
            time.sleep(COOLDOWN_SECONDS)

        print(f"Completed all ablation configs for {pde} w{width}. Model Transition: Cooling down for {COOLDOWN_SECONDS}s...")
        time.sleep(COOLDOWN_SECONDS)

    with open(results_json, "w") as f:
        json.dump(all_run_stats, f, indent=4)

    print(f"\nAll ablation runs complete. Data saved to {run_folder}")


if __name__ == "__main__":
    main()
