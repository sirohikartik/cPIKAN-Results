import os
import subprocess
import re
import numpy as np
import json
import sys
import time
import random
import psutil
from typing import List, Tuple

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
        # Linux with psutil but no sensors reporting -> fall through to N/A below
    except AttributeError:
        # psutil.sensors_temperatures() doesn't exist on macOS -> try macmon
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
    # Handle both la-standard and la-naive (Estimated) output formats
    # NOTE: the '+' in '+/-' must be escaped, otherwise it's parsed as a
    # quantifier on the preceding space and never matches the literal '+'.
    mean_match = re.search(r"Overall Pipeline Time(?: \(Estimated for M=\d+\))?: ([\d.]+)s \+/- ([\d.]+)s", output)
    times_match = re.search(r"Individual Times: ([\d.,\s]+)", output)
    if not mean_match: return None, None, None
    mean = float(mean_match.group(1))
    # This is the tool's OWN computed std (e.g. run.py / run_pytorch.py both
    # compute np.std(total_times) internally over their real per-rep timing
    # and print it here). Previously this was discarded entirely -- only
    # tools that also print "Individual Times" (just the cpp binary) ever
    # got a non-zero std recorded, so np/torch methods always showed 0.0
    # regardless of their actual variance.
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

def compute_relative_l2_exact(pred_path, pde, M):
    if not os.path.exists(pred_path): return np.nan
    u_pred = np.fromfile(pred_path, dtype=np.float32).flatten()
    u_exact = get_exact_solution(pde, M).flatten()
    min_len = min(len(u_pred), len(u_exact))
    norm_diff = np.linalg.norm(u_pred[:min_len] - u_exact[:min_len])
    norm_exact = np.linalg.norm(u_exact[:min_len])
    return norm_diff / norm_exact if norm_exact > 1e-12 else np.nan

def postprocess_prediction(pred_path, pde, M, label):
    """
    Post-run step for a single method's output:
      1. Load the raw .bin prediction.
      2. Save it as a proper .npy file (not just raw bin) for easier reuse.
      3. Compute the relative L2 error against the exact solution.
    Returns (l2_error, npy_path). l2_error is np.nan if the bin is missing/empty.
    """
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

def run_benchmark(cmd: List[str], log_path: str = None):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w") as f:
            f.write(f"COMMAND: {' '.join(cmd)}\n")
            f.write("STDOUT:\n" + result.stdout + "\n\nSTDERR:\n" + result.stderr)

    if result.returncode != 0:
        print(f"CRITICAL ERROR: Command failed with return code {result.returncode}")
        print(f"STDERR: {result.stderr}")
        return None

    print(result.stdout) # Print raw output to terminal for visibility
    return result.stdout

def append_cpu_temp_log(log_path, start_load, end_load, start_temp, end_temp):
    """Append a CPU/thermal block to the per-run log file."""
    if not log_path:
        return
    with open(log_path, "a") as f:
        f.write("\n\n--- CPU / THERMAL ---\n")
        f.write(f"Start CPU load: {start_load}%\n")
        f.write(f"End CPU load:   {end_load}%\n")
        f.write(f"Start CPU temp: {start_temp}\n")
        f.write(f"End CPU temp:   {end_temp}\n")

def main():
    M = 1001 * 1001
    warmup, reps = 5, 15
    threads = "8"

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_folder = f"run_{timestamp}"
    os.makedirs(run_folder, exist_ok=True)
    log_dir = os.path.join(run_folder, "logs")
    os.makedirs(log_dir, exist_ok=True)
    results_json = os.path.join(run_folder, "results_summary.json")

    print(f"All results for this session will be saved in: {run_folder}")

    os.environ["OMP_NUM_THREADS"] = threads
    os.environ["VECLIB_MAXIMUM_THREADS"] = threads
    os.environ["MKL_NUM_THREADS"] = threads

    baselines = [("diffusion", "32"), ("diffusion", "64"), ("diffusion", "128"), ("reaction_diffusion", "32")]
    random.shuffle(baselines)
    print("\n" + "="*30 + " STAGE 1: THERMALLY CONTROLLED BASELINES " + "="*30)
    print(f"Model order for this session: {baselines}")

    all_run_stats = []

    for pde, width in baselines:
        weights_dir = f"results/cpikan_{pde}_weights/width{width}"

        # Define methods to be run for this model
        # Dropped np_naive and onnx as requested
        methods = [
            ("np", [sys.executable, "run.py", "--weights", weights_dir, "--M", str(M), "--warmup", str(warmup), "--reps", str(reps), "--output", os.path.join(run_folder, f"pred_{pde}_{width}_np.bin")], os.path.join(log_dir, f"baseline_{pde}_{width}_np.txt")),
            ("torch_eager", [sys.executable, "run_pytorch.py", "--weights", weights_dir, "--M", str(M), "--warmup", str(warmup), "--reps", str(reps), "--threads", threads, "--mode", "eager", "--output", os.path.join(run_folder, f"pred_{pde}_{width}_torch_eager.bin")], os.path.join(log_dir, f"baseline_{pde}_{width}_torch_eager.txt")),
            ("torch_script", [sys.executable, "run_pytorch.py", "--weights", weights_dir, "--M", str(M), "--warmup", str(warmup), "--reps", str(reps), "--threads", threads, "--mode", "script", "--output", os.path.join(run_folder, f"pred_{pde}_{width}_torch_script.bin")], os.path.join(log_dir, f"baseline_{pde}_{width}_torch_script.txt")),
            ("torch_compile", [sys.executable, "run_pytorch.py", "--weights", weights_dir, "--M", str(M), "--warmup", str(warmup), "--reps", str(reps), "--threads", threads, "--mode", "compile", "--output", os.path.join(run_folder, f"pred_{pde}_{width}_torch_compile.bin")], os.path.join(log_dir, f"baseline_{pde}_{width}_torch_compile.txt")),
            ("cpp", ["./build/run_latency", "--weights", weights_dir, "--M", str(M), "--warmup", str(warmup), "--reps", str(reps), "--output", os.path.join(run_folder, f"pred_{pde}_{width}_cpp.bin")], os.path.join(log_dir, f"baseline_{pde}_{width}_cpp.txt")),
        ]

        random.shuffle(methods)
        print(f"\n>>> Testing {pde} width {width} (Random Order: {[m[0] for m in methods]})")

        for label, cmd, log_path in methods:
            start_load = get_system_load()
            start_temp = get_cpu_temp()

            out = run_benchmark(cmd, log_path)

            end_load = get_system_load()
            end_temp = get_cpu_temp()
            # Always record CPU/temp for this attempt, even on crash/parse failure,
            # so the log always reflects what conditions the run happened under.
            append_cpu_temp_log(log_path, start_load, end_load, start_temp, end_temp)

            if out is None:
                print(f"Skipping results for {label} due to crash. Check {log_path}")
                print(f"Cooling down for 180s...")
                time.sleep(180)
                continue

            mean, tool_std, times = parse_output(out)
            if mean is None:
                print(f"Error: Could not parse output for {label}. Check {log_path}")
                print(f"Cooling down for 180s...")
                time.sleep(180)
                continue

            # Prefer std recomputed from per-rep "Individual Times" when available
            # (currently only the cpp harness prints those). Otherwise fall back
            # to the tool's own reported std -- NOT a hardcoded 0.0, which was
            # silently discarding real variance for every non-cpp method.
            std = np.std(times) if times else tool_std

            # This filename must match exactly what each command's --output flag
            # was given above (previously the "np" case had a typo: "npP.bin"
            # instead of "np.bin", which silently broke L2 for that method).
            pred_file = os.path.join(run_folder, f"pred_{pde}_{width}_{label}.bin")

            l2, npy_path = postprocess_prediction(pred_file, pde, M, label)

            # Cast numpy scalar types (float32/float64) to native Python floats.
            # json.dump cannot serialize numpy types, which is what crashed the
            # final write last time -- everything held only in memory was lost.
            res_entry = {
                "model": pde, "width": width, "method": label,
                "mean": float(mean), "std": float(std),
                "l2": float(l2) if l2 is not None and not np.isnan(l2) else None,
                "npy_path": npy_path,
                "start_load": float(start_load), "end_load": float(end_load),
                "start_temp": start_temp, "end_temp": end_temp
            }
            all_run_stats.append(res_entry)

            print(f"[{label}] Mean: {mean:.4f}s +/- {std:.4f}s | L2: {l2:.2e} | Load: {start_load}% -> {end_load}% | Temp: {start_temp} -> {end_temp}")

            # Write results_summary.json after EVERY method, not just at the end.
            # A crash mid-run (e.g. a non-serializable value) now only risks the
            # current entry, not the entire session's accumulated results.
            try:
                with open(results_json, "w") as f:
                    json.dump(all_run_stats, f, indent=4)
            except TypeError as e:
                print(f"WARNING: could not write results_summary.json after [{label}]: {e}")
            print(f"Cooling down for 180s...")
            time.sleep(180)

        print(f"Completed all methods for {pde} w{width}. Model Transition: Cooling down for 180s...")
        time.sleep(180)

    # Stage 2-4 simplified for final run focus on baselines
    with open(results_json, "w") as f:
        json.dump(all_run_stats, f, indent=4)

    print(f"\nAll benchmarks complete. Data saved to {run_folder}")

if __name__ == "__main__":
    main()
