// ablate.cpp
//
// Single-file ablation harness for the cPIKAN C++ inference engine.
// Mirrors the structure of runner_rec.hpp / run_latency, but every
// "optimization" is a runtime-selectable switch instead of being baked in,
// so all three ablations can be run from one binary without recompiling:
//
//   1. Recurrence kernel:  --recurrence simd   (NEON, default, matches original)
//                          --recurrence scalar (plain loop, no vectorization)
//
//   2. Matmul backend:     --matmul blas       (Accelerate cblas_sgemm, default)
//                          --matmul naive      (hand-written triple loop)
//
//   3. Block size:         --block-size N      (default 512, matches original
//                                                BLOCK_SIZE constant)
//
// Change ONE flag at a time from the defaults to isolate that optimization's
// individual contribution -- that's the point of an ablation. E.g.:
//   ./ablate --weights <dir> --M 1002001 --recurrence scalar   (isolates SIMD/FMA)
//   ./ablate --weights <dir> --M 1002001 --matmul naive        (isolates BLAS)
//   ./ablate --weights <dir> --M 1002001 --block-size 64       (isolates blocking)
//   ./ablate --weights <dir> --M 1002001                       (full baseline,
//                                                                 all optimizations on)
//
// Output format matches the existing run_latency binary exactly (same
// "Overall Pipeline Time: Xs +/- Ys" and "Individual Times: ..." lines) so
// it's parseable by the same parse_output() regex in the orchestrator without
// any changes there.

#include "npy.hpp"
#include <arm_neon.h>
#include <Accelerate/Accelerate.h>
#include <vector>
#include <string>
#include <iostream>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <omp.h>
#include <sys/stat.h>

// ----------------------------------------------------------------------------
// KERNELS -- both variants of each ablated operation live side by side
// ----------------------------------------------------------------------------

static void simd_tanh(const float* in, float* out, size_t n) {
    // Not ablated (tanh isn't one of the three requested axes); kept
    // identical across all configurations so it isn't a confound.
    for (size_t i = 0; i < n; ++i) {
        out[i] = std::tanh(in[i]);
    }
}

// --- Ablation 1: recurrence kernel ---

static void simd_recurrence_step(const float* act, const float* T_prev,
                                  const float* T_prev2, float* T_cur, size_t n) {
    const float32x4_t two = vdupq_n_f32(2.0f);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t a   = vld1q_f32(&act[i]);
        float32x4_t tp  = vld1q_f32(&T_prev[i]);
        float32x4_t tp2 = vld1q_f32(&T_prev2[i]);
        float32x4_t two_a = vmulq_f32(two, a);
        float32x4_t r = vfmaq_f32(vnegq_f32(tp2), two_a, tp);
        vst1q_f32(&T_cur[i], r);
    }
    for (; i < n; ++i)
        T_cur[i] = 2.0f * act[i] * T_prev[i] - T_prev2[i];
}

static void scalar_recurrence_step(const float* act, const float* T_prev,
                                    const float* T_prev2, float* T_cur, size_t n) {
    // Plain scalar loop -- no NEON, no fused multiply-add. This is the
    // "remove SIMD+FMA" ablation. Compiled with -O1/-fno-tree-vectorize
    // (see build notes at bottom of this file) to prevent the compiler
    // from silently auto-vectorizing this back into something equivalent
    // to the SIMD path above.
    for (size_t i = 0; i < n; ++i) {
        T_cur[i] = 2.0f * act[i] * T_prev[i] - T_prev2[i];
    }
}

// --- Ablation 2: matmul backend ---

static void blas_matmul_accumulate(const float* A, size_t M, size_t K,
                                    const float* B, size_t N, float* C) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                (int)M, (int)N, (int)K, 1.0f, A, (int)K, B, (int)N, 1.0f, C, (int)N);
}

static void naive_matmul_accumulate(const float* A, size_t M, size_t K,
                                     const float* B, size_t N, float* C) {
    // Hand-written triple loop, C += A @ B. No tiling, no vendor BLAS.
    // This is the "remove Accelerate/BLAS" ablation -- isolates how much
    // of the speedup comes from Apple's hand-tuned BLAS vs. our own code.
    for (size_t i = 0; i < M; ++i) {
        const float* a_row = A + i * K;
        float* c_row = C + i * N;
        for (size_t k = 0; k < K; ++k) {
            float a_val = a_row[k];
            const float* b_row = B + k * N;
            for (size_t j = 0; j < N; ++j) {
                c_row[j] += a_val * b_row[j];
            }
        }
    }
}

// ----------------------------------------------------------------------------
// MODEL
// ----------------------------------------------------------------------------

enum class RecurrenceMode { SIMD, SCALAR };
enum class MatmulMode { BLAS, NAIVE };

struct AblationConfig {
    RecurrenceMode recurrence = RecurrenceMode::SIMD;
    MatmulMode matmul = MatmulMode::BLAS;
    size_t block_size = 512;  // runtime-configurable; was a compile-time
                               // constexpr (512) in the original engine
};

struct WeightSlice {
    std::vector<float> data;
    size_t in_dim, out_dim;
    const float* raw() const { return data.data(); }
};

static bool file_exists(const std::string& path) {
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0);
}

class Model {
    std::vector<std::vector<WeightSlice>> weights;
    size_t max_dim = 0;
    size_t M_alloc = 0;
    float* h_ptr[2] = {nullptr, nullptr};
    AblationConfig cfg;

    static float* alloc_aligned(size_t bytes) {
        void* ptr = nullptr;
        if (posix_memalign(&ptr, 64, bytes) != 0) throw std::bad_alloc();
        return static_cast<float*>(ptr);
    }

    void matmul_accumulate(const float* A, size_t M, size_t K,
                            const float* B, size_t N, float* C) const {
        if (cfg.matmul == MatmulMode::BLAS) {
            blas_matmul_accumulate(A, M, K, B, N, C);
        } else {
            naive_matmul_accumulate(A, M, K, B, N, C);
        }
    }

    void recurrence_step(const float* act, const float* T_prev,
                          const float* T_prev2, float* T_cur, size_t n) const {
        if (cfg.recurrence == RecurrenceMode::SIMD) {
            simd_recurrence_step(act, T_prev, T_prev2, T_cur, n);
        } else {
            scalar_recurrence_step(act, T_prev, T_prev2, T_cur, n);
        }
    }

public:
    explicit Model(AblationConfig config) : cfg(config) {}

    ~Model() {
        free(h_ptr[0]); free(h_ptr[1]);
    }

    void load_weights(const std::string& weights_dir) {
        weights.clear();
        int l = 0;
        while (true) {
            std::string path = weights_dir + "/layer" + std::to_string(l) + "_W.npy";
            if (!file_exists(path)) break;

            npy::npy_data d = npy::read_npy<float>(path);
            const std::vector<float>& raw_data = d.data;
            const std::vector<unsigned long>& shape = d.shape;
            size_t in_dim = shape[0], out_dim = shape[1], deg_max = shape[2];
            max_dim = std::max({max_dim, in_dim, out_dim});

            std::vector<WeightSlice> layer_by_degree(deg_max);
            for (size_t deg = 0; deg < deg_max; ++deg) {
                WeightSlice ws;
                ws.in_dim = in_dim; ws.out_dim = out_dim;
                ws.data.resize(in_dim * out_dim);
                for (size_t i = 0; i < in_dim; ++i)
                    for (size_t j = 0; j < out_dim; ++j)
                        ws.data[i * out_dim + j] = raw_data[i * (out_dim * deg_max) + j * deg_max + deg];
                layer_by_degree[deg] = std::move(ws);
            }
            weights.push_back(std::move(layer_by_degree));
            ++l;
        }
        if (weights.empty()) {
            throw std::runtime_error("No layer*_W.npy files found in " + weights_dir);
        }
    }

    void allocate_scratch(size_t M) {
        if (M_alloc == M) return;
        free(h_ptr[0]); free(h_ptr[1]);
        M_alloc = M;
        size_t full_bytes = M * max_dim * sizeof(float);
        h_ptr[0] = alloc_aligned(full_bytes);
        h_ptr[1] = alloc_aligned(full_bytes);
    }

    void forward(const float* x_in, size_t M, std::vector<float>& out_final) {
        if (M != M_alloc) allocate_scratch(M);

        const float* shared_cur_h = x_in;
        size_t shared_cur_in_dim = 2;
        int shared_out_idx = 0;
        const size_t BLOCK_SIZE = cfg.block_size;

        #pragma omp parallel
        {
            size_t block_bytes = BLOCK_SIZE * max_dim * sizeof(float);
            float* t_act  = (float*)aligned_alloc(64, block_bytes);
            float* t_T0   = (float*)aligned_alloc(64, block_bytes);
            float* t_T1   = (float*)aligned_alloc(64, block_bytes);
            float* t_Tcur = (float*)aligned_alloc(64, block_bytes);

            for (size_t l = 0; l < weights.size(); ++l) {
                size_t out_dim = weights[l][0].out_dim;
                size_t deg_max = weights[l].size();
                float* out_ptr = h_ptr[shared_out_idx];

                #pragma omp for schedule(static)
                for (size_t b_start = 0; b_start < M; b_start += BLOCK_SIZE) {
                    size_t b_end = std::min(b_start + BLOCK_SIZE, M);
                    size_t b_size = b_end - b_start;
                    size_t b_n = b_size * shared_cur_in_dim;
                    float* out_block = out_ptr + b_start * out_dim;
                    std::memset(out_block, 0, b_size * out_dim * sizeof(float));

                    simd_tanh(shared_cur_h + b_start * shared_cur_in_dim, t_act, b_n);

                    std::fill(t_T0, t_T0 + b_n, 1.0f);
                    std::memcpy(t_T1, t_act, b_n * sizeof(float));

                    matmul_accumulate(t_T0, b_size, shared_cur_in_dim, weights[l][0].raw(), out_dim, out_block);
                    matmul_accumulate(t_T1, b_size, shared_cur_in_dim, weights[l][1].raw(), out_dim, out_block);

                    float* p_prev2 = t_T0;
                    float* p_prev1 = t_T1;
                    float* p_cur   = t_Tcur;

                    for (size_t d = 2; d < deg_max; ++d) {
                        recurrence_step(t_act, p_prev1, p_prev2, p_cur, b_n);
                        matmul_accumulate(p_cur, b_size, shared_cur_in_dim, weights[l][d].raw(), out_dim, out_block);

                        float* tmp = p_prev2;
                        p_prev2 = p_prev1;
                        p_prev1 = p_cur;
                        p_cur = tmp;
                    }
                }

                #pragma omp single
                {
                    shared_cur_h = out_ptr;
                    shared_cur_in_dim = out_dim;
                    shared_out_idx = 1 - shared_out_idx;
                }
                #pragma omp barrier
            }
            free(t_act); free(t_T0); free(t_T1); free(t_Tcur);
        }

        out_final.assign(shared_cur_h, shared_cur_h + M * shared_cur_in_dim);
    }
};

// ----------------------------------------------------------------------------
// INPUT GENERATION -- mirrors run.py / run_pytorch.py exactly so all three
// harnesses (np, torch, cpp/ablate) evaluate on identical points.
// ----------------------------------------------------------------------------

static std::vector<float> generate_input(size_t M) {
    std::vector<float> xy(M * 2);
    size_t n = (size_t)std::sqrt((double)M);
    if (n * n == M) {
        // meshgrid of linspace(-1, 1, n) x linspace(-1, 1, n), row-major,
        // matching np.meshgrid(x, t) then ravel() in the Python harnesses.
        std::vector<float> lin(n);
        for (size_t i = 0; i < n; ++i) {
            lin[i] = -1.0f + 2.0f * (float)i / (float)(n - 1);
        }
        size_t idx = 0;
        for (size_t row = 0; row < n; ++row) {      // t index (outer, matches meshgrid 'xy' indexing)
            for (size_t col = 0; col < n; ++col) {   // x index (inner)
                xy[idx * 2 + 0] = lin[col];
                xy[idx * 2 + 1] = lin[row];
                ++idx;
            }
        }
    } else {
        // linspace(0, 1, M*2).reshape(M, 2)
        size_t total = M * 2;
        for (size_t i = 0; i < total; ++i) {
            xy[i] = (float)i / (float)(total - 1);
        }
    }
    return xy;
}

// ----------------------------------------------------------------------------
// CLI + MAIN
// ----------------------------------------------------------------------------

struct Args {
    std::string weights;
    size_t M = 1001 * 1001;
    int warmup = 5;
    int reps = 15;
    std::string output;
    RecurrenceMode recurrence = RecurrenceMode::SIMD;
    MatmulMode matmul = MatmulMode::BLAS;
    size_t block_size = 512;
};

static std::string get_arg(int argc, char** argv, const std::string& flag, const std::string& def = "") {
    for (int i = 1; i < argc - 1; ++i) {
        if (flag == argv[i]) return std::string(argv[i + 1]);
    }
    return def;
}

static bool has_flag(int argc, char** argv, const std::string& flag) {
    for (int i = 1; i < argc; ++i) {
        if (flag == argv[i]) return true;
    }
    return false;
}

static Args parse_args(int argc, char** argv) {
    Args args;
    if (has_flag(argc, argv, "--help") || has_flag(argc, argv, "-h")) {
        std::cout <<
            "Usage: ./ablate --weights <dir> [options]\n"
            "  --weights <dir>          Required. Directory with layerN_W.npy files.\n"
            "  --M <int>                Number of inference points (default 1002001).\n"
            "  --warmup <int>           Warmup iterations (default 5).\n"
            "  --reps <int>             Timed repetitions (default 15).\n"
            "  --output <path>          Save predictions to this path (optional).\n"
            "  --recurrence simd|scalar Recurrence kernel ablation (default simd).\n"
            "  --matmul blas|naive      Matmul backend ablation (default blas).\n"
            "  --block-size <int>       Block size for tiling ablation (default 512).\n";
        std::exit(0);
    }

    args.weights = get_arg(argc, argv, "--weights");
    if (args.weights.empty()) {
        std::cerr << "ERROR: --weights <dir> is required. Use --help for usage.\n";
        std::exit(1);
    }

    std::string m_str = get_arg(argc, argv, "--M");
    if (!m_str.empty()) args.M = (size_t)std::stoul(m_str);

    std::string warmup_str = get_arg(argc, argv, "--warmup");
    if (!warmup_str.empty()) args.warmup = std::stoi(warmup_str);

    std::string reps_str = get_arg(argc, argv, "--reps");
    if (!reps_str.empty()) args.reps = std::stoi(reps_str);

    args.output = get_arg(argc, argv, "--output");

    std::string rec_str = get_arg(argc, argv, "--recurrence", "simd");
    if (rec_str == "scalar") args.recurrence = RecurrenceMode::SCALAR;
    else if (rec_str == "simd") args.recurrence = RecurrenceMode::SIMD;
    else { std::cerr << "ERROR: --recurrence must be 'simd' or 'scalar'\n"; std::exit(1); }

    std::string matmul_str = get_arg(argc, argv, "--matmul", "blas");
    if (matmul_str == "naive") args.matmul = MatmulMode::NAIVE;
    else if (matmul_str == "blas") args.matmul = MatmulMode::BLAS;
    else { std::cerr << "ERROR: --matmul must be 'blas' or 'naive'\n"; std::exit(1); }

    std::string block_str = get_arg(argc, argv, "--block-size");
    if (!block_str.empty()) args.block_size = (size_t)std::stoul(block_str);

    return args;
}

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);

    AblationConfig cfg;
    cfg.recurrence = args.recurrence;
    cfg.matmul = args.matmul;
    cfg.block_size = args.block_size;

    Model model(cfg);
    model.load_weights(args.weights);

    std::vector<float> xy = generate_input(args.M);

    std::vector<float> out;

    // Warmup (untimed)
    for (int i = 0; i < args.warmup; ++i) {
        model.forward(xy.data(), args.M, out);
    }

    // Timed repetitions
    std::vector<double> times;
    times.reserve(args.reps);
    for (int i = 0; i < args.reps; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        model.forward(xy.data(), args.M, out);
        auto t1 = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration<double>(t1 - t0).count());
    }

    double mean = 0.0;
    for (double t : times) mean += t;
    mean /= times.size();

    double var = 0.0;
    for (double t : times) var += (t - mean) * (t - mean);
    var /= times.size();
    double stddev = std::sqrt(var);

    std::string rec_label = (args.recurrence == RecurrenceMode::SIMD) ? "simd" : "scalar";
    std::string matmul_label = (args.matmul == MatmulMode::BLAS) ? "blas" : "naive";

    // Output format matches the existing run_latency binary so the same
    // parse_output() regex in the Python orchestrator works unmodified.
    std::cout << "\n--- Results (ABLATION recurrence=" << rec_label
               << " matmul=" << matmul_label
               << " block_size=" << args.block_size << ") ---\n";
    std::cout << "Overall Pipeline Time: " << mean << "s +/- " << stddev << "s\n";
    std::cout << "Throughput: " << (double)args.M / mean << " points/sec\n";
    std::cout << "Latency per point: " << (mean / (double)args.M) * 1e6 << " us\n";
    std::cout << "Individual Times: ";
    for (size_t i = 0; i < times.size(); ++i) {
        std::cout << times[i];
        if (i + 1 < times.size()) std::cout << ",";
    }
    std::cout << "\n";

    if (!args.output.empty()) {
        FILE* f = fopen(args.output.c_str(), "wb");
        if (f) {
            fwrite(out.data(), sizeof(float), out.size(), f);
            fclose(f);
        } else {
            std::cerr << "WARNING: could not open output path " << args.output << " for writing.\n";
        }
    }

    return 0;
}

// ----------------------------------------------------------------------------
// BUILD NOTES
// ----------------------------------------------------------------------------
//
// Baseline (all optimizations on, matches original run_latency binary):
//   clang++ -O3 -march=native -fopenmp -std=c++17 ablate.cpp -o ablate \
//       -framework Accelerate
//
// IMPORTANT for the --recurrence scalar ablation specifically: -O3 can
// auto-vectorize (and even auto-FMA) the plain scalar loop, silently
// reintroducing what you're trying to ablate away and making "scalar" not
// actually scalar in the compiled binary. To guarantee a clean scalar
// baseline, either:
//   (a) build a second binary at lower optimization for that comparison:
//         clang++ -O1 -fopenmp -std=c++17 ablate.cpp -o ablate_scalar_safe \
//             -framework Accelerate
//   (b) or keep -O3 but add -fno-tree-vectorize -fno-slp-vectorize:
//         clang++ -O3 -fno-tree-vectorize -fno-slp-vectorize -fopenmp \
//             -std=c++17 ablate.cpp -o ablate -framework Accelerate
// Option (b) is preferable since it keeps everything else (matmul path,
// general codegen quality) consistent with the baseline build -- only the
// vectorizer is disabled, isolating exactly the SIMD/FMA variable you
// intend to ablate. Verify with `otool -tv ablate | grep -i fmla` (should
// find zero matches in the scalar build) before trusting the numbers.
