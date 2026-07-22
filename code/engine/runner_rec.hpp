#ifndef RUNNER_REC_HPP
#define RUNNER_REC_HPP

#include "npy.hpp"
#include <arm_neon.h>
#include <Accelerate/Accelerate.h>
#include <vector>
#include <string>
#include <iostream>
#include <chrono>
#include <map>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <stdexcept>
#include <omp.h>

struct ProfilingResults {
    double total_time;
    std::vector<double> layer_times;
    std::vector<std::map<std::string, double>> op_times;
};

// ----------------------------------------------------------------------------
// KERNELS
// ----------------------------------------------------------------------------

static void simd_tanh(const float* in, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        out[i] = std::tanh(in[i]);
    }
}

static void simd_recurrence_step(const float* act, const float* T_prev,
                                  const float* T_prev2, float* T_cur, size_t n) {
    const float32x4_t two = vdupq_n_f32(2.0f);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t a  = vld1q_f32(&act[i]);
        float32x4_t tp = vld1q_f32(&T_prev[i]);
        float32x4_t tp2 = vld1q_f32(&T_prev2[i]);
        float32x4_t two_a = vmulq_f32(two, a);
        float32x4_t r = vfmaq_f32(vnegq_f32(tp2), two_a, tp);
        vst1q_f32(&T_cur[i], r);
    }
    for (; i < n; ++i)
        T_cur[i] = 2.0f * act[i] * T_prev[i] - T_prev2[i];
}

static void blas_matmul_accumulate(const float* A, size_t M, size_t K,
                                    const float* B, size_t N, float* C) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                (int)M, (int)N, (int)K, 1.0f, A, (int)K, B, (int)N, 1.0f, C, (int)N);
}

struct WeightSlice {
    std::vector<float> data;
    size_t in_dim, out_dim;
    const float* raw() const { return data.data(); }
};

class Model {
    std::vector<std::vector<WeightSlice>> weights;
    size_t max_dim = 0;
    size_t M_alloc = 0;
    static constexpr size_t BLOCK_SIZE = 512;
    float* h_ptr[2] = {nullptr, nullptr};
    float* b_act = nullptr;
    float* b_T0 = nullptr;
    float* b_T1 = nullptr;
    float* b_Tcur = nullptr;

    static float* alloc_aligned(size_t bytes) {
        void* ptr = nullptr;
        if (posix_memalign(&ptr, 64, bytes) != 0) throw std::bad_alloc();
        return static_cast<float*>(ptr);
    }

public:
    Model() = default;
    ~Model() {
        free(h_ptr[0]); free(h_ptr[1]);
        free(b_act); free(b_T0); free(b_T1); free(b_Tcur);
    }

    void load_weights(const std::string& weights_dir) {
        weights.clear();
        for (int l = 0; l < 5; ++l) {
            std::string path = weights_dir + "/layer" + std::to_string(l) + "_W.npy";
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
        }
    }

    void allocate_scratch(size_t M) {
        if (M_alloc == M) return;
        free(h_ptr[0]); free(h_ptr[1]);
        free(b_act); free(b_T0); free(b_T1); free(b_Tcur);
        M_alloc = M;
        size_t full_bytes = M * max_dim * sizeof(float);
        size_t block_bytes = BLOCK_SIZE * max_dim * sizeof(float);
        h_ptr[0] = alloc_aligned(full_bytes);
        h_ptr[1] = alloc_aligned(full_bytes);
        b_act  = alloc_aligned(block_bytes);
        b_T0   = alloc_aligned(block_bytes);
        b_T1   = alloc_aligned(block_bytes);
        b_Tcur = alloc_aligned(block_bytes);
    }

    void forward(const float* x_in, size_t M, std::vector<float>& out_final,
                 bool profile = false, ProfilingResults* results = nullptr, bool verbose = false) {
        if (M != M_alloc) allocate_scratch(M);

        auto t_total_start = std::chrono::high_resolution_clock::now();
        
        // Variables that must be shared across threads
        const float* shared_cur_h = x_in;
        size_t shared_cur_in_dim = 2;
        int shared_out_idx = 0;

        std::vector<double> layer_tanh(weights.size(), 0.0);
        std::vector<double> layer_matmul(weights.size(), 0.0);
        std::vector<double> layer_rec(weights.size(), 0.0);

        #pragma omp parallel
        {
            size_t block_bytes = BLOCK_SIZE * max_dim * sizeof(float);
            float* t_act = (float*)aligned_alloc(64, block_bytes);
            float* t_T0 = (float*)aligned_alloc(64, block_bytes);
            float* t_T1 = (float*)aligned_alloc(64, block_bytes);
            float* t_Tcur = (float*)aligned_alloc(64, block_bytes);

            for (size_t l = 0; l < weights.size(); ++l) {
                size_t out_dim = weights[l][0].out_dim;
                size_t deg_max = weights[l].size();
                float* out_ptr = h_ptr[shared_out_idx];

                double local_tanh = 0, local_matmul = 0, local_rec = 0;

                #pragma omp for schedule(static)
                for (size_t b_start = 0; b_start < M; b_start += BLOCK_SIZE) {
                    size_t b_end = std::min(b_start + BLOCK_SIZE, M);
                    size_t b_size = b_end - b_start;
                    size_t b_n = b_size * shared_cur_in_dim;
                    float* out_block = out_ptr + b_start * out_dim;
                    std::memset(out_block, 0, b_size * out_dim * sizeof(float));

                    auto s0 = std::chrono::high_resolution_clock::now();
                    simd_tanh(shared_cur_h + b_start * shared_cur_in_dim, t_act, b_n);
                    auto s1 = std::chrono::high_resolution_clock::now();
                    local_tanh += std::chrono::duration<double>(s1 - s0).count();

                    std::fill(t_T0, t_T0 + b_n, 1.0f);
                    std::memcpy(t_T1, t_act, b_n * sizeof(float));

                    auto s2 = std::chrono::high_resolution_clock::now();
                    blas_matmul_accumulate(t_T0, b_size, shared_cur_in_dim, weights[l][0].raw(), out_dim, out_block);
                    blas_matmul_accumulate(t_T1, b_size, shared_cur_in_dim, weights[l][1].raw(), out_dim, out_block);
                    auto s3 = std::chrono::high_resolution_clock::now();
                    local_matmul += std::chrono::duration<double>(s3 - s2).count();

                    float* p_prev2 = t_T0;
                    float* p_prev1 = t_T1;
                    float* p_cur   = t_Tcur;
                    
                    for (size_t d = 2; d < deg_max; ++d) {
                        auto rs0 = std::chrono::high_resolution_clock::now();
                        simd_recurrence_step(t_act, p_prev1, p_prev2, p_cur, b_n);
                        auto rs1 = std::chrono::high_resolution_clock::now();
                        local_rec += std::chrono::duration<double>(rs1 - rs0).count();

                        auto ms0 = std::chrono::high_resolution_clock::now();
                        blas_matmul_accumulate(p_cur, b_size, shared_cur_in_dim, weights[l][d].raw(), out_dim, out_block);
                        auto ms1 = std::chrono::high_resolution_clock::now();
                        local_matmul += std::chrono::duration<double>(ms1 - ms0).count();

                        float* tmp = p_prev2;
                        p_prev2 = p_prev1;
                        p_prev1 = p_cur;
                        p_cur = tmp;
                    }
                }

                #pragma omp critical
                {
                    layer_tanh[l] += local_tanh;
                    layer_matmul[l] += local_matmul;
                    layer_rec[l] += local_rec;
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

        auto t_total_end = std::chrono::high_resolution_clock::now();
        if (profile && results) {
            results->total_time = std::chrono::duration<double>(t_total_end - t_total_start).count();
            for (size_t l = 0; l < weights.size(); ++l) {
                std::map<std::string, double> ops;
                ops["tanh"] = layer_tanh[l];
                ops["matmul"] = layer_matmul[l];
                ops["recurrence"] = layer_rec[l];
                results->op_times.push_back(ops);
            }
        }

        // Final pointer is the one shared_cur_h was last set to
        out_final.assign(shared_cur_h, shared_cur_h + M * shared_cur_in_dim);
    }
};

#endif
