#include "runner_rec.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <fstream>
#include <string>

double calculate_std(const std::vector<double>& data, double mean) {
    double sum_sq_diff = 0;
    for (double val : data) {
        sum_sq_diff += (val - mean) * (val - mean);
    }
    return std::sqrt(sum_sq_diff / data.size());
}

void save_tensor(const std::string& path, const std::vector<float>& data) {
    std::ofstream os(path, std::ios::binary);
    if (!os) {
        std::cerr << "Error: Could not open file for writing: " << path << std::endl;
        return;
    }
    os.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
}

int main(int argc, char** argv) {
    std::string weights_path;
    int M = 1001 * 1001;
    int warmup = 5;
    int reps = 20;
    std::string output_path;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--weights" && i + 1 < argc) weights_path = argv[++i];
        else if (arg == "--M" && i + 1 < argc) M = std::stoi(argv[++i]);
        else if (arg == "--warmup" && i + 1 < argc) warmup = std::stoi(argv[++i]);
        else if (arg == "--reps" && i + 1 < argc) reps = std::stoi(argv[++i]);
        else if (arg == "--output" && i + 1 < argc) output_path = argv[++i];
    }

    if (weights_path.empty()) {
        std::cerr << "Error: --weights <path> is required." << std::endl;
        return 1;
    }

    Model model;
    try {
        model.load_weights(weights_path);
    } catch (const std::exception& e) {
        std::cerr << "Error loading weights: " << e.what() << std::endl;
        return 1;
    }

    std::vector<float> input_data(M * 2);
    int n = std::sqrt(M);
    if (n * n == M) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                float x = -1.0f + 2.0f * j / (n - 1);
                float y = -1.0f + 2.0f * i / (n - 1);
                input_data[(i * n + j) * 2 + 0] = x;
                input_data[(i * n + j) * 2 + 1] = y;
            }
        }
    } else {
        for (int i = 0; i < M * 2; ++i) {
            input_data[i] = (float)i / (M * 2);
        }
    }

    model.allocate_scratch(M);

    std::vector<float> dummy_out;
    for (int i = 0; i < warmup; ++i) {
        model.forward(input_data.data(), M, dummy_out, false);
    }

    std::vector<double> total_times;
    std::vector<double> avg_tanh(5, 0.0), avg_matmul(5, 0.0), avg_rec(5, 0.0);

    for (int i = 0; i < reps; ++i) {
        ProfilingResults res;
        model.forward(input_data.data(), M, dummy_out, true, &res);
        total_times.push_back(res.total_time);
        for (size_t l = 0; l < res.op_times.size(); ++l) {
            avg_tanh[l] += res.op_times[l]["tanh"];
            avg_matmul[l] += res.op_times[l]["matmul"];
            avg_rec[l] += res.op_times[l]["recurrence"];
        }
    }

    double mean_total = std::accumulate(total_times.begin(), total_times.end(), 0.0) / reps;
    double std_total = calculate_std(total_times, mean_total);

    std::cout << "\n--- Results (Fused Recurrence) ---" << std::endl;
    std::cout << "Overall Pipeline Time: " << mean_total << "s +/- " << std_total << "s" << std::endl;
    std::cout << "Throughput: " << (double)M / mean_total << " points/sec" << std::endl;
    std::cout << "Latency per point: " << (mean_total / M) * 1e6 << " us" << std::endl;

    std::cout << "\n--- Per-Op Breakdown (Averaged over " << reps << " reps) ---" << std::endl;
    for (size_t l = 0; l < avg_tanh.size(); ++l) {
        std::cout << "Layer " << l << ": " 
                  << "tanh=" << avg_tanh[l]/reps << "s "
                  << "matmul=" << avg_matmul[l]/reps << "s "
                  << "recurrence=" << avg_rec[l]/reps << "s" << std::endl;
    }
    
    std::cout << "Individual Times: ";
    for (size_t i = 0; i < total_times.size(); ++i) {
        std::cout << total_times[i] << (i == total_times.size() - 1 ? "" : ",");
    }
    std::cout << std::endl;

    if (!output_path.empty()) {
        save_tensor(output_path, dummy_out);
    }

    return 0;
}
