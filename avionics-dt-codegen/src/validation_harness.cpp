/**
 * @file validation_harness.cpp
 * @brief Host-side validation harness for the generated classifier.
 *
 * Loads a CSV test dataset, runs the generated classify() function on
 * each sample, and reports accuracy, confusion matrix, per-class metrics,
 * and latency statistics.
 *
 * @note HOST-SIDE ONLY.
 * @note Usage: validation_harness <test_data.csv> [--threshold 0.95]
 *
 * CSV format: one header row, then N rows of (feature_0, ..., feature_N-1, label_string).
 *
 * @note DO-178C: This harness is an objective verification artifact — it
 *       validates the generated code against ground truth test vectors.
 */

#include "generated_classifier.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** @brief Default accuracy threshold (fraction, 0.0–1.0). */
constexpr double kDefaultAccuracyThreshold = 0.95;

/** @brief Number of repeated inference calls for latency measurement. */
constexpr int kLatencyRepetitions = 10000;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace {

/**
 * @brief Resolve a class label string to avionics_ml::Label.
 *
 * @param name  Class label string from CSV.
 * @return int  Index (0-based), or -1 if not found.
 */
int resolve_label(const std::string& name) {
    for (uint8_t i = 0; i < avionics_ml::kNumClasses; ++i) {
        if (name == avionics_ml::kClassNames[i]) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

/**
 * @brief Trim leading/trailing whitespace from a string.
 *
 * @param s  Input string.
 * @return std::string  Trimmed string.
 */
std::string trim(const std::string& s) {
    const auto start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    const auto end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

/**
 * @brief Split a CSV line into tokens.
 *
 * @param line  Input line.
 * @return std::vector<std::string>  Tokens.
 */
std::vector<std::string> split_csv(const std::string& line) {
    std::vector<std::string> tokens;
    std::istringstream iss(line);
    std::string token;
    while (std::getline(iss, token, ',')) {
        tokens.push_back(trim(token));
    }
    return tokens;
}

/**
 * @brief A single test sample.
 */
struct TestSample {
    float features[256];  // over-provisioned; only kNumFeatures used.
    int ground_truth;     // class index.
};

}  // anonymous namespace

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    // --- Parse arguments ----------------------------------------------------
    if (argc < 2) {
        std::cerr << "Usage: validation_harness <test_data.csv> "
                     "[--threshold 0.95]\n";
        return 1;
    }

    const std::string csv_path = argv[1];
    double accuracy_threshold = kDefaultAccuracyThreshold;

    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--threshold") == 0 && i + 1 < argc) {
            accuracy_threshold = std::atof(argv[++i]);
        }
    }

    // --- Load CSV -----------------------------------------------------------
    std::ifstream ifs(csv_path);
    if (!ifs.is_open()) {
        std::cerr << "ERROR: cannot open " << csv_path << "\n";
        return 1;
    }

    std::vector<TestSample> samples;
    std::string line;

    // Skip header.
    if (!std::getline(ifs, line)) {
        std::cerr << "ERROR: empty CSV file\n";
        return 1;
    }

    int line_no = 1;
    while (std::getline(ifs, line)) {
        ++line_no;
        if (line.empty()) continue;

        auto tokens = split_csv(line);
        if (static_cast<int>(tokens.size()) != kNumFeatures + 1) {
            std::cerr << "WARNING: skipping line " << line_no
                      << " (expected " << (kNumFeatures + 1)
                      << " columns, got " << tokens.size() << ")\n";
            continue;
        }

        TestSample sample{};
        bool parse_ok = true;
        for (int f = 0; f < kNumFeatures; ++f) {
            try {
                sample.features[f] = std::stof(tokens[static_cast<std::size_t>(f)]);
            } catch (...) {
                std::cerr << "WARNING: bad float on line " << line_no
                          << " col " << f << "\n";
                parse_ok = false;
                break;
            }
        }
        if (!parse_ok) continue;

        sample.ground_truth =
            resolve_label(tokens[static_cast<std::size_t>(kNumFeatures)]);
        if (sample.ground_truth < 0) {
            std::cerr << "WARNING: unknown label '"
                      << tokens[static_cast<std::size_t>(kNumFeatures)]
                      << "' on line " << line_no << "\n";
            continue;
        }

        samples.push_back(sample);
    }

    if (samples.empty()) {
        std::cerr << "ERROR: no valid samples loaded\n";
        return 1;
    }

    std::cout << "Loaded " << samples.size() << " test samples.\n\n";

    // --- Inference + accuracy -----------------------------------------------
    const int n_classes = static_cast<int>(avionics_ml::kNumClasses);
    std::vector<std::vector<int>> confusion(
        static_cast<std::size_t>(n_classes),
        std::vector<int>(static_cast<std::size_t>(n_classes), 0));

    int correct = 0;
    for (const auto& s : samples) {
        const auto pred = avionics_ml::classify(s.features);
        const int pred_idx = static_cast<int>(pred);
        confusion[static_cast<std::size_t>(s.ground_truth)]
                 [static_cast<std::size_t>(pred_idx)]++;
        if (pred_idx == s.ground_truth) {
            ++correct;
        }
    }

    const double accuracy =
        static_cast<double>(correct) / static_cast<double>(samples.size());

    // --- Print results ------------------------------------------------------
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Accuracy: " << accuracy
              << " (" << correct << "/" << samples.size() << ")\n\n";

    // Confusion matrix.
    std::cout << "Confusion Matrix (rows = truth, cols = predicted):\n";
    std::cout << std::setw(20) << " ";
    for (int c = 0; c < n_classes; ++c) {
        std::cout << std::setw(14) << avionics_ml::kClassNames[c];
    }
    std::cout << "\n";
    for (int r = 0; r < n_classes; ++r) {
        std::cout << std::setw(20) << avionics_ml::kClassNames[r];
        for (int c = 0; c < n_classes; ++c) {
            std::cout << std::setw(14)
                      << confusion[static_cast<std::size_t>(r)]
                                  [static_cast<std::size_t>(c)];
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    // Per-class precision, recall, F1.
    std::cout << std::setw(20) << "Class"
              << std::setw(12) << "Precision"
              << std::setw(12) << "Recall"
              << std::setw(12) << "F1"
              << "\n";
    for (int c = 0; c < n_classes; ++c) {
        int tp = confusion[static_cast<std::size_t>(c)]
                           [static_cast<std::size_t>(c)];
        int col_sum = 0;
        int row_sum = 0;
        for (int k = 0; k < n_classes; ++k) {
            col_sum += confusion[static_cast<std::size_t>(k)]
                                [static_cast<std::size_t>(c)];
            row_sum += confusion[static_cast<std::size_t>(c)]
                                [static_cast<std::size_t>(k)];
        }
        const double precision =
            col_sum > 0 ? static_cast<double>(tp) / static_cast<double>(col_sum) : 0.0;
        const double recall =
            row_sum > 0 ? static_cast<double>(tp) / static_cast<double>(row_sum) : 0.0;
        const double f1 =
            (precision + recall > 0.0)
                ? 2.0 * precision * recall / (precision + recall)
                : 0.0;

        std::cout << std::setw(20) << avionics_ml::kClassNames[c]
                  << std::setw(12) << precision
                  << std::setw(12) << recall
                  << std::setw(12) << f1
                  << "\n";
    }
    std::cout << "\n";

    // --- Latency measurement ------------------------------------------------
    // Use the first sample for repeated timing.
    const auto& bench_sample = samples[0];
    volatile avionics_ml::Label sink;  // prevent optimization.

    std::vector<double> latencies_ns;
    latencies_ns.reserve(kLatencyRepetitions);

    for (int rep = 0; rep < kLatencyRepetitions; ++rep) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        sink = avionics_ml::classify(bench_sample.features);
        const auto t1 = std::chrono::high_resolution_clock::now();
        const double ns = static_cast<double>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0)
                .count());
        latencies_ns.push_back(ns);
    }
    (void)sink;

    std::sort(latencies_ns.begin(), latencies_ns.end());
    double sum = 0.0;
    for (double v : latencies_ns) sum += v;

    std::cout << "Latency over " << kLatencyRepetitions << " calls:\n"
              << "  Min  : " << latencies_ns.front() << " ns\n"
              << "  Mean : " << (sum / static_cast<double>(kLatencyRepetitions))
              << " ns\n"
              << "  Max  : " << latencies_ns.back() << " ns\n\n";

    // --- Threshold check ----------------------------------------------------
    if (accuracy < accuracy_threshold) {
        std::cerr << "FAIL: accuracy " << accuracy
                  << " below threshold " << accuracy_threshold << "\n";
        return 1;
    }

    std::cout << "PASS: accuracy meets threshold (" << accuracy
              << " >= " << accuracy_threshold << ")\n";
    return 0;
}
