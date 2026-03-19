/**
 * @file main_codegen.cpp
 * @brief Entry point for the Decision Tree code generation tool.
 *
 * Usage:
 *   codegen_tool <input_tree.json> <output_dir> [--analyze-wcet] [--fixed-point] [--scale N]
 *
 * Workflow:
 *   1. Parse and validate the JSON tree model.
 *   2. Generate deterministic C++ classifier source files.
 *   3. Optionally perform WCET analysis and write report.
 *
 * @note HOST-SIDE ONLY — this binary runs on the development PC.
 * @note DO-178C: all steps are logged to stdout for audit trail.
 */

#include "tree_parser.hpp"
#include "codegen.hpp"
#include "wcet_analyzer.hpp"

#include <cstring>
#include <iostream>
#include <string>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** @brief Tool version string. */
constexpr const char* kToolVersion = "1.0.0";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace {

/**
 * @brief Print usage information to stderr.
 *
 * @param prog  Program name (argv[0]).
 */
void print_usage(const char* prog) {
    std::cerr
        << "Avionics Decision Tree Code Generator v" << kToolVersion << "\n\n"
        << "Usage:\n"
        << "  " << prog
        << " <input_tree.json> <output_dir> [options]\n\n"
        << "Options:\n"
        << "  --analyze-wcet   Run static WCET analysis and write report.\n"
        << "  --fixed-point    Generate int32_t comparisons (for MCUs without FPU).\n"
        << "  --scale N        Fixed-point scale factor (default: 1000).\n"
        << "  --no-likely      Omit [[likely]]/[[unlikely]] attributes.\n"
        << "  --help           Show this message.\n";
}

}  // anonymous namespace

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    // --- Parse CLI arguments ------------------------------------------------
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    std::string input_path;
    std::string output_dir;
    bool analyze_wcet = false;
    avionics_dt::CodegenConfig config{};

    // First two positional arguments.
    input_path = argv[1];
    output_dir = argv[2];
    config.output_dir = output_dir;

    for (int i = 3; i < argc; ++i) {
        if (std::strcmp(argv[i], "--analyze-wcet") == 0) {
            analyze_wcet = true;
        } else if (std::strcmp(argv[i], "--fixed-point") == 0) {
            config.use_fixed_point = true;
        } else if (std::strcmp(argv[i], "--scale") == 0 && i + 1 < argc) {
            config.fixed_point_scale = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--no-likely") == 0) {
            config.emit_likely_hints = false;
        } else if (std::strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown option: " << argv[i] << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    // --- Step 1: Parse and validate -----------------------------------------
    std::cout << "[1/3] Parsing " << input_path << " ...\n";
    avionics_dt::TreeModel model;
    try {
        model = avionics_dt::parse_tree_json(input_path);
    } catch (const avionics_dt::ParseError& e) {
        std::cerr << "PARSE ERROR: " << e.what() << "\n";
        return 1;
    }

    std::cout << "      Features : " << model.n_features << "\n"
              << "      Classes  : " << model.class_names.size() << "\n"
              << "      Nodes    : " << model.nodes.size() << "\n"
              << "      Depth    : " << model.depth << "\n\n";

    // --- Step 2: Generate code ----------------------------------------------
    std::cout << "[2/3] Generating classifier in " << output_dir << " ...\n";
    try {
        avionics_dt::generate_classifier(model, config);
    } catch (const std::exception& e) {
        std::cerr << "CODEGEN ERROR: " << e.what() << "\n";
        return 1;
    }
    std::cout << "      -> generated_classifier.hpp\n"
              << "      -> generated_classifier.cpp\n\n";

    // --- Step 3: WCET analysis (optional) -----------------------------------
    if (analyze_wcet) {
        std::cout << "[3/3] Running WCET analysis ...\n";
        const auto report = avionics_dt::analyze_wcet(model);
        const std::string report_path = output_dir + "/wcet_report.txt";
        avionics_dt::write_wcet_report(report, model, report_path);

        std::cout << avionics_dt::format_wcet_report(report, model);
        std::cout << "      -> " << report_path << "\n\n";
    } else {
        std::cout << "[3/3] WCET analysis skipped (use --analyze-wcet).\n\n";
    }

    std::cout << "Done. Generated files are ready in " << output_dir << "/\n";
    return 0;
}
