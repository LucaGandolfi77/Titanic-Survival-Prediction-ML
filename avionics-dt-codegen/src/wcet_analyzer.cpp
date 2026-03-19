/**
 * @file wcet_analyzer.cpp
 * @brief Implementation of the static WCET analyzer.
 *
 * Traverses every root-to-leaf path in the Decision Tree IR, computing
 * exact comparison counts and estimated cycle counts using a configurable
 * micro-architecture cost model.
 *
 * @note HOST-SIDE ONLY.
 */

#include "wcet_analyzer.hpp"

#include <algorithm>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace avionics_dt {

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------
namespace {

/**
 * @brief Gather the depth of every leaf (root-to-leaf path length).
 *
 * @param model        The tree model.
 * @param node_index   Current node.
 * @param current_depth  Depth accumulated so far.
 * @param depths       Output vector of leaf depths.
 */
void collect_leaf_depths(const TreeModel& model, int node_index,
                         int current_depth, std::vector<int>& depths) {
    const auto& node = model.nodes[static_cast<std::size_t>(node_index)];
    if (node.is_leaf) {
        depths.push_back(current_depth);
        return;
    }
    collect_leaf_depths(model, node.left_child, current_depth + 1, depths);
    collect_leaf_depths(model, node.right_child, current_depth + 1, depths);
}

/**
 * @brief Compute the cycle count for a given number of comparisons.
 *
 * @param comparisons  Number of float comparisons on the path.
 * @param cm           Cycle cost model.
 * @return int         Estimated cycle count.
 */
int estimate_cycles(int comparisons, const CycleModel& cm) {
    // Each comparison: 1 memory load + 1 float compare + 1 branch.
    const int per_comparison =
        cm.memory_access_penalty + cm.float_compare_cycles + cm.branch_cycles;
    return cm.function_call_overhead +
           comparisons * per_comparison +
           cm.function_return_overhead;
}

}  // anonymous namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

WcetReport analyze_wcet(const TreeModel& model, const CycleModel& cycle_model) {
    WcetReport report{};

    report.total_nodes = static_cast<int>(model.nodes.size());
    report.tree_depth = model.depth;

    // Count leaves vs. internal nodes.
    report.leaf_count = 0;
    report.internal_count = 0;
    for (const auto& nd : model.nodes) {
        if (nd.is_leaf) {
            ++report.leaf_count;
        } else {
            ++report.internal_count;
        }
    }

    // Gather all leaf depths.
    std::vector<int> leaf_depths;
    leaf_depths.reserve(static_cast<std::size_t>(report.leaf_count));
    collect_leaf_depths(model, model.root_index, 0, leaf_depths);

    // Comparison statistics.
    report.worst_case_comparisons =
        *std::max_element(leaf_depths.begin(), leaf_depths.end());
    report.best_case_comparisons =
        *std::min_element(leaf_depths.begin(), leaf_depths.end());

    const double sum = std::accumulate(leaf_depths.begin(),
                                       leaf_depths.end(), 0.0);
    report.avg_comparisons = sum / static_cast<double>(leaf_depths.size());

    // Cycle estimates.
    report.worst_case_cycles =
        estimate_cycles(report.worst_case_comparisons, cycle_model);
    report.best_case_cycles =
        estimate_cycles(report.best_case_comparisons, cycle_model);
    report.avg_cycles =
        static_cast<double>(estimate_cycles(0, cycle_model)) +
        report.avg_comparisons *
            static_cast<double>(cycle_model.memory_access_penalty +
                                cycle_model.float_compare_cycles +
                                cycle_model.branch_cycles);

    return report;
}

std::string format_wcet_report(const WcetReport& report,
                               const TreeModel& model) {
    std::ostringstream oss;

    oss << "========================================\n"
        << "  WCET ANALYSIS REPORT\n"
        << "========================================\n\n";

    oss << "Tree Structure\n"
        << "  Total nodes      : " << report.total_nodes << "\n"
        << "  Internal nodes   : " << report.internal_count << "\n"
        << "  Leaf nodes       : " << report.leaf_count << "\n"
        << "  Tree depth       : " << report.tree_depth << "\n"
        << "  Features         : " << model.n_features << "\n"
        << "  Classes          : " << model.class_names.size() << "\n\n";

    oss << "Comparison Count (root-to-leaf paths)\n"
        << "  Worst case       : " << report.worst_case_comparisons << "\n"
        << "  Best case        : " << report.best_case_comparisons << "\n"
        << "  Average          : " << report.avg_comparisons << "\n\n";

    oss << "Estimated Cycle Count (ARM Cortex-M4)\n"
        << "  Worst case       : " << report.worst_case_cycles << " cycles\n"
        << "  Best case        : " << report.best_case_cycles << " cycles\n"
        << "  Average          : " << report.avg_cycles << " cycles\n\n";

    oss << "Features:\n";
    for (std::size_t i = 0; i < model.feature_names.size(); ++i) {
        oss << "  [" << i << "] " << model.feature_names[i] << "\n";
    }
    oss << "\nClasses:\n";
    for (std::size_t i = 0; i < model.class_names.size(); ++i) {
        oss << "  [" << i << "] " << model.class_names[i] << "\n";
    }

    oss << "\n========================================\n";
    return oss.str();
}

void write_wcet_report(const WcetReport& report,
                       const TreeModel& model,
                       const std::string& file_path) {
    std::ofstream ofs(file_path);
    if (!ofs.is_open()) {
        throw std::runtime_error("Cannot open " + file_path + " for writing");
    }
    ofs << format_wcet_report(report, model);
}

}  // namespace avionics_dt
