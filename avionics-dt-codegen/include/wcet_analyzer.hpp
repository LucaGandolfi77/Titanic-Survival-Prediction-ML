/**
 * @file wcet_analyzer.hpp
 * @brief Host-side static Worst-Case Execution Time (WCET) analyzer.
 *
 * Traverses the Decision Tree IR and computes exact worst-case, best-case,
 * and average-case comparison counts.  Optionally estimates cycle counts
 * for a target ARM Cortex-M4 micro-architecture.
 *
 * @note HOST-SIDE ONLY.
 * @note DO-178C: provides deterministic, auditable WCET evidence for
 *       the generated classifier.
 */

#ifndef AVIONICS_DT_CODEGEN_WCET_ANALYZER_HPP
#define AVIONICS_DT_CODEGEN_WCET_ANALYZER_HPP

#include "tree_parser.hpp"

#include <string>

namespace avionics_dt {

// ---------------------------------------------------------------------------
// ARM Cortex-M4 micro-architecture cost model (configurable)
// ---------------------------------------------------------------------------

/**
 * @brief Cycle cost model for WCET estimation on a target MCU.
 *
 * Default values assume a Cortex-M4 with FPU (single-precision).
 */
struct CycleModel {
    int float_compare_cycles = 1;   ///< Cycles for a VCMP.F32 instruction.
    int branch_cycles = 2;          ///< Cycles for a conditional branch (taken).
    int memory_access_penalty = 1;  ///< Additional cycles per memory access.
    int function_call_overhead = 4; ///< BL + stack frame setup.
    int function_return_overhead = 4; ///< Stack restore + BX LR.
};

/**
 * @brief Results of the WCET analysis.
 */
struct WcetReport {
    int tree_depth;           ///< Maximum depth of the tree.
    int total_nodes;          ///< Total number of nodes.
    int leaf_count;           ///< Number of leaf nodes.
    int internal_count;       ///< Number of internal (split) nodes.

    int worst_case_comparisons;  ///< Max comparisons on any path (== depth).
    int best_case_comparisons;   ///< Min comparisons on any root-to-leaf path.
    double avg_comparisons;      ///< Average comparisons across all leaves.

    int worst_case_cycles;    ///< Estimated worst-case cycle count.
    int best_case_cycles;     ///< Estimated best-case cycle count.
    double avg_cycles;        ///< Estimated average cycle count.
};

/**
 * @brief Perform static WCET analysis on a parsed tree model.
 *
 * @param model       Validated TreeModel.
 * @param cycle_model Target MCU cycle cost model.
 * @return WcetReport Analysis results.
 */
WcetReport analyze_wcet(const TreeModel& model,
                        const CycleModel& cycle_model = CycleModel{});

/**
 * @brief Write a human-readable WCET report to a file.
 *
 * @param report     Analysis results.
 * @param model      The tree model (for feature/class metadata).
 * @param file_path  Output file path (e.g. "wcet_report.txt").
 */
void write_wcet_report(const WcetReport& report,
                       const TreeModel& model,
                       const std::string& file_path);

/**
 * @brief Format the WCET report as a string (for stdout).
 *
 * @param report  Analysis results.
 * @param model   The tree model.
 * @return std::string  Formatted report text.
 */
std::string format_wcet_report(const WcetReport& report,
                               const TreeModel& model);

}  // namespace avionics_dt

#endif  // AVIONICS_DT_CODEGEN_WCET_ANALYZER_HPP
