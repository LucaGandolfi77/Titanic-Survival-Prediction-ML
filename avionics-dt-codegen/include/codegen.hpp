/**
 * @file codegen.hpp
 * @brief Host-side code generator that emits deterministic C++ classifier code.
 *
 * Takes the parsed TreeModel IR and emits a generated_classifier.hpp and
 * generated_classifier.cpp that satisfy DO-178C / MISRA-C++ constraints:
 *   - No dynamic memory allocation.
 *   - No recursion.
 *   - No unbounded loops.
 *   - No exceptions in generated code.
 *   - Fully unrolled if/else chain with provable WCET.
 *
 * @note HOST-SIDE ONLY.
 */

#ifndef AVIONICS_DT_CODEGEN_CODEGEN_HPP
#define AVIONICS_DT_CODEGEN_CODEGEN_HPP

#include "tree_parser.hpp"

#include <string>

namespace avionics_dt {

/**
 * @brief Configuration for the code generator.
 */
struct CodegenConfig {
    bool use_fixed_point = false;    ///< Emit int32_t comparisons instead of float.
    int fixed_point_scale = 1000;    ///< Scale factor when use_fixed_point is true.
    bool emit_likely_hints = true;   ///< Emit [[likely]]/[[unlikely]] attributes.
    std::string output_dir = ".";    ///< Directory for generated files.
    std::string namespace_name = "avionics_ml";  ///< Namespace for generated code.
};

/**
 * @brief Generate the classifier header and source files.
 *
 * @param model   Validated TreeModel.
 * @param config  Code generation configuration.
 *
 * @note Deterministic: identical input always produces byte-identical output.
 * @note Generated files: generated_classifier.hpp, generated_classifier.cpp
 *       placed in config.output_dir.
 */
void generate_classifier(const TreeModel& model, const CodegenConfig& config);

/**
 * @brief Generate code for an ensemble (random forest) of trees.
 *
 * Each tree gets its own classify_tree_N() function. A wrapper
 * classify_ensemble() calls all N and returns the majority vote.
 *
 * @param models  Vector of validated TreeModels (must share feature/class lists).
 * @param config  Code generation configuration.
 */
void generate_ensemble_classifier(const std::vector<TreeModel>& models,
                                  const CodegenConfig& config);

}  // namespace avionics_dt

#endif  // AVIONICS_DT_CODEGEN_CODEGEN_HPP
