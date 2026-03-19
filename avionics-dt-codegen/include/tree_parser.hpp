/**
 * @file tree_parser.hpp
 * @brief Host-side Decision Tree JSON parser and IR builder.
 *
 * This module reads a trained Decision Tree from a JSON file (scikit-learn
 * inspired format) and builds a flat, validated intermediate representation
 * suitable for deterministic code generation.
 *
 * @note HOST-SIDE ONLY — not intended for embedded targets.
 * @note DO-178C: Parser validates all structural invariants before any code
 *       generation step, ensuring traceability from model to generated artifact.
 *
 * Copyright 2026 — Avionics ML Codegen Project
 * SPDX-License-Identifier: MIT
 */

#ifndef AVIONICS_DT_CODEGEN_TREE_PARSER_HPP
#define AVIONICS_DT_CODEGEN_TREE_PARSER_HPP

#include <cstdint>
#include <string>
#include <vector>
#include <stdexcept>

namespace avionics_dt {

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** @brief Maximum supported tree depth (safety limit). */
constexpr int kMaxTreeDepth = 32;

/** @brief Maximum supported number of nodes in a single tree. */
constexpr int kMaxNodeCount = 100000;

/** @brief Maximum supported number of input features. */
constexpr int kMaxFeatureCount = 256;

/** @brief Maximum supported number of output classes. */
constexpr int kMaxClassCount = 64;

// ---------------------------------------------------------------------------
// Intermediate Representation
// ---------------------------------------------------------------------------

/**
 * @brief Flat representation of a single tree node.
 *
 * Leaves have is_leaf == true, feature_index == -1, left_child == -1,
 * right_child == -1.  Internal nodes have class_index == -1.
 */
struct TreeNode {
    int node_id;          ///< Original node identifier from JSON.
    bool is_leaf;         ///< True if this is a leaf (terminal) node.
    int feature_index;    ///< Index into the feature array; -1 if leaf.
    float threshold;      ///< Split threshold; unused if leaf.
    int class_index;      ///< Predicted class index; -1 if internal node.
    int left_child;       ///< Index into flat node vector; -1 if leaf.
    int right_child;      ///< Index into flat node vector; -1 if leaf.
};

/**
 * @brief Complete parsed tree model.
 */
struct TreeModel {
    int n_features;                           ///< Number of input features.
    std::vector<std::string> feature_names;   ///< Human-readable feature names.
    std::vector<std::string> class_names;     ///< Human-readable class labels.
    std::vector<TreeNode> nodes;              ///< Flat array of all tree nodes.
    int root_index;                           ///< Index of root in nodes[].
    int depth;                                ///< Computed tree depth.
};

// ---------------------------------------------------------------------------
// Parser errors
// ---------------------------------------------------------------------------

/**
 * @brief Exception thrown on parse or validation failures.
 */
class ParseError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * @brief Parse a JSON file and produce a validated TreeModel.
 *
 * @param json_path  Absolute or relative path to the JSON model file.
 * @return TreeModel Fully validated intermediate representation.
 * @throws ParseError on any structural or semantic error.
 *
 * @note Safety: every node is reachable from root; every leaf has a valid
 *       class_index; every internal node references a valid feature_index.
 */
TreeModel parse_tree_json(const std::string& json_path);

/**
 * @brief Parse a JSON string directly (useful for unit tests).
 *
 * @param json_str  JSON content as a string.
 * @return TreeModel Fully validated intermediate representation.
 * @throws ParseError on any structural or semantic error.
 */
TreeModel parse_tree_string(const std::string& json_str);

/**
 * @brief Compute the depth of the tree from the root.
 *
 * @param model  Parsed tree model.
 * @param index  Current node index (start with model.root_index).
 * @return int   Depth of the deepest path from this node.
 */
int compute_depth(const TreeModel& model, int index);

/**
 * @brief Validate structural invariants of a parsed tree.
 *
 * Checks: acyclicity, correct child references, feature/class index bounds,
 * depth <= kMaxTreeDepth, node count <= kMaxNodeCount.
 *
 * @param model  The tree model to validate.
 * @throws ParseError if any invariant is violated.
 */
void validate_tree(const TreeModel& model);

}  // namespace avionics_dt

#endif  // AVIONICS_DT_CODEGEN_TREE_PARSER_HPP
