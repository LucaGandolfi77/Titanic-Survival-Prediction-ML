/**
 * @file tree_parser.cpp
 * @brief Implementation of the host-side Decision Tree JSON parser.
 *
 * Uses nlohmann/json (single-header, bundled) for JSON parsing.
 * Builds a flat node vector and validates all structural invariants.
 *
 * @note HOST-SIDE ONLY.
 */

#include "tree_parser.hpp"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <sstream>
#include <unordered_set>

#include "nlohmann/json.hpp"

using json = nlohmann::json;

namespace avionics_dt {

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------
namespace {

/**
 * @brief Resolve a feature name to its index in the feature_names vector.
 *
 * @param feature_names  The list of feature names.
 * @param name           The feature name to look up.
 * @return int           Zero-based index, or -1 if not found.
 */
int resolve_feature_index(const std::vector<std::string>& feature_names,
                          const std::string& name) {
    for (int i = 0; i < static_cast<int>(feature_names.size()); ++i) {
        if (feature_names[static_cast<std::size_t>(i)] == name) {
            return i;
        }
    }
    return -1;
}

/**
 * @brief Resolve a class name to its index in the class_names vector.
 *
 * @param class_names  The list of class labels.
 * @param name         The class label to look up.
 * @return int         Zero-based index, or -1 if not found.
 */
int resolve_class_index(const std::vector<std::string>& class_names,
                        const std::string& name) {
    for (int i = 0; i < static_cast<int>(class_names.size()); ++i) {
        if (class_names[static_cast<std::size_t>(i)] == name) {
            return i;
        }
    }
    return -1;
}

/**
 * @brief Recursively flatten a JSON tree node into the flat node vector.
 *
 * @param j              Current JSON node object.
 * @param model          Model being constructed (nodes vector is appended).
 * @param feature_names  Feature name list for index resolution.
 * @param class_names    Class name list for index resolution.
 * @return int           Index of this node in model.nodes.
 * @throws ParseError    On malformed JSON structure.
 */
int flatten_node(const json& j, TreeModel& model,
                 const std::vector<std::string>& feature_names,
                 const std::vector<std::string>& class_names) {
    if (static_cast<int>(model.nodes.size()) >= kMaxNodeCount) {
        throw ParseError("Tree exceeds maximum node count (" +
                         std::to_string(kMaxNodeCount) + ")");
    }

    TreeNode node{};
    node.node_id = j.at("node_id").get<int>();

    const bool is_leaf = j.contains("leaf") && j.at("leaf").get<bool>();
    node.is_leaf = is_leaf;

    if (is_leaf) {
        node.feature_index = -1;
        node.threshold = 0.0f;
        node.left_child = -1;
        node.right_child = -1;

        const std::string class_name = j.at("class").get<std::string>();
        node.class_index = resolve_class_index(class_names, class_name);
        if (node.class_index < 0) {
            throw ParseError("Unknown class label '" + class_name +
                             "' at node " + std::to_string(node.node_id));
        }
    } else {
        node.class_index = -1;

        const std::string feat_name = j.at("feature").get<std::string>();
        node.feature_index = resolve_feature_index(feature_names, feat_name);
        if (node.feature_index < 0) {
            throw ParseError("Unknown feature '" + feat_name +
                             "' at node " + std::to_string(node.node_id));
        }

        node.threshold = j.at("threshold").get<float>();

        // Reserve this node's slot in the vector.
        const int my_index = static_cast<int>(model.nodes.size());
        model.nodes.push_back(node);  // placeholder — will be updated

        // Recurse children.
        if (!j.contains("left") || !j.contains("right")) {
            throw ParseError("Internal node " + std::to_string(node.node_id) +
                             " missing left or right child");
        }
        const int left_idx =
            flatten_node(j.at("left"), model, feature_names, class_names);
        const int right_idx =
            flatten_node(j.at("right"), model, feature_names, class_names);

        model.nodes[static_cast<std::size_t>(my_index)].left_child = left_idx;
        model.nodes[static_cast<std::size_t>(my_index)].right_child = right_idx;
        return my_index;
    }

    // Leaf: just append and return index.
    const int my_index = static_cast<int>(model.nodes.size());
    model.nodes.push_back(node);
    return my_index;
}

/**
 * @brief Core parsing logic from a json object.
 *
 * @param root  Top-level JSON object.
 * @return TreeModel  The parsed and validated model.
 */
TreeModel parse_json_object(const json& root) {
    TreeModel model{};

    // --- Top-level fields ---------------------------------------------------
    model.n_features = root.at("n_features").get<int>();
    if (model.n_features <= 0 || model.n_features > kMaxFeatureCount) {
        throw ParseError("n_features out of range [1, " +
                         std::to_string(kMaxFeatureCount) + "]");
    }

    for (const auto& fn : root.at("feature_names")) {
        model.feature_names.push_back(fn.get<std::string>());
    }
    if (static_cast<int>(model.feature_names.size()) != model.n_features) {
        throw ParseError("feature_names length does not match n_features");
    }

    for (const auto& cn : root.at("classes")) {
        model.class_names.push_back(cn.get<std::string>());
    }
    if (model.class_names.empty() ||
        static_cast<int>(model.class_names.size()) > kMaxClassCount) {
        throw ParseError("classes count out of range [1, " +
                         std::to_string(kMaxClassCount) + "]");
    }

    // --- Flatten tree -------------------------------------------------------
    model.root_index = flatten_node(root.at("tree"), model,
                                    model.feature_names, model.class_names);

    // --- Compute depth and validate -----------------------------------------
    model.depth = compute_depth(model, model.root_index);
    validate_tree(model);

    return model;
}

}  // anonymous namespace

// ---------------------------------------------------------------------------
// Public API implementation
// ---------------------------------------------------------------------------

TreeModel parse_tree_json(const std::string& json_path) {
    std::ifstream ifs(json_path);
    if (!ifs.is_open()) {
        throw ParseError("Cannot open file: " + json_path);
    }

    json root;
    try {
        ifs >> root;
    } catch (const json::parse_error& e) {
        throw ParseError(std::string("JSON parse error: ") + e.what());
    }

    return parse_json_object(root);
}

TreeModel parse_tree_string(const std::string& json_str) {
    json root;
    try {
        root = json::parse(json_str);
    } catch (const json::parse_error& e) {
        throw ParseError(std::string("JSON parse error: ") + e.what());
    }

    return parse_json_object(root);
}

int compute_depth(const TreeModel& model, int index) {
    if (index < 0 || index >= static_cast<int>(model.nodes.size())) {
        return 0;
    }
    const auto& node = model.nodes[static_cast<std::size_t>(index)];
    if (node.is_leaf) {
        return 0;
    }
    const int ld = compute_depth(model, node.left_child);
    const int rd = compute_depth(model, node.right_child);
    return 1 + std::max(ld, rd);
}

void validate_tree(const TreeModel& model) {
    if (model.nodes.empty()) {
        throw ParseError("Tree has no nodes");
    }

    if (model.depth > kMaxTreeDepth) {
        throw ParseError("Tree depth " + std::to_string(model.depth) +
                         " exceeds maximum " + std::to_string(kMaxTreeDepth));
    }

    if (static_cast<int>(model.nodes.size()) > kMaxNodeCount) {
        throw ParseError("Node count exceeds maximum");
    }

    // Check every node for valid references.
    const int n = static_cast<int>(model.nodes.size());
    std::unordered_set<int> visited;

    // Iterative DFS to verify reachability and acyclicity.
    std::vector<int> stack;
    stack.push_back(model.root_index);

    while (!stack.empty()) {
        const int idx = stack.back();
        stack.pop_back();

        if (idx < 0 || idx >= n) {
            throw ParseError("Node reference out of range: " +
                             std::to_string(idx));
        }
        if (visited.count(idx) != 0) {
            throw ParseError("Cycle detected at node index " +
                             std::to_string(idx));
        }
        visited.insert(idx);

        const auto& nd = model.nodes[static_cast<std::size_t>(idx)];

        if (nd.is_leaf) {
            if (nd.class_index < 0 ||
                nd.class_index >= static_cast<int>(model.class_names.size())) {
                throw ParseError("Leaf node " + std::to_string(nd.node_id) +
                                 " has invalid class_index");
            }
        } else {
            if (nd.feature_index < 0 || nd.feature_index >= model.n_features) {
                throw ParseError("Node " + std::to_string(nd.node_id) +
                                 " has invalid feature_index");
            }
            if (nd.left_child < 0 || nd.left_child >= n) {
                throw ParseError("Node " + std::to_string(nd.node_id) +
                                 " has invalid left_child");
            }
            if (nd.right_child < 0 || nd.right_child >= n) {
                throw ParseError("Node " + std::to_string(nd.node_id) +
                                 " has invalid right_child");
            }
            stack.push_back(nd.left_child);
            stack.push_back(nd.right_child);
        }
    }

    // Ensure all nodes are reachable.
    if (static_cast<int>(visited.size()) != n) {
        throw ParseError("Not all nodes are reachable from root (" +
                         std::to_string(visited.size()) + "/" +
                         std::to_string(n) + ")");
    }
}

}  // namespace avionics_dt
