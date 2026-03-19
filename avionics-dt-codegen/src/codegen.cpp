/**
 * @file codegen.cpp
 * @brief Implementation of the deterministic C++ code generator.
 *
 * Emits human-readable, auditable C++ source that implements a Decision
 * Tree classifier as a flat if/else chain with no dynamic allocation,
 * no recursion, and no unbounded loops at runtime.
 *
 * @note HOST-SIDE ONLY.
 */

#include "codegen.hpp"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace avionics_dt {

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------
namespace {

/** @brief Canonical date string for the generation timestamp. */
std::string current_date_string() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    struct tm buf{};
    gmtime_r(&t, &buf);
    char date_buf[16];
    std::strftime(date_buf, sizeof(date_buf), "%Y-%m-%d", &buf);
    return std::string(date_buf);
}

/**
 * @brief Convert a class name to a valid C++ enum identifier.
 *
 * Replaces any non-alphanumeric character with '_' and uppercases.
 *
 * @param name  Original class name (e.g. "WARN_VIBRATION").
 * @return std::string  Sanitised enum identifier.
 */
std::string sanitize_enum_name(const std::string& name) {
    std::string result;
    result.reserve(name.size());
    for (char c : name) {
        if (std::isalnum(static_cast<unsigned char>(c))) {
            result += static_cast<char>(
                std::toupper(static_cast<unsigned char>(c)));
        } else {
            result += '_';
        }
    }
    return result;
}

/**
 * @brief Emit a single indent of the given depth.
 *
 * @param depth  Nesting level (each level = 4 spaces).
 * @return std::string  The indent string.
 */
std::string indent(int depth) {
    constexpr int kSpacesPerIndent = 4;
    return std::string(static_cast<std::size_t>(depth * kSpacesPerIndent), ' ');
}

/**
 * @brief Format a float threshold as a C literal.
 *
 * @param value  Threshold value.
 * @return std::string  e.g. "650.5f".
 */
std::string float_literal(float value) {
    std::ostringstream oss;
    oss << std::setprecision(10) << value;
    std::string s = oss.str();
    // Ensure the literal contains a decimal point (valid C++ float literal).
    if (s.find('.') == std::string::npos && s.find('e') == std::string::npos &&
        s.find('E') == std::string::npos) {
        s += ".0";
    }
    s += 'f';
    return s;
}

/**
 * @brief Format a fixed-point threshold as an int32_t literal.
 *
 * @param value  Original float threshold.
 * @param scale  Fixed-point scale factor.
 * @return std::string  e.g. "650500".
 */
std::string fixed_literal(float value, int scale) {
    const auto scaled = static_cast<int32_t>(value * static_cast<float>(scale));
    return std::to_string(scaled);
}

/**
 * @brief Recursively emit if/else code for a tree node (host-side recursion
 *        only — the *generated* code has no recursion).
 *
 * @param model     The tree model.
 * @param idx       Current node index.
 * @param depth     Current nesting depth for indentation.
 * @param config    Codegen config.
 * @param out       Output stream.
 */
void emit_node(const TreeModel& model, int idx, int depth,
               const CodegenConfig& config, std::ostream& out) {
    const auto& node = model.nodes[static_cast<std::size_t>(idx)];

    if (node.is_leaf) {
        const auto& class_name = model.class_names[static_cast<std::size_t>(
            node.class_index)];
        out << indent(depth) << "return Label::"
            << sanitize_enum_name(class_name) << ";  // Node "
            << node.node_id << " (leaf)\n";
        return;
    }

    const auto& feat_name = model.feature_names[static_cast<std::size_t>(
        node.feature_index)];

    out << indent(depth) << "// Node " << node.node_id << ": "
        << feat_name << " <= " << node.threshold << "\n";

    // Build comparison expression.
    std::string cmp_expr;
    if (config.use_fixed_point) {
        cmp_expr = "features[" + std::to_string(node.feature_index) +
                   "] <= " + fixed_literal(node.threshold,
                                           config.fixed_point_scale);
    } else {
        cmp_expr = "features[" + std::to_string(node.feature_index) +
                   "] <= " + float_literal(node.threshold);
    }

    std::string likely_hint;
    if (config.emit_likely_hints) {
        likely_hint = " [[likely]]";
    }

    out << indent(depth) << "if (" << cmp_expr << ")" << likely_hint
        << " {\n";
    emit_node(model, node.left_child, depth + 1, config, out);
    out << indent(depth) << "} else {\n";
    emit_node(model, node.right_child, depth + 1, config, out);
    out << indent(depth) << "}\n";
}

}  // anonymous namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

void generate_classifier(const TreeModel& model, const CodegenConfig& config) {
    const std::string date = current_date_string();
    const int node_count = static_cast<int>(model.nodes.size());
    const std::string ns = config.namespace_name;

    // Determine the value type used in the features array.
    const std::string value_type =
        config.use_fixed_point ? "int32_t" : "float";
    const std::string value_header =
        config.use_fixed_point ? "<cstdint>" : "";

    // -----------------------------------------------------------------------
    // HEADER FILE
    // -----------------------------------------------------------------------
    {
        const std::string header_path =
            config.output_dir + "/generated_classifier.hpp";
        std::ofstream hdr(header_path);
        if (!hdr.is_open()) {
            throw std::runtime_error("Cannot open " + header_path +
                                     " for writing");
        }

        hdr << "/**\n"
            << " * @file generated_classifier.hpp\n"
            << " * @brief AUTO-GENERATED Decision Tree classifier — DO NOT EDIT.\n"
            << " *\n"
            << " * Tree depth : " << model.depth << "\n"
            << " * Nodes      : " << node_count << "\n"
            << " * Features   : " << model.n_features << "\n"
            << " * Classes    : " << model.class_names.size() << "\n"
            << " * Generated  : " << date << "\n"
            << " *\n"
            << " * WCET bound : " << model.depth
            << " comparisons (== tree depth)\n"
            << " *\n"
            << " * @note No dynamic allocation. No recursion. No exceptions.\n"
            << " * @note MISRA-C++ 2023 compliant generated code.\n"
            << " */\n\n"
            << "#ifndef GENERATED_CLASSIFIER_HPP\n"
            << "#define GENERATED_CLASSIFIER_HPP\n\n"
            << "#include <cstdint>\n";
        if (!value_header.empty()) {
            hdr << "#include " << value_header << "\n";
        }
        hdr << "\n";

        // Feature count constant.
        hdr << "/** @brief Number of input features expected by the classifier. */\n"
            << "constexpr int kNumFeatures = " << model.n_features << ";\n\n";

        // Fixed-point scale (if applicable).
        if (config.use_fixed_point) {
            hdr << "/** @brief Fixed-point scale factor. */\n"
                << "constexpr int kFixedPointScale = "
                << config.fixed_point_scale << ";\n\n";
        }

        hdr << "namespace " << ns << " {\n\n";

        // Enum class for labels.
        hdr << "/**\n"
            << " * @brief Classification result labels.\n"
            << " */\n"
            << "enum class Label : uint8_t {\n";
        for (std::size_t i = 0; i < model.class_names.size(); ++i) {
            hdr << "    " << sanitize_enum_name(model.class_names[i])
                << " = " << i;
            if (i + 1 < model.class_names.size()) {
                hdr << ",";
            }
            hdr << "\n";
        }
        hdr << "};\n\n";

        // Number of classes constant.
        hdr << "/** @brief Total number of output classes. */\n"
            << "constexpr uint8_t kNumClasses = "
            << model.class_names.size() << "U;\n\n";

        // Classify function declaration.
        hdr << "/**\n"
            << " * @brief Classify an input feature vector.\n"
            << " *\n"
            << " * @param features  Array of " << model.n_features
            << " " << value_type << " feature values.\n"
            << " * @return Label    Predicted class label.\n"
            << " *\n"
            << " * @note noexcept — never throws.\n"
            << " * @note WCET: at most " << model.depth
            << " comparisons.\n"
            << " */\n"
            << "[[nodiscard]] Label classify(const " << value_type
            << " features[" << model.n_features << "]) noexcept;\n\n";

        // Feature names array for diagnostics.
        hdr << "/** @brief Human-readable feature names (for diagnostics). */\n"
            << "extern const char* const kFeatureNames[" << model.n_features
            << "];\n\n";

        // Class names array for diagnostics.
        hdr << "/** @brief Human-readable class labels (for diagnostics). */\n"
            << "extern const char* const kClassNames["
            << model.class_names.size() << "];\n\n";

        hdr << "}  // namespace " << ns << "\n\n"
            << "#endif  // GENERATED_CLASSIFIER_HPP\n";
    }

    // -----------------------------------------------------------------------
    // SOURCE FILE
    // -----------------------------------------------------------------------
    {
        const std::string src_path =
            config.output_dir + "/generated_classifier.cpp";
        std::ofstream src(src_path);
        if (!src.is_open()) {
            throw std::runtime_error("Cannot open " + src_path +
                                     " for writing");
        }

        src << "/**\n"
            << " * @file generated_classifier.cpp\n"
            << " * @brief AUTO-GENERATED Decision Tree classifier — DO NOT EDIT.\n"
            << " *\n"
            << " * Tree depth : " << model.depth << "\n"
            << " * Nodes      : " << node_count << "\n"
            << " * Generated  : " << date << "\n"
            << " * WCET bound : " << model.depth
            << " comparisons (== tree depth)\n"
            << " */\n\n"
            << "#include \"generated_classifier.hpp\"\n\n";

        // Static assert on feature count.
        src << "static_assert(kNumFeatures == " << model.n_features
            << ", \"Feature count mismatch — regenerate classifier.\");\n\n";

        src << "namespace " << ns << " {\n\n";

        // Feature names.
        src << "const char* const kFeatureNames[" << model.n_features
            << "] = {\n";
        for (std::size_t i = 0; i < model.feature_names.size(); ++i) {
            src << "    \"" << model.feature_names[i] << "\"";
            if (i + 1 < model.feature_names.size()) {
                src << ",";
            }
            src << "\n";
        }
        src << "};\n\n";

        // Class names.
        src << "const char* const kClassNames[" << model.class_names.size()
            << "] = {\n";
        for (std::size_t i = 0; i < model.class_names.size(); ++i) {
            src << "    \"" << model.class_names[i] << "\"";
            if (i + 1 < model.class_names.size()) {
                src << ",";
            }
            src << "\n";
        }
        src << "};\n\n";

        // Classify function.
        src << "[[nodiscard]] Label classify(const " << value_type
            << " features[" << model.n_features << "]) noexcept {\n";

        emit_node(model, model.root_index, 1, config, src);

        src << "}\n\n";

        src << "}  // namespace " << ns << "\n";
    }
}

void generate_ensemble_classifier(const std::vector<TreeModel>& models,
                                  const CodegenConfig& config) {
    if (models.empty()) {
        throw std::runtime_error("No models provided for ensemble generation");
    }

    const auto& ref = models[0];
    const std::string date = current_date_string();
    const std::string ns = config.namespace_name;
    const std::string value_type =
        config.use_fixed_point ? "int32_t" : "float";
    const int n_trees = static_cast<int>(models.size());

    // -----------------------------------------------------------------------
    // HEADER
    // -----------------------------------------------------------------------
    {
        const std::string header_path =
            config.output_dir + "/generated_classifier.hpp";
        std::ofstream hdr(header_path);
        if (!hdr.is_open()) {
            throw std::runtime_error("Cannot open " + header_path);
        }

        hdr << "/**\n"
            << " * @file generated_classifier.hpp\n"
            << " * @brief AUTO-GENERATED Random Forest classifier — DO NOT EDIT.\n"
            << " *\n"
            << " * Trees     : " << n_trees << "\n"
            << " * Features  : " << ref.n_features << "\n"
            << " * Classes   : " << ref.class_names.size() << "\n"
            << " * Generated : " << date << "\n"
            << " */\n\n"
            << "#ifndef GENERATED_CLASSIFIER_HPP\n"
            << "#define GENERATED_CLASSIFIER_HPP\n\n"
            << "#include <cstdint>\n\n"
            << "constexpr int kNumFeatures = " << ref.n_features << ";\n"
            << "constexpr int kNumTrees = " << n_trees << ";\n\n"
            << "namespace " << ns << " {\n\n";

        // Enum.
        hdr << "enum class Label : uint8_t {\n";
        for (std::size_t i = 0; i < ref.class_names.size(); ++i) {
            hdr << "    " << sanitize_enum_name(ref.class_names[i])
                << " = " << i;
            if (i + 1 < ref.class_names.size()) hdr << ",";
            hdr << "\n";
        }
        hdr << "};\n\n"
            << "constexpr uint8_t kNumClasses = "
            << ref.class_names.size() << "U;\n\n";

        // Per-tree declarations.
        for (int t = 0; t < n_trees; ++t) {
            hdr << "[[nodiscard]] Label classify_tree_" << t
                << "(const " << value_type << " features["
                << ref.n_features << "]) noexcept;\n";
        }
        hdr << "\n";

        // Ensemble entry point.
        hdr << "/**\n"
            << " * @brief Majority-vote ensemble classifier.\n"
            << " */\n"
            << "[[nodiscard]] Label classify(const " << value_type
            << " features[" << ref.n_features << "]) noexcept;\n\n"
            << "extern const char* const kFeatureNames["
            << ref.n_features << "];\n"
            << "extern const char* const kClassNames["
            << ref.class_names.size() << "];\n\n"
            << "}  // namespace " << ns << "\n\n"
            << "#endif  // GENERATED_CLASSIFIER_HPP\n";
    }

    // -----------------------------------------------------------------------
    // SOURCE
    // -----------------------------------------------------------------------
    {
        const std::string src_path =
            config.output_dir + "/generated_classifier.cpp";
        std::ofstream src(src_path);
        if (!src.is_open()) {
            throw std::runtime_error("Cannot open " + src_path);
        }

        src << "/**\n"
            << " * @file generated_classifier.cpp\n"
            << " * @brief AUTO-GENERATED Random Forest classifier — DO NOT EDIT.\n"
            << " */\n\n"
            << "#include \"generated_classifier.hpp\"\n\n"
            << "static_assert(kNumFeatures == " << ref.n_features
            << ", \"Feature count mismatch.\");\n\n"
            << "namespace " << ns << " {\n\n";

        // Feature names.
        src << "const char* const kFeatureNames[" << ref.n_features
            << "] = {\n";
        for (std::size_t i = 0; i < ref.feature_names.size(); ++i) {
            src << "    \"" << ref.feature_names[i] << "\"";
            if (i + 1 < ref.feature_names.size()) src << ",";
            src << "\n";
        }
        src << "};\n\n";

        // Class names.
        src << "const char* const kClassNames[" << ref.class_names.size()
            << "] = {\n";
        for (std::size_t i = 0; i < ref.class_names.size(); ++i) {
            src << "    \"" << ref.class_names[i] << "\"";
            if (i + 1 < ref.class_names.size()) src << ",";
            src << "\n";
        }
        src << "};\n\n";

        // Per-tree classify functions.
        for (int t = 0; t < n_trees; ++t) {
            src << "[[nodiscard]] Label classify_tree_" << t
                << "(const " << value_type << " features["
                << ref.n_features << "]) noexcept {\n";
            emit_node(models[static_cast<std::size_t>(t)],
                      models[static_cast<std::size_t>(t)].root_index,
                      1, config, src);
            src << "}\n\n";
        }

        // Majority vote.
        src << "[[nodiscard]] Label classify(const " << value_type
            << " features[" << ref.n_features << "]) noexcept {\n"
            << "    uint8_t votes[kNumClasses] = {};\n";
        for (int t = 0; t < n_trees; ++t) {
            src << "    votes[static_cast<uint8_t>(classify_tree_"
                << t << "(features))]++;\n";
        }
        src << "    uint8_t best = 0U;\n"
            << "    for (uint8_t i = 1U; i < kNumClasses; ++i) {\n"
            << "        if (votes[i] > votes[best]) {\n"
            << "            best = i;\n"
            << "        }\n"
            << "    }\n"
            << "    return static_cast<Label>(best);\n"
            << "}\n\n"
            << "}  // namespace " << ns << "\n";
    }
}

}  // namespace avionics_dt
