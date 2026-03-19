/**
 * @file generated_classifier.hpp
 * @brief AUTO-GENERATED Decision Tree classifier — DO NOT EDIT.
 *
 * Tree depth : 4
 * Nodes      : 13
 * Features   : 8
 * Classes    : 4
 * Generated  : 2026-03-19
 *
 * WCET bound : 4 comparisons (== tree depth)
 *
 * @note No dynamic allocation. No recursion. No exceptions.
 * @note MISRA-C++ 2023 compliant generated code.
 */

#ifndef GENERATED_CLASSIFIER_HPP
#define GENERATED_CLASSIFIER_HPP

#include <cstdint>

/** @brief Number of input features expected by the classifier. */
constexpr int kNumFeatures = 8;

namespace avionics_ml {

/**
 * @brief Classification result labels.
 */
enum class Label : uint8_t {
    NOMINAL = 0,
    WARN_VIBRATION = 1,
    WARN_THERMAL = 2,
    FAULT_CRITICAL = 3
};

/** @brief Total number of output classes. */
constexpr uint8_t kNumClasses = 4U;

/**
 * @brief Classify an input feature vector.
 *
 * @param features  Array of 8 float feature values.
 * @return Label    Predicted class label.
 *
 * @note noexcept — never throws.
 * @note WCET: at most 4 comparisons.
 */
[[nodiscard]] Label classify(const float features[8]) noexcept;

/** @brief Human-readable feature names (for diagnostics). */
extern const char* const kFeatureNames[8];

/** @brief Human-readable class labels (for diagnostics). */
extern const char* const kClassNames[4];

}  // namespace avionics_ml

#endif  // GENERATED_CLASSIFIER_HPP
