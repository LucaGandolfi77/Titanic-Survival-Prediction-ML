/**
 * @file generated_classifier.cpp
 * @brief AUTO-GENERATED Decision Tree classifier — DO NOT EDIT.
 *
 * Tree depth : 4
 * Nodes      : 13
 * Generated  : 2026-03-19
 * WCET bound : 4 comparisons (== tree depth)
 */

#include "generated_classifier.hpp"

static_assert(kNumFeatures == 8, "Feature count mismatch — regenerate classifier.");

namespace avionics_ml {

const char* const kFeatureNames[8] = {
    "T_exhaust",
    "P_inlet",
    "RPM",
    "vibration_x",
    "vibration_y",
    "oil_temp",
    "fuel_flow",
    "EGT"
};

const char* const kClassNames[4] = {
    "NOMINAL",
    "WARN_VIBRATION",
    "WARN_THERMAL",
    "FAULT_CRITICAL"
};

[[nodiscard]] Label classify(const float features[8]) noexcept {
    // Node 0: T_exhaust <= 650.5
    if (features[0] <= 650.5f) [[likely]] {
        // Node 1: RPM <= 8200
        if (features[2] <= 8200.0f) [[likely]] {
            // Node 3: vibration_x <= 2.5
            if (features[3] <= 2.5f) [[likely]] {
                return Label::NOMINAL;  // Node 5 (leaf)
            } else {
                // Node 6: oil_temp <= 120
                if (features[5] <= 120.0f) [[likely]] {
                    return Label::WARN_VIBRATION;  // Node 9 (leaf)
                } else {
                    return Label::FAULT_CRITICAL;  // Node 10 (leaf)
                }
            }
        } else {
            // Node 4: EGT <= 850
            if (features[7] <= 850.0f) [[likely]] {
                return Label::WARN_VIBRATION;  // Node 7 (leaf)
            } else {
                return Label::FAULT_CRITICAL;  // Node 8 (leaf)
            }
        }
    } else {
        // Node 2: EGT <= 900
        if (features[7] <= 900.0f) [[likely]] {
            return Label::WARN_THERMAL;  // Node 11 (leaf)
        } else {
            return Label::FAULT_CRITICAL;  // Node 12 (leaf)
        }
    }
}

}  // namespace avionics_ml
