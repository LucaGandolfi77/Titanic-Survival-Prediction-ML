# Avionics Decision Tree Code Generator

**Deterministic, auditable, MISRA-C++/DO-178C compliant Decision Tree inference for safety-critical embedded systems.**

This project takes a trained Decision Tree (or small ensemble) exported as JSON—e.g. from scikit-learn—and automatically generates highly optimized, human-readable C++ code suitable for deployment on bare-metal or RTOS microcontrollers (ARM Cortex-M4/M7). The generated code guarantees strict worst-case execution time (WCET): no dynamic memory allocation, no recursion, no unbounded loops at runtime. The toolchain also includes a static WCET analyzer and a validation harness for host-side accuracy and latency verification.

---

## Project Structure

```
avionics-dt-codegen/
├── CMakeLists.txt                   # Build system
├── README.md                        # This file
├── include/
│   ├── tree_parser.hpp              # JSON parser and IR builder
│   ├── codegen.hpp                  # C++ code generator
│   └── wcet_analyzer.hpp            # Static WCET analyzer
├── src/
│   ├── tree_parser.cpp
│   ├── codegen.cpp
│   ├── wcet_analyzer.cpp
│   ├── main_codegen.cpp             # codegen_tool entry point
│   └── validation_harness.cpp       # Host-side validation
├── tests/
│   └── test_main.cpp                # Unit tests (assert-based, no deps)
├── scripts/
│   └── export_tree.py               # Python: train & export to JSON
├── sample_data/
│   ├── sample_tree.json             # Example JSON tree
│   └── test_data.csv                # Example test dataset
└── output/                          # Generated files go here
```

---

## Build Instructions

### Prerequisites

- **Host**: GCC ≥ 9 or Clang ≥ 10, CMake ≥ 3.14
- **Cross-compile** (optional): `arm-none-eabi-g++` toolchain
- **Python** (optional, for `export_tree.py`): Python 3.8+, scikit-learn, numpy

### Host Build (Linux/macOS)

```bash
cd avionics-dt-codegen
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

This produces:
- `codegen_tool` — the code generator binary
- `unit_tests` — the test suite

### Run Unit Tests

```bash
./unit_tests
```

### Generate Classifier

```bash
./codegen_tool ../sample_data/sample_tree.json ../output --analyze-wcet
```

### Build Validation Harness (after code generation)

```bash
cd build
cmake .. -DGENERATED_DIR=$(pwd)/../output
make validation_harness
./validation_harness ../sample_data/test_data.csv --threshold 0.80
```

### Cross-Compile for ARM Cortex-M4

```bash
mkdir build-arm && cd build-arm
cmake .. \
  -DCMAKE_C_COMPILER=arm-none-eabi-gcc \
  -DCMAKE_CXX_COMPILER=arm-none-eabi-g++ \
  -DBUILD_EMBEDDED=ON \
  -DGENERATED_DIR=$(pwd)/../output \
  -DCMAKE_BUILD_TYPE=Release
make embedded_target
# Output: embedded_target.elf
arm-none-eabi-size embedded_target.elf
```

---

## End-to-End Usage Example

```bash
# 1. Train a Decision Tree in Python and export to JSON.
cd scripts
python3 export_tree.py --max-depth 6 --output ../sample_data/trained_tree.json \
                        --export-csv ../sample_data/test_data_generated.csv

# 2. Run the code generator.
cd ../build
./codegen_tool ../sample_data/trained_tree.json ../output --analyze-wcet

# 3. Rebuild to include validation harness with generated code.
cmake .. -DGENERATED_DIR=$(pwd)/../output
make validation_harness

# 4. Validate accuracy and latency.
./validation_harness ../sample_data/test_data_generated.csv --threshold 0.90

# 5. (Optional) Cross-compile for target MCU.
# See "Cross-Compile for ARM Cortex-M4" above.
```

---

## DO-178C / Safety Design Justification

| Design Decision | DO-178C Justification |
|---|---|
| **No dynamic memory allocation** in generated code | Eliminates heap fragmentation and non-deterministic `malloc` latency. Required for DAL-A/B. |
| **No recursion** in generated code | Guarantees bounded stack usage; stack depth is statically known at compile time. |
| **No unbounded loops** at runtime | The generated if/else chain has exactly as many comparisons as the tree depth — provably bounded. |
| **No exceptions** (`noexcept`, `-fno-exceptions`) | Avoids unwinding tables and non-deterministic exception propagation. MISRA-C++ 2023 Rule 18.0.1. |
| **No RTTI** (`-fno-rtti`) | Reduces code size and avoids dynamic type checks at runtime. |
| **No STL containers** in generated code | STL containers use dynamic allocation internally; replaced with plain C arrays and POD types. |
| **`[[nodiscard]]` on `classify()`** | Prevents accidental silent discard of safety-critical classification results. |
| **`static_assert` on feature count** | Compile-time guard: if the model changes, the build fails immediately rather than producing wrong results. |
| **Flat if/else chain** (no pointer-based traversal) | Eliminates pointer dereference chains; branch prediction is more effective. Code is auditable by safety engineers. |
| **Deterministic code generation** | Same JSON input → byte-identical output. Supports configuration management and reproducible builds (DO-178C §12.1.3). |
| **Static WCET analysis** | Provides auditable evidence for worst-case timing; required for scheduling analysis in RTOS environments. |
| **Human-readable generated code** | The generated `.cpp` file can be reviewed by a safety engineer without specialized tools. Supports code review objectives of DO-178C §6.3.4. |
| **Fixed-point alternative** (`USE_FIXED_POINT`) | Enables deployment on MCUs without FPU (e.g. Cortex-M0/M3) while maintaining deterministic behavior. |

---

## Extending to Random Forest (Ensemble)

The code generator supports ensemble (Random Forest) models through `generate_ensemble_classifier()`. To use it:

1. **Export multiple trees**: Modify `export_tree.py` to train a `RandomForestClassifier` and export each `estimator_` as a separate JSON tree, or combine them into a JSON array.

2. **Parse all trees**:
   ```cpp
   std::vector<avionics_dt::TreeModel> models;
   for (const auto& path : tree_json_paths) {
       models.push_back(avionics_dt::parse_tree_json(path));
   }
   ```

3. **Generate ensemble code**:
   ```cpp
   avionics_dt::generate_ensemble_classifier(models, config);
   ```

4. **Generated structure**: Each tree gets its own `classify_tree_N()` function (where N is the tree index). A wrapper `classify()` calls all N trees and returns the majority-vote result using a fixed-size vote counter array — no dynamic allocation.

5. **WCET for ensemble**: The ensemble WCET is the sum of individual tree WCETs plus the majority-vote overhead (O(N × C) where C is the number of classes).

---

## MISRA-C++ 2023 Compliance Notes

The generated code targets full MISRA-C++ 2023 compliance. Known deviations:

| Rule | Deviation | Rationale |
|---|---|---|
| Rule 6.0.1 (plain char) | `const char* const` arrays for feature/class names | Required for string literal storage; read-only usage. |
| Rule 11.3.1 (narrowing) | `static_cast<uint8_t>(Label)` | Enum underlying type is `uint8_t`; cast is safe by construction. |

All deviations are documented and justified per MISRA Deviation Permit process.

---

## License

MIT License. See source file headers.
