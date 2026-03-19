# CMake generated Testfile for 
# Source directory: /workspaces/Titanic-Survival-Prediction-ML/avionics-dt-codegen
# Build directory: /workspaces/Titanic-Survival-Prediction-ML/avionics-dt-codegen/build2
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(avionics_dt_unit_tests "/workspaces/Titanic-Survival-Prediction-ML/avionics-dt-codegen/build2/unit_tests")
set_tests_properties(avionics_dt_unit_tests PROPERTIES  _BACKTRACE_TRIPLES "/workspaces/Titanic-Survival-Prediction-ML/avionics-dt-codegen/CMakeLists.txt;86;add_test;/workspaces/Titanic-Survival-Prediction-ML/avionics-dt-codegen/CMakeLists.txt;0;")
add_test(avionics_dt_validation "/workspaces/Titanic-Survival-Prediction-ML/avionics-dt-codegen/build2/validation_harness" "/workspaces/Titanic-Survival-Prediction-ML/avionics-dt-codegen/sample_data/test_data.csv" "--threshold" "0.95")
set_tests_properties(avionics_dt_validation PROPERTIES  _BACKTRACE_TRIPLES "/workspaces/Titanic-Survival-Prediction-ML/avionics-dt-codegen/CMakeLists.txt;112;add_test;/workspaces/Titanic-Survival-Prediction-ML/avionics-dt-codegen/CMakeLists.txt;0;")
subdirs("_deps/nlohmann_json-build")
