# Recommendation:
# If you are working with multiple machines, it is advisable to create separate configuration files, e.g., config_machine1.py, ..., config_machinen.py.
# Then, use different scripts to load machine-specific settings, for example:
# CONFIG_VALUES=$(python -c "from config_machine1 import dataset_root, predictor_path")

dataset_root = "/your_path/OWDFA40-Benchmark/data"
predictor_path = "/your_path/OWDFA40-Benchmark/shape_predictor_68_face_landmarks.dat"