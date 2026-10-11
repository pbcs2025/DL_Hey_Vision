import tensorflow as tf

from src.cnn.pooling_experiment import (
    build_max_pool_model,
    build_average_pool_model,
    create_sample_feature_map
)


print("\nAI Vision Assistant - CNN Pooling Experiment")
print("=" * 65)


# --------------------------------------------------
# 1. Create feature map
# --------------------------------------------------

feature_map = create_sample_feature_map()

print("\nInput Feature Map")
print("-" * 65)
print(feature_map.numpy())


# Add batch and channel dimensions:
# (8, 8) -> (1, 8, 8, 1)

input_tensor = tf.reshape(
    feature_map,
    (1, 8, 8, 1)
)


# --------------------------------------------------
# 2. Max Pooling
# --------------------------------------------------

max_model = build_max_pool_model()

max_output = max_model(input_tensor)

print("\nMax Pooling Output")
print("-" * 65)
print(max_output.numpy().squeeze())


# --------------------------------------------------
# 3. Average Pooling
# --------------------------------------------------

average_model = build_average_pool_model()

average_output = average_model(input_tensor)

print("\nAverage Pooling Output")
print("-" * 65)
print(average_output.numpy().squeeze())


# --------------------------------------------------
# 4. Output shapes
# --------------------------------------------------

print("\nOutput Shapes")
print("-" * 65)
print(
    "Input          :",
    input_tensor.shape
)

print(
    "Max Pooling    :",
    max_output.shape
)

print(
    "Average Pooling:",
    average_output.shape
)


print("\nPooling experiment completed.")