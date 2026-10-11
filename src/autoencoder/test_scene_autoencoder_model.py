from src.autoencoder.scene_autoencoder_model import (
    build_scene_autoencoder
)


model = build_scene_autoencoder()

print("\nScene Autoencoder Architecture")
print("=" * 50)

model.summary()

print("\nInput shape:")
print(model.input_shape)

print("\nOutput shape:")
print(model.output_shape)

print("\nTotal parameters:")
print(model.count_params())