from src.autoencoder.autoencoder_model import build_autoencoder


model = build_autoencoder()

print("\nAutoencoder Architecture")
print("=" * 50)

model.summary()

print("\nInput shape:")
print(model.input_shape)

print("\nOutput shape:")
print(model.output_shape)

print("\nTotal parameters:")
print(model.count_params())