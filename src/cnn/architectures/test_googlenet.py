from src.cnn.architectures.googlenet import build_googlenet


model = build_googlenet()

print("\nGoogLeNet / Inception-Style Architecture")
print("=" * 60)

model.summary()

print("\nTotal Parameters:", model.count_params())