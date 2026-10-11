from src.cnn.architectures.alexnet import build_alexnet


model = build_alexnet()

print("\nAlexNet-Style Architecture")
print("=" * 60)

model.summary()

print("\nTotal Parameters")
print("=" * 60)
print(model.count_params())