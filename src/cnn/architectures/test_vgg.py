from src.cnn.architectures.vgg import build_vgg


model = build_vgg()

print("\nVGG-Style Architecture")
print("=" * 60)

model.summary()

print("\nTotal Parameters")
print("=" * 60)
print(model.count_params())