from src.cnn.architectures.resnet import build_resnet


model = build_resnet()

print("\nResNet-Style Architecture")
print("=" * 60)

model.summary()

print("\nTotal Parameters:", model.count_params())