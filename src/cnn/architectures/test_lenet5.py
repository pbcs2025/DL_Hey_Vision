from src.cnn.architectures.lenet5 import build_lenet5


model = build_lenet5()

print("\nLeNet-5 Architecture")
print("=" * 60)

model.summary()

print("\nTotal Parameters")
print("=" * 60)
print(model.count_params())