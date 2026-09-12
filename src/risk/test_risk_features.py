from src.risk.risk_features import create_risk_features


features = create_risk_features(
    object_name="chair",
    position="center",
    width=300,
    height=400,
    image_width=1200,
    image_height=1600,
    confidence=0.91,
    movement="stationary",
    ahead=True
)


print("\nGenerated ANN Features")
print("=" * 40)

print(features)

print("\nFeature meanings")
print("-" * 40)

print(f"Object type : {features[0]:.4f}")
print(f"Position    : {features[1]:.4f}")
print(f"Size        : {features[2]:.4f}")
print(f"Confidence  : {features[3]:.4f}")
print(f"Movement    : {features[4]:.4f}")
print(f"Ahead       : {features[5]:.4f}")