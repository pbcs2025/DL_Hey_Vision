from .obstacle_reasoning import find_obstacles_ahead


detections = [
    {
        "object": "person",
        "position": "left"
    },
    {
        "object": "chair",
        "position": "center"
    },
    {
        "object": "table",
        "position": "right"
    },
    {
        "object": "bottle",
        "position": "center"
    }
]


obstacles = find_obstacles_ahead(detections)


print("\nObstacle Reasoning Test")
print("=" * 50)

print("\nDetected obstacles:")

for obstacle in obstacles:
    print(
        f"Object: {obstacle['object']} | "
        f"Position: {obstacle['position']}"
    )