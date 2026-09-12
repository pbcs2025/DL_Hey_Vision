from .obstacle_response import generate_obstacle_response


tests = [

    [],

    [
        {
            "object": "chair",
            "position": "center"
        }
    ],

    [
        {
            "object": "chair",
            "position": "center"
        },
        {
            "object": "table",
            "position": "center"
        }
    ],

    [
        {
            "object": "chair",
            "position": "center"
        },
        {
            "object": "chair",
            "position": "center"
        }
    ]
]


print("\nObstacle Response Test")
print("=" * 50)


for obstacles in tests:

    response = generate_obstacle_response(
        obstacles
    )

    print("\nObstacles:", obstacles)
    print("Response:", response)