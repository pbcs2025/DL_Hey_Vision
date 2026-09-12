OBJECT_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "backpack",
    "bottle",
    "chair",
    "table",
    "traffic_light",
    "stop_sign",
    "door"
]


def encode_object(object_name):
    """
    Convert an object class name into its
    numerical YOLO class index.
    """

    if object_name not in OBJECT_CLASSES:
        return 0

    return OBJECT_CLASSES.index(object_name)


def encode_position(position):
    position_map = {
        "left": 0,
        "center": 1,
        "right": 2
    }

    return position_map.get(position, 1)


def encode_movement(movement):
    movement_map = {
        "stationary": 0,
        "approaching": 1,
        "moving away": 2,
        "crossing": 3
    }

    return movement_map.get(movement, 0)


def calculate_normalized_size(
    width,
    height,
    image_width,
    image_height
):
    box_area = width * height
    image_area = image_width * image_height

    if image_area == 0:
        return 0.0

    return box_area / image_area


def create_risk_features(
    object_name,
    position,
    width,
    height,
    image_width,
    image_height,
    confidence,
    movement="stationary",
    ahead=False
):
    """
    Create the six numerical features
    used by the risk ANN.

    Features:
        1. Object type
        2. Position
        3. Bounding-box size
        4. Detection confidence
        5. Movement
        6. Whether the object is ahead
    """

    object_type = encode_object(object_name)

    normalized_object_type = object_type / 12.0
    normalized_position = encode_position(position) / 2.0

    normalized_size = calculate_normalized_size(
        width,
        height,
        image_width,
        image_height
    )

    normalized_movement = encode_movement(movement) / 3.0

    ahead_value = 1 if ahead else 0

    return [
        normalized_object_type,
        normalized_position,
        normalized_size,
        confidence,
        normalized_movement,
        ahead_value
    ]