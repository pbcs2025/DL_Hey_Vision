def get_horizontal_position(x_center, image_width):
    """
    Determine whether an object is on the left, center,
    or right side of the image.
    """

    normalized_x = x_center / image_width

    if normalized_x < 0.33:
        return "left"

    elif normalized_x < 0.66:
        return "center"

    else:
        return "right"


def get_object_position(x1, y1, x2, y2, image_width, image_height):
    """
    Determine the spatial position of an object
    from its bounding box.
    """

    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2

    horizontal_position = get_horizontal_position(
        x_center,
        image_width
    )

    return {
        "horizontal": horizontal_position,
        "x_center": x_center,
        "y_center": y_center
    }


def process_yolo_detections(results):
    """
    Convert YOLO detections into objects with
    spatial positions.
    """

    detections = []

    for result in results:

        image_height, image_width = result.orig_shape

        if result.boxes is None:
            continue

        for box in result.boxes:

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            class_name = result.names[class_id]

            position = get_object_position(
                x1, y1, x2, y2,
                image_width,
                image_height
            )

            box_width = x2 - x1
            box_height = y2 - y1

            detections.append({
                "object": class_name,
                "confidence": confidence,
                "position": position["horizontal"],
                "x_center": position["x_center"],
                "y_center": position["y_center"],
                "width": box_width,
                "height": box_height
            })

    return detections