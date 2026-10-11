import colorsys
from collections import Counter
import cv2
import numpy as np


# Pre-defined consistent color mapping for common YOLO object classes (BGR format for OpenCV)
CLASS_COLORS = {
    "person": (235, 115, 40),       # Cyan-Blue
    "chair": (46, 204, 113),        # Emerald Green
    "table": (241, 196, 15),        # Bright Gold
    "door": (155, 89, 182),         # Amethyst Purple
    "car": (231, 76, 60),           # Crimson Red
    "bottle": (26, 188, 156),       # Teal
    "backpack": (230, 126, 34),     # Orange
    "cell phone": (52, 152, 219),   # Dodger Blue
    "phone": (52, 152, 219),        # Dodger Blue
    "laptop": (142, 68, 173),       # Deep Purple
    "book": (243, 156, 18),         # Amber
    "cup": (22, 160, 133),          # Dark Cyan
    "tv": (192, 57, 43),            # Rust Red
    "monitor": (192, 57, 43),       # Rust Red
    "couch": (39, 174, 96),         # Dark Green
    "sofa": (39, 174, 96),          # Dark Green
    "bed": (41, 128, 185),          # Ocean Blue
}

OBJECT_ICONS = {
    "person": "👤",
    "chair": "🪑",
    "table": "🟫",
    "door": "🚪",
    "car": "🚗",
    "bottle": "🍾",
    "backpack": "🎒",
    "cell phone": "📱",
    "phone": "📱",
    "laptop": "💻",
    "book": "📖",
    "cup": "☕",
    "tv": "📺",
    "monitor": "📺",
    "couch": "🛋️",
    "sofa": "🛋️",
    "bed": "🛏️",
    "mouse": "🖱️",
    "keyboard": "⌨️",
    "clock": "⏰",
}


def get_class_color(class_name: str) -> tuple:
    """
    Get a consistent, deterministic color (BGR) for an object class name.
    If the class is in CLASS_COLORS, returns the curated color.
    Otherwise, generates a deterministic bright color based on a hash of the class name.
    """
    clean_name = class_name.lower().strip()
    if clean_name in CLASS_COLORS:
        return CLASS_COLORS[clean_name]

    # Deterministic HSL -> BGR for unknown classes
    hash_val = sum(ord(char) * (idx + 1) for idx, char in enumerate(clean_name))
    hue = (hash_val * 137.5) % 360.0 / 360.0  # Golden ratio hue distribution
    saturation = 0.85
    lightness = 0.5

    rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
    bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
    return bgr


def draw_detections(image: np.ndarray, detections: list) -> np.ndarray:
    """
    Draw YOLO bounding boxes, object names, and confidence scores onto an image.

    :param image: Input OpenCV image (BGR format).
    :param detections: List of detection dictionaries.
    :return: Annotated OpenCV image (BGR format).
    """
    if image is None:
        return None

    annotated = image.copy()
    img_h, img_w = annotated.shape[:2]

    for det in detections:
        obj_name = det.get("object", "unknown")
        confidence = det.get("confidence", 0.0)

        # Get box coordinates
        if "bbox" in det and len(det["bbox"]) == 4:
            x1, y1, x2, y2 = det["bbox"]
        else:
            x_center = det.get("x_center", img_w / 2)
            y_center = det.get("y_center", img_h / 2)
            w = det.get("width", 0)
            h = det.get("height", 0)
            x1 = int(x_center - w / 2)
            y1 = int(y_center - h / 2)
            x2 = int(x_center + w / 2)
            y2 = int(y_center + h / 2)

        # Ensure box is within frame boundaries
        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(0, min(x2, img_w - 1))
        y2 = max(0, min(y2, img_h - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        color = get_class_color(obj_name)

        # Draw bounding rectangle
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness=2)

        # Format label: e.g., "PERSON 91%"
        label_text = f"{obj_name.upper()} {int(confidence * 100)}%"

        # Label dimensions
        font_scale = 0.55
        font_thickness = 1
        font_face = cv2.FONT_HERSHEY_SIMPLEX

        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, font_face, font_scale, font_thickness
        )

        # Draw label background banner above box (or inside if near top edge)
        banner_y1 = y1 - text_height - baseline - 6
        banner_y2 = y1

        if banner_y1 < 0:  # Shift inside box if box touches top border
            banner_y1 = y1
            banner_y2 = y1 + text_height + baseline + 6
            text_y = y1 + text_height + 3
        else:
            text_y = y1 - baseline - 3

        banner_x2 = min(x1 + text_width + 8, img_w - 1)

        # Filled background rectangle for text readability
        cv2.rectangle(
            annotated,
            (x1, max(0, banner_y1)),
            (banner_x2, banner_y2),
            color,
            thickness=-1
        )

        # Draw label text (white for high contrast)
        cv2.putText(
            annotated,
            label_text,
            (x1 + 4, text_y),
            font_face,
            font_scale,
            (255, 255, 255),
            font_thickness,
            lineType=cv2.LINE_AA
        )

    return annotated


def get_scene_summary_data(detections: list) -> list:
    """
    Return structured object counts and icons for UI rendering.
    """
    if not detections:
        return []

    counts = Counter(det.get("object", "unknown").lower() for det in detections)
    summary_items = []

    for obj_name, count in counts.most_common():
        icon = OBJECT_ICONS.get(obj_name, "📦")
        display_name = obj_name.capitalize()
        summary_items.append({
            "object": display_name,
            "count": count,
            "icon": icon,
            "color": get_class_color(obj_name)
        })

    return summary_items
