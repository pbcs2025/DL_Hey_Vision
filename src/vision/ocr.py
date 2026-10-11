import cv2
import easyocr


# --------------------------------------------------
# Create the OCR reader
# --------------------------------------------------
reader = easyocr.Reader(
    ['en'],
    gpu=False
)


def preprocess_image(image_path):
    """
    Preprocess the image before OCR.
    """

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {image_path}"
        )

    # Convert to grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Upscale the image
    # This helps OCR detect small characters.
    scale = 2

    enlarged = cv2.resize(
        gray,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC
    )

    # Reduce small image noise
    denoised = cv2.GaussianBlur(
        enlarged,
        (3, 3),
        0
    )

    # Improve local contrast
    enhanced = cv2.equalizeHist(
        denoised
    )

    return enhanced


def read_text(image_path):
    """
    Detect and recognize text from an image.
    """

    processed_image = preprocess_image(
        image_path
    )

    results = reader.readtext(
        processed_image,
        detail=1,
        paragraph=False
    )

    texts = []

    for result in results:

        text = result[1]
        confidence = float(result[2])

        # Ignore extremely low-confidence results.
        if confidence < 0.30:
            continue

        texts.append({
            "text": text,
            "confidence": confidence
        })

    return texts