import easyocr


# Create the OCR reader
reader = easyocr.Reader(
    ['en'],
    gpu=False
)


def read_text(image_path):
    """
    Detect and recognize text from an image.
    """

    results = reader.readtext(image_path)

    texts = []

    for result in results:
        text = result[1]
        confidence = result[2]

        texts.append({
            "text": text,
            "confidence": confidence
        })

    return texts