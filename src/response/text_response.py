def generate_text_response(ocr_results):
    """
    Convert OCR results into a natural-language response.
    """

    if not ocr_results:
        return "I cannot read any text from the image."

    texts = []

    for result in ocr_results:
        text = result["text"].strip()

        if text:
            texts.append(text)

    if not texts:
        return "I cannot read any text from the image."

    text = ", ".join(texts)

    return f"The sign says {text}."