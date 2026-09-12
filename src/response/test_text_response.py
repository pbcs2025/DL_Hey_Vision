from .text_response import generate_text_response


ocr_results = [
    {"text": "LIBRARY", "confidence": 1.00},
    {"text": "ROOM 204", "confidence": 0.99}
]

response = generate_text_response(ocr_results)

print(response)