from .ocr import read_text
from ..response.text_response import generate_text_response


image_path = "data/test/ocr_test.png"

ocr_results = read_text(image_path)

print("\nOCR Results")
print("=" * 40)

for result in ocr_results:
    print(
        f"Text: {result['text']} "
        f"(confidence: {result['confidence']:.2f})"
    )

print("\nAssistant Response")
print("=" * 40)

response = generate_text_response(ocr_results)

print(response)