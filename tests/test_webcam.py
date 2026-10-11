import cv2


def main():
    print("Opening webcam...")

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("❌ Could not open webcam.")
        return

    print("✅ Webcam opened successfully.")
    print("Press Q to close the webcam window.")

    while True:
        success, frame = camera.read()

        if not success:
            print("❌ Could not read frame.")
            break

        cv2.imshow(
            "AI Vision Assistant - Webcam Test",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

    print("Webcam test finished.")


if __name__ == "__main__":
    main()