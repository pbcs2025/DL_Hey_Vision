from src.vision.live_camera import LiveCamera


def main():
    print("Testing live camera module...")

    with LiveCamera() as camera:

        frame = camera.read_frame()

        print(
            "✅ Live frame captured."
        )

        output_path = camera.save_current_frame()

        print(
            f"✅ Frame saved to: {output_path}"
        )

        print(
            f"Frame shape: {frame.shape}"
        )

    print("✅ Camera released.")
    print("Live camera module test complete.")


if __name__ == "__main__":
    main()