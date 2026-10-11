from src.cnn.padding_stride_experiment import (
    build_conv_model,
    calculate_output_size
)


print("\nAI Vision Assistant - CNN Padding & Stride Experiment")
print("=" * 65)


INPUT_SIZE = 8
FILTER_SIZE = 3


experiments = [
    ("A", "same", 1),
    ("B", "valid", 1),
    ("C", "same", 2),
    ("D", "valid", 2)
]


for name, padding, stride in experiments:

    model = build_conv_model(
        padding=padding,
        stride=stride
    )

    output_shape = model.output_shape

    calculated_size = calculate_output_size(
        input_size=INPUT_SIZE,
        filter_size=FILTER_SIZE,
        padding=padding,
        stride=stride
    )

    print(f"\nExperiment {name}")
    print("-" * 65)

    print(f"Input size : {INPUT_SIZE} x {INPUT_SIZE}")
    print(f"Filter     : {FILTER_SIZE} x {FILTER_SIZE}")
    print(f"Padding    : {padding}")
    print(f"Stride     : {stride}")

    print(
        f"Output shape from Keras : "
        f"{output_shape[1]} x {output_shape[2]} x {output_shape[3]}"
    )

    print(
        f"Calculated output size  : "
        f"{calculated_size} x {calculated_size}"
    )


print("\nPadding and stride experiment completed.")