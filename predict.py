import argparse

from utils import load_model, load_class_names, predict


def build_parser():
    parser = argparse.ArgumentParser(
        description="Predict flower name from an image using a trained Keras model."
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to input image (e.g., ./test_images/orchid.jpg)"
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to saved Keras model (e.g., my_model.h5 or my_model.keras)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Return top K most likely classes (default: 5)"
    )
    parser.add_argument(
        "--category_names",
        type=str,
        default="label_map.json",
        help="Path to JSON label->name mapping (default: label_map.json)"
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    model = load_model(args.model_path)
    class_names = load_class_names(args.category_names)

    probs, classes = predict(args.image_path, model, args.top_k)

    names = [class_names.get(c, f"Unknown({c})") for c in classes]

    print("\nPrediction Results")
    print("-" * 60)
    for i, (p, c, n) in enumerate(zip(probs, classes, names), start=1):
        print(f"{i:>2}. {n:<35} (class={c:>3})  prob={float(p):.6f}")
    print("-" * 60)


if __name__ == "__main__":
    main()

