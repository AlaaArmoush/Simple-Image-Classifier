import argparse
import sys
from utils import load_model, load_class_names, predict

def build_parser():
    parser = argparse.ArgumentParser(
        description="Predict flower species from an image using a trained model."
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to the flower image file"
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the saved Keras model file"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top predictions to display"
    )
    parser.add_argument(
        "--category_names",
        type=str,
        default="label_map.json",
        help="Path to the JSON file mapping labels to flower names"
    )
    return parser

def main():
    arg_parser = build_parser()
    args = arg_parser.parse_args()

    try:
        trained_model = load_model(args.model_path)
        label_map = load_class_names(args.category_names)
        
        probabilities, class_labels = predict(args.image_path, trained_model, args.top_k)
        
        flower_names = [label_map.get(str(label), f"ID:{label}") for label in class_labels]

        print(f"\nTop {args.top_k} Prediction Results")
        print("-" * 65)
        for rank, (prob, label, name) in enumerate(zip(probabilities, class_labels, flower_names), 1):
            print(f"{rank:>2}. {name:<35} (Label: {label:>3}) Confidence: {float(prob):.2%}")
        print("-" * 65)

    except FileNotFoundError as file_error:
        print(f"File Error: {file_error}")
        sys.exit(1)
    except ValueError as value_error:
        print(f"Data Error: {value_error}")
        sys.exit(1)
    except Exception as unexpected_error:
        print(f"Unexpected Error: {unexpected_error}")
        sys.exit(1)

if __name__ == "__main__":
    main()
