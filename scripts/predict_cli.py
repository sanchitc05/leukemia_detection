import argparse
from src.inference import load_trained_model, predict_image
from src.utils import print_system_info
import pprint

def main():
    parser = argparse.ArgumentParser(description="Predict leukemia from an input image.")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the image you want to predict."
    )

    args = parser.parse_args()

    print_system_info()

    print("\n🔍 Loading model...")
    model = load_trained_model()
    print("✅ Model loaded successfully.\n")

    print(f"📷 Predicting image: {args.image}")
    result = predict_image(args.image, model)

    print("\n📊 Prediction Result:")
    pprint.pprint(result)

    print(f"\n🧠 Final Prediction: {result['predicted_class']} (Confidence: {result['confidence']:.2f})")


if __name__ == "__main__":
    main()