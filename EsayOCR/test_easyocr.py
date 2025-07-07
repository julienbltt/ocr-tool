from qai_hub_models.utils.onnx_torch_wrapper import OnnxModelTorchWrapper
from qai_hub_models.models.easyocr.app import EasyOCRApp
from qai_hub_models.models.easyocr.model import LANG_LIST
from qai_hub_models.utils.display import display_or_save_image
from PIL import Image
from time import time
import onnxruntime as ort


def main():
    # Setting up ONNX Runtime logging
    ort.set_default_logger_severity(0)

    # Loading EasyOCR models
    print("Chargement des modèles...")
    try:
        app = EasyOCRApp(
            detector=OnnxModelTorchWrapper.OnCPU("EsayOCR\models\easyocr-easyocrdetector.onnx"),
            recognizer=OnnxModelTorchWrapper.OnCPU("EsayOCR\models\easyocr-easyocrrecognizer.onnx"),
            lang_list=LANG_LIST,
        )
        print("Model Loaded")

    except Exception as e:
        print(f"Error loading models: {e}")
        return

    print("Chargement de l'image...")
    try:
        image = Image.open("test_image.png")
        print(f"Image size: {image.size}, mode: {image.mode}")

        # Rize the image if it is too large
        if max(image.size) > 320:
                print("Image too large, resizing...")
                image.thumbnail((320, 320), Image.Resampling.LANCZOS)
                print(f"New size: {image.size}")

    except Exception as e:
        print(f"Error loading image: {e}")
        return

    print("Beginning prediction...")
    start_time = time()

    try:
        results = app.predict(image)
        end_time = time()
        print(f"Prediction completed in {end_time - start_time:.2f} seconds")

        print("Raw result:", results)

        image_rlt = results[0]
        text_rlt = results[1]
        for text in text_rlt:
            print(text[1], end=" ")
        print()

        display_or_save_image(image_rlt, "output")
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        return


if __name__ == "__main__":
    main()