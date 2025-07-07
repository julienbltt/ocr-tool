# EasyOCR

## Notes

Install dependancies: `pip install "qai-hub-models[easyocr]"`
Test demo script: `python -m qai_hub_models.models.easyocr.demo` : works
Test demo script with a specific image : `python -m qai_hub_models.models.easyocr.demo --image test_image.png`: don't works
```shell
Error => 
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "C:\Users\<username>\Documents\Commpanion\ocr-tool\.venv\Lib\site-packages\qai_hub_models\models\easyocr\demo.py", line 49, in <module>
    main()
  File "C:\Users\<username>\Documents\Commpanion\ocr-tool\.venv\Lib\site-packages\qai_hub_models\models\easyocr\demo.py", line 42, in main
    results = app.predict_text_from_image(image)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\<username>\Documents\Commpanion\ocr-tool\.venv\Lib\site-packages\qai_hub_models\models\easyocr\app.py", line 471, in predict_text_from_image
    draw_box_from_xyxy(
  File "C:\Users\<username>\Documents\Commpanion\ocr-tool\.venv\Lib\site-packages\qai_hub_models\utils\draw.py", line 202, in draw_box_from_xyxy
    cv2.rectangle(frame, top_left, bottom_right, color, size)
cv2.error: OpenCV(4.10.0) :-1: error: (-5:Bad argument) in function 'rectangle'
> Overload resolution failed:
>  - Can't parse 'pt1'. Sequence item with index 0 has a wrong type
>  - Can't parse 'pt1'. Sequence item with index 0 has a wrong type
>  - Can't parse 'rec'. Expected sequence length 4, got 2
>  - Can't parse 'rec'. Expected sequence length 4, got 2
```

Fix this problem (maybe):
```python
# Imports
import qai_hub_models.utils.draw as draw_module
import cv2
# New function
def fixed_draw_box(frame, xyxy, color=(255, 0, 0), size=5):
    try:
        # Forcer conversion en entiers
        x1, y1, x2, y2 = [int(float(coord)) for coord in xyxy[:4]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, size)
    except Exception as e:
        print(f"Coordonnées ignorées: {xyxy} - {e}")

# Apply patch to draw_box_from_xyxy
draw_module.draw_box_from_xyxy = fixed_draw_boxpython
```

The inference with MY script using EasyOCRApp class from Qualcomm AI Hub framework don't work.

