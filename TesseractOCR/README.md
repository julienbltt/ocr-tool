# TesseractOCR

Tesseract is Open Source OCR Engine.

## Installation

### **📋 State 1 : Tesseract**
- [ ] Go on: https://github.com/UB-Mannheim/tesseract/wiki
- [ ] Download this: `tesseract-ocr-w64-setup-v5.x.x.exe` (version 64-bit)
- [ ] Installer avec les options :
  - [ ] ✅ English language data
  - [ ] ✅ French language data (optionnel)
  - [ ] ✅ ScrollView
- [ ] Noter le chemin d'installation : `C:\Program Files\Tesseract-OCR\`

### **📋 State 2 : Environnement variables**
- [ ] Windows + R → `sysdm.cpl`
- [ ] Avancé → Variables d'environnement
- [ ] Path (système) → Modifier → Nouveau
- [ ] Ajouter : `C:\Program Files\Tesseract-OCR`
- [ ] OK sur toutes les fenêtres

### **📋 State 3 : Python package**
```bash
pip install pytesseract pillow opencv-python numpy
```

### **📋 State 4 : Installation test**
```bash
python test_tesseract_installation.py
```

## Test script

**Minimal script**
```python
import pytesseract
from PIL import Image

# Auto-configuration (si Tesseract dans PATH)
image = Image.open("votre_image.png")
text = pytesseract.image_to_string(image, lang='eng')
print(text)
```

**Robust script**
```python
import pytesseract
from PIL import Image

# Manual path configuration
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Option with OCR
image = Image.open("votre_image.png")
text = pytesseract.image_to_string(
    image, 
    lang='eng',  # ou 'fra' ou 'eng+fra'
    config='--oem 3 --psm 6'
)

print(f"Texte détecté : {text}")
```

## Scripts supplied

1. `test_tessact_installation.py`: Complet checkup.
2. `tesseract_complete_ocr.py`: Script with all functions.

## Using manual of TesseractOCR class

> Note: If you use virtual python environnement, your PATH environnement variable is already not disponible. Use the direct executable tesseract path.

---

## Utils links

Installer Windows for Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
Tesseract CLI man: https://github.com/tesseract-ocr/tesseract/blob/main/doc/tesseract.1.asc#languages-and-scripts
Main repository of Tesseract OCR: https://github.com/tesseract-ocr/tesseract

## Languages supported

**afr** (Afrikaans), **amh** (Amharic), **ara** (Arabic), **asm** (Assamese), **aze** (Azerbaijani), **aze_cyrl** (Azerbaijani - Cyrilic), **bel** (Belarusian), **ben** (Bengali), **bod** (Tibetan), **bos** (Bosnian), **bre** (Breton), **bul** (Bulgarian), **cat** (Catalan; Valencian), **ceb** (Cebuano), **ces** (Czech), **chi_sim** (Chinese simplified), **chi_tra** (Chinese traditional), **chr** (Cherokee), **cos** (Corsican), **cym** (Welsh), **dan** (Danish), **deu** (German), **deu_latf** (German Fraktur Latin), **div** (Dhivehi), **dzo** (Dzongkha), **ell** (Greek, Modern, 1453-), **eng** (English), **enm** (English, Middle, 1100-1500), **epo** (Esperanto), **equ** (Math / equation detection module), **est** (Estonian), **eus** (Basque), **fas** (Persian), **fao** (Faroese), **fil** (Filipino), **fin** (Finnish), **fra** (French), **frm** (French, Middle, ca.1400-1600), **fry** (West Frisian), **gla** (Scottish Gaelic), **gle** (Irish), **glg** (Galician), **grc** (Greek, Ancient, to 1453), **guj** (Gujarati), **hat** (Haitian; Haitian Creole), **heb** (Hebrew), **hin** (Hindi), **hrv** (Croatian), **hun** (Hungarian), **hye** (Armenian), **iku** (Inuktitut), **ind** (Indonesian), **isl** (Icelandic), **ita** (Italian), **ita_old** (Italian - Old), **jav** (Javanese), **jpn** (Japanese), **kan** (Kannada), **kat** (Georgian), **kat_old** (Georgian - Old), **kaz** (Kazakh), **khm** (Central Khmer), **kir** (Kirghiz; Kyrgyz), **kmr** (Kurdish Kurmanji), **kor** (Korean), **kor_vert** (Korean vertical), **lao** (Lao), **lat** (Latin), **lav** (Latvian), **lit** (Lithuanian), **ltz** (Luxembourgish), **mal** (Malayalam), **mar** (Marathi), **mkd** (Macedonian), **mlt** (Maltese), **mon** (Mongolian), **mri** (Maori), **msa** (Malay), **mya** (Burmese), **nep** (Nepali), **nld** (Dutch; Flemish), **nor** (Norwegian), **oci** (Occitan post 1500), **ori** (Oriya), **osd** (Orientation and script detection module), **pan** (Panjabi; Punjabi), **pol** (Polish), **por** (Portuguese), **pus** (Pushto; Pashto), **que** (Quechua), **ron** (Romanian; Moldavian; Moldovan), **rus** (Russian), **san** (Sanskrit), **sin** (Sinhala; Sinhalese), **slk** (Slovak), **slv** (Slovenian), **snd** (Sindhi), **spa** (Spanish; Castilian), **spa_old** (Spanish; Castilian - Old), **sqi** (Albanian), **srp** (Serbian), **srp_latn** (Serbian - Latin), **sun** (Sundanese), **swa** (Swahili), **swe** (Swedish), **syr** (Syriac), **tam** (Tamil), **tat** (Tatar), **tel** (Telugu), **tgk** (Tajik), **tha** (Thai), **tir** (Tigrinya), **ton** (Tonga), **tur** (Turkish), **uig** (Uighur; Uyghur), **ukr** (Ukrainian), **urd** (Urdu), **uzb** (Uzbek), **uzb_cyrl** (Uzbek - Cyrilic), **vie** (Vietnamese), **yid** (Yiddish), **yor** (Yoruba)

> For more informations: [doc/tesseract.1.asc#languages-and-scripts](https://github.com/tesseract-ocr/tesseract/blob/main/doc/tesseract.1.asc#languages-and-scripts)

