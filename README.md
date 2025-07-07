# ocr-tool
This tool combines real-time video capture with OCR using a AI model for implemente in the Commpanion project. 

##  Detailed comparison table

| Critère | Tesseract | PaddleOCR | TrOCR | Google Vision | Azure CV | AWS Textract |
|---------|-----------|-----------|-------|---------------|----------|--------------|
| **Installation** | ⭐⭐⭐⭐⭐ Très facile | ⭐⭐⭐⭐ Facile | ⭐⭐⭐ Moyenne | ⭐⭐ Complexe | ⭐⭐ Complexe | ⭐⭐ Complexe |
| **Compatibilité ARM** | ✅ Parfaite | ✅ Très bonne | ✅ Bonne | ✅ Cloud | ✅ Cloud | ✅ Cloud |
| **Précision générale** | ⭐⭐⭐⭐ Bonne | ⭐⭐⭐⭐⭐ Excellente | ⭐⭐⭐⭐ Très bonne | ⭐⭐⭐⭐⭐ Excellente | ⭐⭐⭐⭐⭐ Excellente | ⭐⭐⭐⭐⭐ Excellente |
| **Vitesse** | ⭐⭐⭐⭐ Rapide | ⭐⭐⭐ Moyenne | ⭐⭐ Lente | ⭐⭐⭐ Dépend réseau | ⭐⭐⭐ Dépend réseau | ⭐⭐⭐ Dépend réseau |
| **Hors ligne** | ✅ Oui | ✅ Oui | ✅ Oui | ❌ Non | ❌ Non | ❌ Non |
| **Coût** | 🆓 Gratuit | 🆓 Gratuit | 🆓 Gratuit | 💰 Payant | 💰 Payant | 💰 Payant |
| **Langues supportées** | 100+ | 80+ | Anglais | 50+ | 25+ | 15+ |
| **Texte manuscrit** | ⭐⭐ Limité | ⭐⭐⭐ Bon | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐⭐⭐ Très bon | ⭐⭐⭐⭐ Très bon | ⭐⭐⭐ Bon |
| **Documents complexes** | ⭐⭐⭐ Bon | ⭐⭐⭐⭐ Très bon | ⭐⭐ Limité | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐⭐⭐⭐ Excellent |

## Files architecture

- EasyOCR: easyocr implementation
- PaddleOCR: paddleocr implementation
- TesseractOCR: tesseractocr impletmentation


### Test status

- EsayOCR: Don't work with Qualcomm AI Hub framework.
- PaddleOCR: Don't works beacause incompadible with ARM64 architecture.
- TesseractOCR: Works!


