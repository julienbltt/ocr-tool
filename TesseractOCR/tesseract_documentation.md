# 📚 Documentation TesseractOCR - Guide Complet d'Utilisation

## 🎯 Vue d'Ensemble

La classe `TesseractOCR` est un wrapper Python avancé pour Tesseract OCR, spécialement optimisé pour Windows et Snapdragon X Plus. Elle offre des fonctionnalités complètes d'extraction de texte avec préprocessing automatique, analyse multilingue et optimisations de performance.

### ✨ Fonctionnalités Principales

- 🔧 **Configuration automatique** de Tesseract
- 🖼️ **Préprocessing intelligent** des images
- 🌍 **Support multilingue** (100+ langues)
- 📊 **Extraction avec coordonnées** des zones de texte
- 📈 **Analyse de performance** et optimisation
- 🎨 **Visualisation des résultats**
- ⚡ **Optimisé pour Snapdragon X Plus**

---

## 🚀 Installation et Import

```python
from tesseract_complete_ocr import TesseractOCR
import pytesseract
from PIL import Image
```

---

## 📖 Référence des Méthodes

### 🏗️ **`__init__(tesseract_path=None)`**

**Constructeur de la classe - Configuration automatique de Tesseract**

#### **Paramètres**
- `tesseract_path` (str, optionnel) : Chemin personnalisé vers tesseract.exe

#### **Comportement**
- Détecte automatiquement Tesseract dans les emplacements standards
- Vérifie les langues disponibles
- Configure l'environnement optimal

#### **Exemples d'utilisation**

```python
# Configuration automatique (recommandé)
ocr = TesseractOCR()

# Configuration manuelle
ocr = TesseractOCR(r'C:\Program Files\Tesseract-OCR\tesseract.exe')

# Vérification des langues disponibles
print(f"Langues supportées : {ocr.available_languages}")
```

#### **⚠️ Gestion d'Erreurs**
```python
try:
    ocr = TesseractOCR()
    print("✅ Tesseract configuré avec succès")
except Exception as e:
    print(f"❌ Erreur configuration : {e}")
```

---

### 🎨 **`preprocess_image(image, enhancement_level='medium')`**

**Préprocessing intelligent pour optimiser la reconnaissance OCR**

#### **Paramètres**
- `image` : Image PIL, chemin vers fichier, ou array numpy
- `enhancement_level` : `'light'`, `'medium'`, `'strong'`

#### **Traitements Appliqués**

| Niveau | Redimensionnement | Contraste | Netteté | Réduction Bruit | Seuillage |
|--------|-------------------|-----------|---------|-----------------|-----------|
| `light` | ✅ | ❌ | ❌ | ❌ | ❌ |
| `medium` | ✅ | ✅ (1.5x) | ✅ (1.2x) | ❌ | ❌ |
| `strong` | ✅ | ✅ (2.0x) | ✅ (1.5x) | ✅ | ✅ |

#### **Exemples d'utilisation**

```python
from PIL import Image

# Préprocessing basique
image = Image.open("document.png")
processed = ocr.preprocess_image(image, enhancement_level='light')

# Préprocessing avancé pour texte difficile
processed = ocr.preprocess_image(image, enhancement_level='strong')

# Avec chemin de fichier
processed = ocr.preprocess_image("scan_flou.jpg", enhancement_level='strong')

# Comparaison des niveaux
levels = ['light', 'medium', 'strong']
for level in levels:
    processed = ocr.preprocess_image(image, enhancement_level=level)
    processed.save(f"output_{level}.png")
    print(f"Image {level} sauvegardée")
```

#### **💡 Conseils d'Utilisation**
- **`light`** : Images de bonne qualité, texte net
- **`medium`** : Images standards, légers défauts
- **`strong`** : Images floues, basse résolution, texte dégradé

---

### 📝 **`extract_text_simple(image, lang='eng', config=None)`**

**Extraction de texte simple et rapide**

#### **Paramètres**
- `image` : Image ou chemin vers fichier
- `lang` : Code langue ('eng', 'fra', 'eng+fra', etc.)
- `config` : Configuration Tesseract personnalisée

#### **Valeur de Retour**
- `str` : Texte extrait (nettoyé, sans espaces inutiles)

#### **Exemples d'utilisation**

```python
# Extraction basique en anglais
text = ocr.extract_text_simple("document.png")
print(f"Texte : {text}")

# Extraction en français
text = ocr.extract_text_simple("document_fr.png", lang='fra')

# Extraction multilingue
text = ocr.extract_text_simple("document_mixte.png", lang='eng+fra')

# Avec configuration personnalisée
text = ocr.extract_text_simple(
    "ligne_unique.png", 
    lang='eng',
    config='--oem 3 --psm 7'  # PSM 7 = ligne unique
)

# Traitement par lot
images = ["doc1.png", "doc2.png", "doc3.png"]
results = {}

for img_path in images:
    text = ocr.extract_text_simple(img_path, lang='eng')
    results[img_path] = text
    print(f"{img_path}: {len(text)} caractères extraits")
```

#### **🎯 Configurations PSM Utiles**
```python
# Configurations courantes
configs = {
    'auto': '--oem 3 --psm 3',           # Détection automatique
    'bloc': '--oem 3 --psm 6',           # Bloc de texte uniforme (défaut)
    'ligne': '--oem 3 --psm 7',          # Ligne unique
    'mot': '--oem 3 --psm 8',            # Mot unique
    'sparse': '--oem 3 --psm 11'         # Texte sparse
}

for name, config in configs.items():
    text = ocr.extract_text_simple("test.png", config=config)
    print(f"{name}: '{text[:50]}...'")
```

---

### 📊 **`extract_text_with_boxes(image, lang='eng', confidence_threshold=30)`**

**Extraction avancée avec coordonnées et confiance**

#### **Paramètres**
- `image` : Image ou chemin vers fichier
- `lang` : Code langue
- `confidence_threshold` : Seuil de confiance minimum (0-100)

#### **Valeur de Retour**
```python
[
    {
        'text': 'Mot détecté',
        'confidence': 85,           # Pourcentage de confiance
        'x': 120,                   # Position X
        'y': 50,                    # Position Y  
        'width': 80,                # Largeur
        'height': 25,               # Hauteur
        'level': 5                  # Niveau hiérarchique
    },
    # ... autres détections
]
```

#### **Exemples d'utilisation**

```python
# Extraction avec seuil de confiance élevé
boxes = ocr.extract_text_with_boxes(
    "document.png", 
    lang='eng', 
    confidence_threshold=70
)

print(f"Détections trouvées : {len(boxes)}")

# Analyser les résultats
for i, box in enumerate(boxes):
    print(f"{i+1}. '{box['text']}' - {box['confidence']}% "
          f"à ({box['x']}, {box['y']})")

# Filtrer par niveau de confiance
high_confidence = [b for b in boxes if b['confidence'] >= 80]
medium_confidence = [b for b in boxes if 50 <= b['confidence'] < 80]
low_confidence = [b for b in boxes if b['confidence'] < 50]

print(f"Haute confiance : {len(high_confidence)}")
print(f"Confiance moyenne : {len(medium_confidence)}")
print(f"Faible confiance : {len(low_confidence)}")

# Extraire zones spécifiques
def extract_by_region(boxes, x_min, y_min, x_max, y_max):
    """Extraire le texte dans une région spécifique"""
    region_boxes = []
    for box in boxes:
        if (x_min <= box['x'] <= x_max and 
            y_min <= box['y'] <= y_max):
            region_boxes.append(box)
    return region_boxes

# Exemple : extraire le texte en haut à droite
top_right_boxes = extract_by_region(boxes, 400, 0, 800, 200)
top_right_text = ' '.join([b['text'] for b in top_right_boxes])
print(f"Texte en haut à droite : {top_right_text}")
```

#### **🔍 Analyse des Niveaux Hiérarchiques**
```python
# Grouper par niveau hiérarchique
from collections import defaultdict

level_groups = defaultdict(list)
for box in boxes:
    level_groups[box['level']].append(box)

for level, group in level_groups.items():
    texts = [b['text'] for b in group]
    print(f"Niveau {level}: {len(group)} éléments - {' '.join(texts)}")
```

---

### 🎨 **`draw_results(image_path, results, output_path, show_confidence=True, min_confidence=50)`**

**Visualisation des résultats OCR sur l'image**

#### **Paramètres**
- `image_path` : Chemin vers l'image originale
- `results` : Résultats de `extract_text_with_boxes()`
- `output_path` : Chemin de sauvegarde de l'image annotée
- `show_confidence` : Afficher le pourcentage de confiance
- `min_confidence` : Confiance minimale pour l'affichage

#### **Couleurs Automatiques**
- 🟢 **Vert** : Confiance ≥ 80%
- 🟡 **Jaune** : Confiance ≥ 60%
- 🟠 **Orange** : Confiance < 60%

#### **Exemples d'utilisation**

```python
# Visualisation standard
boxes = ocr.extract_text_with_boxes("document.png")
ocr.draw_results("document.png", boxes, "resultat_annote.png")

# Visualisation sans pourcentages de confiance
ocr.draw_results(
    "document.png", 
    boxes, 
    "resultat_propre.png",
    show_confidence=False
)

# Afficher seulement les détections très fiables
ocr.draw_results(
    "document.png", 
    boxes, 
    "resultat_fiable.png",
    min_confidence=80
)

# Génération de plusieurs versions
confidence_levels = [30, 50, 70, 90]
for confidence in confidence_levels:
    output = f"resultat_conf_{confidence}.png"
    ocr.draw_results(
        "document.png", 
        boxes, 
        output, 
        min_confidence=confidence
    )
    print(f"Version {confidence}% générée : {output}")
```

#### **🎯 Analyse Visuelle Custom**
```python
import cv2
import numpy as np

def create_heatmap(image_path, boxes, output_path):
    """Créer une carte de chaleur des zones de texte"""
    image = cv2.imread(image_path)
    heatmap = np.zeros(image.shape[:2], dtype=np.float32)
    
    for box in boxes:
        x, y, w, h = box['x'], box['y'], box['width'], box['height']
        confidence = box['confidence'] / 100.0
        
        # Ajouter à la heatmap
        heatmap[y:y+h, x:x+w] += confidence
    
    # Normaliser et appliquer colormap
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_HOT)
    
    # Fusionner avec image originale
    result = cv2.addWeighted(image, 0.7, heatmap, 0.3, 0)
    cv2.imwrite(output_path, result)

# Utilisation
boxes = ocr.extract_text_with_boxes("document.png")
create_heatmap("document.png", boxes, "heatmap.png")
```

---

### 📈 **`analyze_image(image_path, languages=['eng'], save_results=True)`**

**Analyse complète multilingue avec rapport détaillé**

#### **Paramètres**
- `image_path` : Chemin vers l'image
- `languages` : Liste des langues à tester
- `save_results` : Sauvegarder les images annotées

#### **Valeur de Retour**
```python
{
    'image_info': {
        'path': 'document.png',
        'size': (800, 600),
        'mode': 'RGB',
        'file_size_kb': 245.7
    },
    'results': {
        'eng': {
            'text': 'Texte extrait...',
            'boxes': [...],
            'stats': {
                'total_characters': 156,
                'total_words': 28,
                'detected_elements': 12,
                'average_confidence': 78.5
            }
        },
        'fra': { ... }
    }
}
```

#### **Exemples d'utilisation**

```python
# Analyse simple en anglais
rapport = ocr.analyze_image("document.png")

# Analyse multilingue
rapport = ocr.analyze_image(
    "document_international.png", 
    languages=['eng', 'fra', 'deu', 'spa']
)

# Analyse sans sauvegarde d'images
rapport = ocr.analyze_image(
    "document.png", 
    languages=['eng'],
    save_results=False
)

# Exploitation du rapport
for lang, result in rapport['results'].items():
    stats = result['stats']
    print(f"\n📊 Statistiques {lang.upper()}:")
    print(f"   Mots détectés: {stats['total_words']}")
    print(f"   Caractères: {stats['total_characters']}")
    print(f"   Éléments: {stats['detected_elements']}")
    print(f"   Confiance moyenne: {stats['average_confidence']:.1f}%")
    
    # Aperçu du texte
    text_preview = result['text'][:200]
    print(f"   Aperçu: {text_preview}...")

# Comparaison des langues
def compare_languages(rapport):
    """Comparer les performances par langue"""
    best_lang = None
    best_score = 0
    
    for lang, result in rapport['results'].items():
        # Score basé sur confiance moyenne et nombre de mots
        stats = result['stats']
        score = (stats['average_confidence'] * 
                stats['total_words'] / 100.0)
        
        print(f"{lang}: Score = {score:.2f}")
        
        if score > best_score:
            best_score = score
            best_lang = lang
    
    print(f"\n🏆 Meilleure langue: {best_lang} (score: {best_score:.2f})")
    return best_lang

best_language = compare_languages(rapport)
```

#### **📊 Génération de Rapports**
```python
def generate_html_report(rapport, output_file="rapport_ocr.html"):
    """Générer un rapport HTML"""
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Rapport OCR - {rapport['image_info']['path']}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .stats {{ background: #f0f0f0; padding: 10px; margin: 10px 0; }}
            .text-preview {{ background: #fff; border: 1px solid #ccc; 
                          padding: 10px; max-height: 200px; overflow-y: scroll; }}
        </style>
    </head>
    <body>
        <h1>Rapport d'Analyse OCR</h1>
        <h2>Informations Image</h2>
        <div class="stats">
            <p><strong>Fichier:</strong> {rapport['image_info']['path']}</p>
            <p><strong>Dimensions:</strong> {rapport['image_info']['size']}</p>
            <p><strong>Taille:</strong> {rapport['image_info']['file_size_kb']:.1f} KB</p>
        </div>
    """
    
    for lang, result in rapport['results'].items():
        stats = result['stats']
        html += f"""
        <h2>Langue: {lang.upper()}</h2>
        <div class="stats">
            <p><strong>Mots:</strong> {stats['total_words']}</p>
            <p><strong>Caractères:</strong> {stats['total_characters']}</p>
            <p><strong>Confiance moyenne:</strong> {stats['average_confidence']:.1f}%</p>
        </div>
        <div class="text-preview">
            <h3>Texte extrait:</h3>
            <p>{result['text']}</p>
        </div>
        """
    
    html += "</body></html>"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"📄 Rapport HTML généré: {output_file}")

# Utilisation
rapport = ocr.analyze_image("document.png", languages=['eng', 'fra'])
generate_html_report(rapport)
```

---

### ⚙️ **`test_configurations(image_path)`**

**Optimisation automatique des paramètres OCR**

#### **Paramètres**
- `image_path` : Chemin vers l'image à tester

#### **Configurations Testées**
- PSM 6 : Bloc de texte uniforme (défaut)
- PSM 7 : Ligne de texte unique  
- PSM 8 : Mot unique
- PSM 11 : Caractère unique sparse
- PSM 12 : Texte sparse avec OSD
- PSM 13 : Ligne brute

#### **Valeur de Retour**
```python
{
    6: {
        'text': 'Texte détecté...',
        'word_count': 25,
        'char_count': 145,
        'processing_time': 1.23,
        'success': True
    },
    7: { ... },
    # etc.
}
```

#### **Exemples d'utilisation**

```python
# Test automatique de toutes les configurations
results = ocr.test_configurations("document_complexe.png")

# Analyser les résultats
best_config = None
best_word_count = 0

for psm, result in results.items():
    if result.get('success', False):
        word_count = result['word_count']
        time_taken = result['processing_time']
        
        print(f"PSM {psm}: {word_count} mots en {time_taken:.2f}s")
        
        if word_count > best_word_count:
            best_word_count = word_count
            best_config = psm

print(f"\n🏆 Meilleure configuration: PSM {best_config}")

# Utiliser la meilleure configuration
best_text = ocr.extract_text_simple(
    "document_complexe.png",
    config=f'--oem 3 --psm {best_config}'
)

print(f"Texte avec meilleure config: {best_text}")
```

#### **🚀 Optimisation Automatique**
```python
def auto_optimize_ocr(ocr, image_path):
    """Optimisation automatique pour une image"""
    
    print(f"🔍 Optimisation automatique pour: {image_path}")
    
    # Test des configurations
    results = ocr.test_configurations(image_path)
    
    # Trouver la meilleure
    successful_configs = {k: v for k, v in results.items() 
                         if v.get('success', False)}
    
    if not successful_configs:
        print("❌ Aucune configuration réussie")
        return None
    
    # Critères de sélection (priorité aux mots détectés)
    best_psm = max(successful_configs.keys(), 
                   key=lambda x: successful_configs[x]['word_count'])
    
    best_result = successful_configs[best_psm]
    
    print(f"✅ Meilleure config: PSM {best_psm}")
    print(f"   📊 {best_result['word_count']} mots détectés")
    print(f"   ⏱️ {best_result['processing_time']:.2f}s")
    
    # Configuration optimale
    optimal_config = f'--oem 3 --psm {best_psm}'
    
    return {
        'psm': best_psm,
        'config': optimal_config,
        'expected_words': best_result['word_count'],
        'expected_time': best_result['processing_time']
    }

# Utilisation
optimization = auto_optimize_ocr(ocr, "document_difficile.png")

if optimization:
    # Appliquer la configuration optimale
    final_text = ocr.extract_text_simple(
        "document_difficile.png",
        config=optimization['config']
    )
    print(f"📝 Texte final optimisé: {final_text}")
```

---

## 🎯 Exemples d'Utilisation Avancée

### 📋 **Traitement par Lot**
```python
import os
from pathlib import Path

def process_folder(folder_path, output_folder="ocr_results"):
    """Traiter tous les images d'un dossier"""
    
    ocr = TesseractOCR()
    
    # Créer dossier de sortie
    Path(output_folder).mkdir(exist_ok=True)
    
    # Extensions supportées
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'}
    
    results = {}
    
    for file_path in Path(folder_path).iterdir():
        if file_path.suffix.lower() in image_extensions:
            print(f"\n📄 Traitement: {file_path.name}")
            
            try:
                # Analyse complète
                rapport = ocr.analyze_image(
                    str(file_path), 
                    languages=['eng'],
                    save_results=True
                )
                
                # Sauver le texte
                text_file = Path(output_folder) / f"{file_path.stem}.txt"
                text_content = rapport['results']['eng']['text']
                
                with open(text_file, 'w', encoding='utf-8') as f:
                    f.write(text_content)
                
                results[file_path.name] = {
                    'text_file': str(text_file),
                    'word_count': rapport['results']['eng']['stats']['total_words'],
                    'confidence': rapport['results']['eng']['stats']['average_confidence']
                }
                
                print(f"   ✅ {len(text_content)} caractères extraits")
                
            except Exception as e:
                print(f"   ❌ Erreur: {e}")
                results[file_path.name] = {'error': str(e)}
    
    # Rapport final
    print(f"\n📊 RAPPORT FINAL")
    print(f"="*40)
    
    total_files = len(results)
    successful = len([r for r in results.values() if 'error' not in r])
    
    print(f"Fichiers traités: {total_files}")
    print(f"Succès: {successful}")
    print(f"Échecs: {total_files - successful}")
    
    return results

# Utilisation
results = process_folder("mes_documents", "textes_extraits")
```

### 🔍 **OCR en Temps Réel**
```python
import cv2
import numpy as np
from threading import Thread
import time

class RealTimeOCR:
    """OCR en temps réel sur flux vidéo"""
    
    def __init__(self):
        self.ocr = TesseractOCR()
        self.current_text = ""
        self.last_update = 0
        self.update_interval = 2.0  # Secondes entre updates OCR
        
    def process_frame(self, frame):
        """Traiter une frame vidéo"""
        current_time = time.time()
        
        # Limiter la fréquence OCR
        if current_time - self.last_update < self.update_interval:
            return self.current_text
        
        try:
            # Convertir frame en image PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            from PIL import Image
            pil_image = Image.fromarray(frame_rgb)
            
            # OCR rapide
            text = self.ocr.extract_text_simple(
                pil_image, 
                config='--oem 3 --psm 8'  # Mode rapide
            )
            
            if text.strip():
                self.current_text = text.strip()
                self.last_update = current_time
                
        except Exception as e:
            print(f"Erreur OCR temps réel: {e}")
        
        return self.current_text
    
    def run_camera(self):
        """Démarrer OCR sur caméra"""
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Traiter OCR
            detected_text = self.process_frame(frame)
            
            # Afficher texte sur frame
            if detected_text:
                cv2.putText(frame, detected_text[:50], (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Afficher frame
            cv2.imshow('OCR Temps Réel', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

# Utilisation
# realtime_ocr = RealTimeOCR()
# realtime_ocr.run_camera()
```

---

## ⚡ Conseils de Performance

### 🚀 **Optimisations Recommandées**
```python
# 1. Redimensionner les grandes images
def optimize_image_size(image_path, max_size=1024):
    """Optimiser la taille d'image pour OCR"""
    image = Image.open(image_path)
    
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    
    return image

# 2. Configuration système pour Snapdragon X Plus
import os
os.environ['OMP_NUM_THREADS'] = '4'  # Adapter selon vos cœurs
os.environ['MKL_NUM_THREADS'] = '4'

# 3. Réutiliser l'instance OCR
# ✅ Correct
ocr = TesseractOCR()
for image in image_list:
    text = ocr.extract_text_simple(image)

# ❌ Éviter
for image in image_list:
    ocr = TesseractOCR()  # Éviter la re-initialisation
    text = ocr.extract_text_simple(image)
```

### 📊 **Benchmarking**
```python
import time
from pathlib import Path

def benchmark_ocr(image_paths, configurations):
    """Benchmarker différentes configurations"""
    
    ocr = TesseractOCR()
    results = {}
    
    for config_name, config in configurations.items():
        print(f"\n🧪 Test configuration: {config_name}")
        
        total_time = 0
        total_words = 0
        
        for image_path in image_paths:
            start_time = time.time()
            
            text = ocr.extract_text_simple(image_path, config=config)
            
            end_time = time.time()
            
            processing_time = end_time - start_time
            word_count = len(text.split())
            
            total_time += processing_time
            total_words += word_count
            
            print(f"  {Path(image_path).name}: {word_count} mots, {processing_time:.2f}s")
        
        avg_time = total_time / len(image_paths)
        avg_words = total_words / len(image_paths)
        
        results[config_name] = {
            'avg_time': avg_time,
            'avg_words': avg_words,
            'total_time': total_time,
            'total_words': total_words
        }
        
        print(f"  📊 Moyenne: {avg_words:.1f} mots/image, {avg_time:.2f}s/image")
    
    return results

# Configurations à tester
test_configs = {
    'defaut': '--oem 3 --psm 6',
    'rapide': '--oem 3 --psm 8',
    'precis': '--oem 3 --psm 3',
    'sparse': '--oem 3 --psm 11'
}

# images_test = ["doc1.png", "doc2.png", "doc3.png"]
# benchmark_results = benchmark_ocr(images_test, test_configs)
```

---

## 🛠️ Dépannage

### ❌ **Erreurs Courantes**

```python
# Erreur: "Tesseract not found"
try:
    ocr = TesseractOCR()
except Exception:
    # Solution: chemin manuel
    ocr = TesseractOCR(r'C:\Program Files\Tesseract-OCR\tesseract.exe')

# Erreur: "No text detected"
# Solution: préprocessing plus agressif
processed = ocr.preprocess_image(image, enhancement_level='strong')
text = ocr.extract_text_simple(processed, config='--oem 3 --psm 11')

# Erreur: "Language not supported"
print("Langues disponibles:", ocr.available_languages)
# Installer la langue manquante ou utiliser 'eng'
```

### 🔧 **Tests de Diagnostic**
```python
def diagnostic_complet(image_path):
    """Diagnostic complet d'une image problématique"""
    
    print(f"🔍 DIAGNOSTIC: {image_path}")
    print("="*50)
    
    try:
        ocr = TesseractOCR()
        
        # 1. Informations image
        from PIL import Image
        img = Image.open(image_path)
        print(f"📐 Taille: {img.size}")
        print(f"🎨 Mode: {img.mode}")
        print(f"📊 Taille fichier: {os.path.getsize(image_path)/1024:.1f} KB")
        
        # 2. Test configurations
        results = ocr.test_configurations(image_path)
        
        # 3. Test préprocessing
        levels = ['light', 'medium', 'strong']
        for level in levels:
            processed = ocr.preprocess_image(img, enhancement_level=level)
            text = ocr.extract_text_simple(processed)
            print(f"Préprocessing {level}: {len(text)} caractères")
        
        # 4. Test multilingue
        languages = ['eng', 'fra'] if 'fra' in ocr.available_languages else ['eng']
        for lang in languages:
            text = ocr.extract_text_simple(image_path, lang=lang)
            print(f"Langue {lang}: {len(text)} caractères")
        
    except Exception as e:
        print(f"❌ Erreur diagnostic: {e}")

# diagnostic_complet("image_problematique.png")
```

---

## 📚 Ressources Supplémentaires

- **Documentation Tesseract** : https://tesseract-ocr.github.io/tessdoc/
- **Langues disponibles** : https://github.com/tesseract-ocr/tessdata
- **Configurations PSM** : https://tesseract-ocr.github.io/tessdoc/ImproveQuality.html
- **Optimisation images** : https://tesseract-ocr.github.io/tessdoc/ImproveQuality.html

---

**🎉 Avec cette documentation, vous maîtrisez maintenant tous les aspects de la classe TesseractOCR !**