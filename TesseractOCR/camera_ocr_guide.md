# 📷 Guide d'Utilisation Camera OCR

## 🎯 Vue d'Ensemble

Deux programmes de test caméra OCR sont disponibles :

1. **`camera_ocr_test.py`** - Version console avec OpenCV
2. **`camera_ocr_gui.py`** - Version interface graphique avec Tkinter

Les deux utilisent la classe **TesseractOCR** pour l'extraction de texte en temps réel.

---

## 📋 Prérequis

### **Installation Required**
```bash
# Packages Python
pip install opencv-python pillow numpy tkinter

# Tesseract OCR (voir guide d'installation précédent)
# + fichier tesseract_complete_ocr.py dans le même répertoire
```

### **Fichiers Nécessaires**
```
votre_dossier/
├── tesseract_complete_ocr.py      # Classe TesseractOCR
├── camera_ocr_test.py             # Version console
└── camera_ocr_gui.py              # Version GUI
```

---

## 🖥️ Programme 1: Version Console (`camera_ocr_test.py`)

### **🚀 Démarrage**
```bash
python camera_ocr_test.py
```

### **🎮 Contrôles Clavier**

| Touche | Action |
|--------|--------|
| **ESPACE** | 📸 Prendre une photo et extraire le texte |
| **P** | 🎛️ Activer/désactiver le préprocessing |
| **S** | 💾 Sauvegarder la dernière capture |
| **C** | 🗑️ Effacer le texte affiché |
| **L** | 🌍 Changer langue (ENG ↔ FRA) |
| **Q ou ESC** | ❌ Quitter l'application |

### **📺 Interface Console**

```
┌─────────────────────────────────────────────────────────┐
│ Camera OCR Test - Snapdragon X Plus                    │
│ Langue: ENG | Preprocessing: ON                        │
│ Captures: 3 | OCR: READY                               │
│ Controles: ESPACE=Capturer, P=Preprocessing, S=Sauver  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│               FLUX VIDÉO EN TEMPS RÉEL                 │
│                                                         │
├─────────────────────────────────────────────────────────┤
│ TEXTE DETECTE:                                          │
│ Hello World                                             │
│ This is a test document                                 │
│ Phone: +33 1 23 45 67 89                               │
└─────────────────────────────────────────────────────────┘
```

### **🔧 Fonctionnalités Avancées**

#### **OCR Asynchrone**
- Traitement en arrière-plan pour éviter les blocages
- Indicateur visuel de traitement
- Mise à jour en temps réel des résultats

#### **Préprocessing Intelligent**
```python
# Niveaux de préprocessing disponibles
enhancement_levels = ['light', 'medium', 'strong']

# Activation/désactivation avec touche P
self.use_preprocessing = True/False
```

#### **Analyse Détaillée**
- Fenêtre séparée avec boîtes de détection
- Codes couleur selon la confiance :
  - 🟢 **Vert** : Confiance ≥ 80%
  - 🟡 **Jaune** : Confiance ≥ 60%
  - 🟠 **Orange** : Confiance < 60%

#### **Sauvegarde Automatique**
```
camera_ocr_captures/
├── capture_20250107_143022.png    # Image originale
├── text_20250107_143022.txt       # Texte extrait + métadonnées
└── output_eng_capture_20250107_143022.png  # Image annotée
```

---

## 🖼️ Programme 2: Version GUI (`camera_ocr_gui.py`)

### **🚀 Démarrage**
```bash
python camera_ocr_gui.py
```

### **🎨 Interface Graphique**

```
┌─────────────────────────────────────────────────────────────────────┐
│ Camera OCR - Snapdragon X Plus                            [─][□][×] │
├─────────────────────────────────┬───────────────────────────────────┤
│ 📷 Flux Caméra                 │ ⚙️ Configuration OCR              │
│                                 │ Langue: [ENG ▼]                   │
│                                 │ ☑ Préprocessing d'image           │
│                                 │ Seuil confiance: [50──●────100]   │
│                                 │                                   │
│         VIDÉO EN TEMPS RÉEL     │ 📊 Statistiques                  │
│                                 │ ┌─────────────────────────────┐   │
│                                 │ │ Capture #3                  │   │
│                                 │ │ Temps OCR: 1.45s            │   │
│                                 │ │ Mots détectés: 12           │   │
│                                 │ │ Détections: 8               │   │
│                                 │ │ Confiance moy.: 78.3%       │   │
│                                 │ └─────────────────────────────┘   │
│                                 │                                   │
├─────────────────────────────────┤ 📝 Texte Détecté                │
│ [🔴 Démarrer] [📸 Capturer]    │ ┌─────────────────────────────┐   │
│ [💾 Sauvegarder]               │ │ Hello World                 │   │
│                                 │ │ This is a test document     │   │
│                                 │ │ Email: test@example.com     │   │
│                                 │ │ Phone: +33 1 23 45 67 89    │   │
│                                 │ │                             │   │
│                                 │ └─────────────────────────────┘   │
│                                 │ [🗑️ Effacer][📋 Copier][📄 Export] │
├─────────────────────────────────┴───────────────────────────────────┤
│ ✅ OCR terminé: 12 mots, 78.3% confiance                           │
└─────────────────────────────────────────────────────────────────────┘
```

### **🎛️ Contrôles GUI**

#### **Boutons Principaux**
- **🔴 Démarrer Caméra** : Active/désactive le flux vidéo
- **📸 Capturer + OCR** : Prend une photo et lance l'extraction
- **💾 Sauvegarder** : Sauvegarde l'image et le texte

#### **Configuration OCR**
- **Langue** : Sélection dans menu déroulant
- **Préprocessing** : Case à cocher
- **Seuil confiance** : Curseur de 0 à 100%

#### **Gestion des Résultats**
- **🗑️ Effacer** : Vide la zone de texte
- **📋 Copier** : Copie dans le presse-papiers
- **📄 Exporter** : Sauvegarde dans un fichier .txt

### **📊 Statistiques en Temps Réel**
```
Capture #5
Temps OCR: 1.23s
Mots détectés: 15
Détections: 12
Confiance moy.: 82.5%
Timestamp: 20250107_143045
```

### **🎨 Fonctionnalités GUI Spéciales**

#### **Interface Adaptative**
- Redimensionnement automatique
- Thème sombre optimisé
- Scrollbars automatiques

#### **Gestion Multi-thread**
- Interface non-bloquante
- OCR en arrière-plan
- Mise à jour fluide des résultats

#### **Export Avancé**
```python
# Dialogue de sauvegarde avec filtres
filetypes = [
    ("Fichiers texte", "*.txt"),
    ("Tous les fichiers", "*.*")
]

# Métadonnées complètes dans l'export
Export OCR - 2025-01-07 14:30:45
Langue: eng
Préprocessing: Activé
Seuil confiance: 70%

Texte détecté:
----------------
[Contenu extrait]
```

---

## ⚡ Comparaison des Versions

| Critère | Version Console | Version GUI |
|---------|----------------|-------------|
| **Performance** | ⭐⭐⭐⭐⭐ Plus rapide | ⭐⭐⭐⭐ Rapide |
| **Facilité d'usage** | ⭐⭐⭐ Moyennes touches clavier | ⭐⭐⭐⭐⭐ Très intuitive |
| **Fonctionnalités** | ⭐⭐⭐⭐ Complètes | ⭐⭐⭐⭐⭐ Très complètes |
| **Ressources** | ⭐⭐⭐⭐⭐ Légère | ⭐⭐⭐⭐ Modérée |
| **Debug** | ⭐⭐⭐⭐⭐ Console détaillée | ⭐⭐⭐ Interface graphique |

### **🎯 Quand Utiliser Chaque Version**

#### **Version Console - Recommandée pour :**
- Tests rapides et développement
- Performance maximale
- Intégration dans scripts automatisés
- Environnements sans interface graphique
- Debug approfondi

#### **Version GUI - Recommandée pour :**
- Utilisation quotidienne
- Démonstrations
- Utilisateurs non-techniques
- Export et sauvegarde faciles
- Configuration interactive

---

## 🔧 Configuration Avancée

### **Optimisation Performance Snapdragon X Plus**

#### **Variables d'Environnement**
```bash
# Limiter les threads pour ARM
set OMP_NUM_THREADS=4
set MKL_NUM_THREADS=4

# Optimisations mémoire
set OPENCV_FFMPEG_CAPTURE_OPTIONS=threads;4
```

#### **Configuration Caméra**
```python
# Dans les deux programmes
self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # Résolution
self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
self.cap.set(cv2.CAP_PROP_FPS, 30)             # Frame rate
self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)       # Buffer minimal
```

#### **Configuration OCR Optimale**
```python
# Pour texte standard
config = '--oem 3 --psm 6'

# Pour lignes uniques
config = '--oem 3 --psm 7'

# Pour mots isolés
config = '--oem 3 --psm 8'

# Pour texte sparse
config = '--oem 3 --psm 11'
```

### **Langues Supportées**
```python
# Langues courantes testées
supported_languages = [
    'eng',      # Anglais
    'fra',      # Français
    'deu',      # Allemand
    'spa',      # Espagnol
    'ita',      # Italien
    'por',      # Portugais
    'eng+fra'   # Multilingue
]
```

---

## 🛠️ Dépannage

### **❌ Erreurs Courantes**

#### **"Cannot open camera"**
```python
# Solutions possibles :
1. Vérifier les permissions caméra Windows
2. Fermer autres applications utilisant la caméra
3. Changer l'ID caméra : CameraOCR(camera_id=1)
4. Redémarrer l'application
```

#### **"TesseractOCR import error"**
```python
# Vérifier :
1. tesseract_complete_ocr.py dans le même dossier
2. Installation Tesseract complète
3. Variables d'environnement PATH
```

#### **"OCR processing too slow"**
```python
# Optimisations :
1. Réduire la résolution caméra
2. Désactiver le préprocessing
3. Utiliser PSM 8 (mot unique)
4. Limiter les threads système
```

#### **"Low OCR accuracy"**
```python
# Améliorations :
1. Activer le préprocessing 'strong'
2. Améliorer l'éclairage
3. Stabiliser la caméra
4. Utiliser la bonne langue
5. Ajuster le seuil de confiance
```

### **🔍 Debug Mode**

#### **Logs Console Détaillés**
```python
# Dans les deux programmes
print(f"📊 Résultats OCR:")
print(f"   📝 Texte: '{text[:100]}...'")
print(f"   📊 {word_count} mots, {detection_count} détections")
print(f"   🎯 Confiance moyenne: {avg_confidence:.1f}%")
print(f"   ⏱️  Temps: {processing_time:.2f}s")
```

#### **Test de Performance**
```python
# Benchmark intégré
def benchmark_ocr_performance():
    start_time = time.time()
    # ... OCR processing ...
    end_time = time.time()
    
    fps = 1.0 / (end_time - start_time)
    print(f"OCR FPS: {fps:.2f}")
```

---

## 📁 Structure des Fichiers de Sortie

### **Dossier de Sauvegarde**
```
camera_ocr_captures/           # Version console
camera_ocr_gui_captures/       # Version GUI
├── capture_20250107_143022.png              # Image originale
├── text_20250107_143022.txt                 # Texte + métadonnées
├── output_eng_capture_20250107_143022.png   # Image annotée (console)
└── rapport_ocr.html                         # Rapport complet (optionnel)
```

### **Format du Fichier Texte**
```
Capture OCR - 20250107_143022
Langue: eng
Préprocessing: Activé
Configuration: --oem 3 --psm 6
Seuil confiance: 50%

Statistiques:
  Temps traitement: 1.45s
  Mots détectés: 12
  Détections: 8
  Confiance moyenne: 78.3%

Texte détecté:
----------------------------------------
Hello World
This is a test document
Email: test@example.com
Phone: +33 1 23 45 67 89
```

---

## 🎉 Exemples d'Utilisation

### **📋 Numérisation de Documents**
1. Démarrer l'application
2. Placer le document devant la caméra
3. Ajuster l'éclairage
4. Appuyer sur ESPACE (console) ou Capturer (GUI)
5. Sauvegarder le résultat

### **🔍 Lecture de Codes/Numéros**
1. Configurer PSM 8 (mot unique) dans le code
2. Activer préprocessing 'strong'
3. Augmenter le seuil de confiance (70%+)
4. Capturer et vérifier

### **🌍 Texte Multilingue**
1. Configurer langue 'eng+fra'
2. Activer préprocessing 'medium'
3. Réduire le seuil de confiance (30%)
4. Capturer et analyser

**🎯 Ces programmes vous offrent une solution complète et optimisée pour l'OCR en temps réel sur Snapdragon X Plus !**