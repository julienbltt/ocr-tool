#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR Camera Real-Time Reader
Lecture de texte en temps réel depuis une caméra avec OCR

Author: Assistant
Date: 08/07/2025
Version: 1.0
"""

import cv2
import numpy as np
import time
import threading
from pathlib import Path
import logging
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
import queue
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

# Import du module OCR
try:
    from ocr import TesseractOCR, OCRResult, EnhancementLevel, PSMMode
except ImportError:
    print("ERREUR: Le module 'ocr.py' doit être dans le même répertoire que ce script")
    exit(1)

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class CameraConfig:
    """Configuration de la caméra et de l'OCR."""
    camera_id: int = 0
    resolution: Tuple[int, int] = (1280, 720)
    fps: int = 30
    ocr_interval: float = 1.0  # Intervalle entre les traitements OCR (secondes)
    confidence_threshold: float = 50.0
    languages: List[str] = None
    enhancement_level: EnhancementLevel = EnhancementLevel.MEDIUM
    psm_mode: PSMMode = PSMMode.SINGLE_BLOCK
    show_boxes: bool = True
    save_frames: bool = False


class OCRCameraReader:
    """
    Lecteur OCR en temps réel utilisant une caméra.
    
    Cette classe capture les images d'une caméra en temps réel, 
    les traite avec OCR et affiche les résultats sur l'écran.
    """
    
    def __init__(self, config: CameraConfig = None):
        """
        Initialise le lecteur OCR caméra.
        
        Args:
            config: Configuration de la caméra et OCR
        """
        self.config = config or CameraConfig()
        # S'assurer que languages est toujours une liste
        if self.config.languages is None:
            self.config.languages = ['eng']
        elif isinstance(self.config.languages, str):
            self.config.languages = [self.config.languages]
        elif not isinstance(self.config.languages, list):
            self.config.languages = ['eng']
        
        # Initialisation de l'OCR
        try:
            self.ocr = TesseractOCR()
            logger.info("OCR initialisé avec succès")
        except Exception as e:
            logger.error(f"Erreur d'initialisation OCR: {e}")
            raise
        
        # Variables d'état
        self.camera = None
        self.running = False
        self.current_frame = None
        self.last_ocr_time = 0
        self.ocr_results: List[OCRResult] = []
        self.ocr_text = ""
        self.processing_time = 0
        self.frame_count = 0
        
        # Queue pour les frames à traiter
        self.frame_queue = queue.Queue(maxsize=2)
        
        # Thread pour le traitement OCR
        self.ocr_thread = None
        
        # Statistiques
        self.stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'average_processing_time': 0,
            'detection_count': 0
        }
    
    def initialize_camera(self) -> bool:
        """
        Initialise la caméra avec la configuration spécifiée.
        
        Returns:
            True si l'initialisation réussit, False sinon
        """
        try:
            self.camera = cv2.VideoCapture(self.config.camera_id)
            
            if not self.camera.isOpened():
                logger.error(f"Impossible d'ouvrir la caméra {self.config.camera_id}")
                return False
            
            # Configuration de la caméra
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.resolution[0])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.resolution[1])
            self.camera.set(cv2.CAP_PROP_FPS, self.config.fps)
            
            # Vérification de la résolution effective
            actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"Caméra initialisée: {actual_width}x{actual_height} @ {actual_fps} FPS")
            return True
            
        except Exception as e:
            logger.error(f"Erreur d'initialisation caméra: {e}")
            return False
    
    def process_frame_ocr(self, frame: np.ndarray) -> Tuple[str, List[OCRResult], float]:
        """
        Traite une frame avec OCR.
        
        Args:
            frame: Frame à traiter
            
        Returns:
            Tuple (texte_extrait, résultats_avec_boxes, temps_traitement)
        """
        start_time = time.perf_counter()
        
        try:
            # Conversion en PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Extraction du texte avec boîtes
            results = self.ocr.extract_text_with_boxes(
                pil_image,
                languages=self.config.languages,
                confidence_threshold=self.config.confidence_threshold,
                enhancement_level=self.config.enhancement_level
            )
            
            # Extraction du texte simple
            text = self.ocr.extract_text(
                pil_image,
                languages=self.config.languages,
                psm_mode=self.config.psm_mode,
                enhancement_level=self.config.enhancement_level
            )
            
            processing_time = time.perf_counter() - start_time
            
            return text, results, processing_time
            
        except Exception as e:
            logger.error(f"Erreur de traitement OCR: {e}")
            return "", [], 0
    
    def ocr_worker(self):
        """Thread worker pour le traitement OCR."""
        while self.running:
            try:
                # Récupération d'une frame depuis la queue
                frame = self.frame_queue.get(timeout=1.0)
                
                if frame is not None:
                    # Traitement OCR
                    text, results, proc_time = self.process_frame_ocr(frame)
                    
                    # Mise à jour des résultats
                    self.ocr_text = text
                    self.ocr_results = results
                    self.processing_time = proc_time
                    
                    # Mise à jour des statistiques
                    self.stats['processed_frames'] += 1
                    self.stats['detection_count'] += len(results)
                    
                    # Calcul du temps de traitement moyen
                    total_processed = self.stats['processed_frames']
                    if total_processed > 0:
                        avg_time = self.stats['average_processing_time']
                        self.stats['average_processing_time'] = (
                            (avg_time * (total_processed - 1) + proc_time) / total_processed
                        )
                
                self.frame_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Erreur dans le thread OCR: {e}")
    
    def draw_ocr_results(self, frame: np.ndarray) -> np.ndarray:
        """
        Dessine les résultats OCR sur la frame.
        
        Args:
            frame: Frame originale
            
        Returns:
            Frame avec les résultats OCR dessinés
        """
        if not self.config.show_boxes or not self.ocr_results:
            return frame
        
        # Copie de la frame pour éviter de modifier l'original
        result_frame = frame.copy()
        
        # Couleurs pour différents niveaux de confiance
        colors = {
            'high': (0, 255, 0),     # Vert pour >80%
            'medium': (0, 255, 255), # Jaune pour 60-80%
            'low': (0, 165, 255)     # Orange pour <60%
        }
        
        for result in self.ocr_results:
            # Détermination de la couleur selon la confiance
            if result.confidence >= 80:
                color = colors['high']
            elif result.confidence >= 60:
                color = colors['medium']
            else:
                color = colors['low']
            
            # Dessin du rectangle
            cv2.rectangle(result_frame,
                         (result.x, result.y),
                         (result.x + result.width, result.y + result.height),
                         color, 2)
            
            # Préparation du label
            label = f"{result.text} ({result.confidence:.1f}%)"
            if len(label) > 25:
                label = label[:22] + "..."
            
            # Dessin du fond du label
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(result_frame,
                         (result.x, result.y - 25),
                         (result.x + label_size[0], result.y),
                         color, -1)
            
            # Dessin du texte
            cv2.putText(result_frame, label,
                       (result.x, result.y - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return result_frame
    
    def draw_info_overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        Dessine les informations système sur la frame.
        
        Args:
            frame: Frame originale
            
        Returns:
            Frame avec les informations système
        """
        info_frame = frame.copy()
        height, width = info_frame.shape[:2]
        
        # Informations à afficher
        info_lines = [
            f"FPS: {self.config.fps} | Resolution: {width}x{height}",
            f"OCR Language: {'+'.join(self.config.languages)}",
            f"Confidence: {self.config.confidence_threshold}%",
            f"Processing Time: {self.processing_time:.3f}s",
            f"Detections: {len(self.ocr_results)}",
            f"Processed: {self.stats['processed_frames']}/{self.stats['total_frames']}"
        ]
        
        # Fond semi-transparent pour les informations
        overlay = info_frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, info_frame, 0.3, 0, info_frame)
        
        # Affichage des informations
        for i, line in enumerate(info_lines):
            y_pos = 30 + i * 20
            cv2.putText(info_frame, line, (15, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return info_frame
    
    def save_frame_with_results(self, frame: np.ndarray, timestamp: str):
        """
        Sauvegarde une frame avec les résultats OCR.
        
        Args:
            frame: Frame à sauvegarder
            timestamp: Horodatage pour le nom du fichier
        """
        if not self.config.save_frames:
            return
        
        try:
            # Création du dossier de sauvegarde
            save_dir = Path("ocr_captures")
            save_dir.mkdir(exist_ok=True)
            
            # Nom du fichier
            filename = f"ocr_frame_{timestamp}.png"
            filepath = save_dir / filename
            
            # Sauvegarde
            cv2.imwrite(str(filepath), frame)
            
            # Sauvegarde des résultats texte
            if self.ocr_text:
                text_file = save_dir / f"ocr_text_{timestamp}.txt"
                with open(text_file, 'w', encoding='utf-8') as f:
                    f.write(f"Timestamp: {timestamp}\n")
                    f.write(f"Detections: {len(self.ocr_results)}\n")
                    f.write(f"Processing Time: {self.processing_time:.3f}s\n\n")
                    f.write("Extracted Text:\n")
                    f.write(self.ocr_text)
            
            logger.info(f"Frame sauvegardée: {filepath}")
            
        except Exception as e:
            logger.error(f"Erreur de sauvegarde: {e}")
    
    def run(self):
        """
        Lance la capture et le traitement en temps réel.
        """
        # Initialisation de la caméra
        if not self.initialize_camera():
            logger.error("Impossible d'initialiser la caméra")
            return
        
        # Démarrage du thread OCR
        self.running = True
        self.ocr_thread = threading.Thread(target=self.ocr_worker, daemon=True)
        self.ocr_thread.start()
        
        logger.info("Démarrage de la capture en temps réel...")
        logger.info("Appuyez sur 'q' pour quitter, 's' pour sauvegarder, 'c' pour changer la config")
        
        try:
            while self.running:
                # Capture d'une frame
                ret, frame = self.camera.read()
                if not ret:
                    logger.error("Erreur de capture de frame")
                    break
                
                self.current_frame = frame
                self.stats['total_frames'] += 1
                
                # Ajout de la frame à la queue pour traitement OCR (si c'est le moment)
                current_time = time.time()
                if current_time - self.last_ocr_time >= self.config.ocr_interval:
                    if not self.frame_queue.full():
                        self.frame_queue.put(frame.copy())
                        self.last_ocr_time = current_time
                
                # Dessin des résultats OCR
                display_frame = self.draw_ocr_results(frame)
                
                # Ajout des informations système
                display_frame = self.draw_info_overlay(display_frame)
                
                # Affichage de la frame
                cv2.imshow('OCR Camera Reader', display_frame)
                
                # Gestion des touches
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    self.save_frame_with_results(display_frame, timestamp)
                elif key == ord('c'):
                    self.show_config_window()
                elif key == ord('h'):
                    self.show_help()
        
        except KeyboardInterrupt:
            logger.info("Interruption par l'utilisateur")
        
        finally:
            self.cleanup()
    
    def show_help(self):
        """Affiche l'aide dans la console."""
        # S'assurer que languages est une liste
        languages = self.config.languages if isinstance(self.config.languages, list) else [self.config.languages]
        
        help_text = """
        === OCR Camera Reader - Aide ===
        
        Touches:
        - 'q': Quitter l'application
        - 's': Sauvegarder la frame actuelle avec résultats
        - 'c': Ouvrir la fenêtre de configuration
        - 'h': Afficher cette aide
        
        Informations affichées:
        - Boîtes vertes: Confiance >80%
        - Boîtes jaunes: Confiance 60-80%
        - Boîtes orange: Confiance <60%
        
        Configuration actuelle:
        - Langue(s): {languages}
        - Seuil de confiance: {confidence}%
        - Intervalle OCR: {interval}s
        - Résolution: {resolution}
        """.format(
            languages='+'.join(languages),
            confidence=self.config.confidence_threshold,
            interval=self.config.ocr_interval,
            resolution=f"{self.config.resolution[0]}x{self.config.resolution[1]}"
        )
        
        print(help_text)
    
    def show_config_window(self):
        """Affiche une fenêtre de configuration simple."""
        # S'assurer que languages est une liste
        languages = self.config.languages if isinstance(self.config.languages, list) else [self.config.languages]
        
        # Cette fonction pourrait être étendue avec une interface Tkinter
        # Pour l'instant, affichage dans la console
        print("\n=== Configuration Actuelle ===")
        print(f"Langue(s): {'+'.join(languages)}")
        print(f"Seuil de confiance: {self.config.confidence_threshold}%")
        print(f"Intervalle OCR: {self.config.ocr_interval}s")
        print(f"Amélioration: {self.config.enhancement_level}")
        print(f"Mode PSM: {self.config.psm_mode}")
        print(f"Affichage des boîtes: {self.config.show_boxes}")
        print(f"Sauvegarde: {self.config.save_frames}")
        print("===============================\n")
    
    def cleanup(self):
        """Nettoie les ressources utilisées."""
        logger.info("Nettoyage des ressources...")
        
        self.running = False
        
        # Arrêt du thread OCR
        if self.ocr_thread and self.ocr_thread.is_alive():
            self.ocr_thread.join(timeout=2)
        
        # Libération de la caméra
        if self.camera:
            self.camera.release()
        
        # Fermeture des fenêtres OpenCV
        cv2.destroyAllWindows()
        
        # Nettoyage du cache OCR
        if hasattr(self, 'ocr'):
            self.ocr.clear_cache()
        
        # Affichage des statistiques finales
        print("\n=== Statistiques Finales ===")
        print(f"Frames totales: {self.stats['total_frames']}")
        print(f"Frames traitées: {self.stats['processed_frames']}")
        print(f"Détections totales: {self.stats['detection_count']}")
        print(f"Temps moyen traitement: {self.stats['average_processing_time']:.3f}s")
        print("============================")
        
        logger.info("Nettoyage terminé")


def create_gui_config() -> Optional[CameraConfig]:
    """
    Crée une interface graphique pour la configuration.
    
    Returns:
        Configuration sélectionnée ou None si annulé
    """
    root = tk.Tk()
    root.title("Configuration OCR Camera")
    root.geometry("400x500")
    
    config = CameraConfig()
    # S'assurer que languages est une liste
    if config.languages is None:
        config.languages = ['eng']
    elif isinstance(config.languages, str):
        config.languages = [config.languages]
    elif not isinstance(config.languages, list):
        config.languages = ['eng']
    
    result = [None]  # Utilisation d'une liste pour modification dans les callbacks
    
    # Variables tkinter
    camera_id_var = tk.StringVar(value=str(config.camera_id))
    confidence_var = tk.StringVar(value=str(config.confidence_threshold))
    interval_var = tk.StringVar(value=str(config.ocr_interval))
    languages_var = tk.StringVar(value='+'.join(config.languages))
    enhancement_var = tk.StringVar(value=config.enhancement_level.value)
    show_boxes_var = tk.BooleanVar(value=config.show_boxes)
    save_frames_var = tk.BooleanVar(value=config.save_frames)
    
    # Interface
    frame = ttk.Frame(root, padding="10")
    frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    
    # Configuration de la caméra
    ttk.Label(frame, text="ID Caméra:").grid(row=0, column=0, sticky=tk.W)
    ttk.Entry(frame, textvariable=camera_id_var, width=10).grid(row=0, column=1, sticky=tk.W)
    
    ttk.Label(frame, text="Seuil de confiance (%):").grid(row=1, column=0, sticky=tk.W)
    ttk.Entry(frame, textvariable=confidence_var, width=10).grid(row=1, column=1, sticky=tk.W)
    
    ttk.Label(frame, text="Intervalle OCR (s):").grid(row=2, column=0, sticky=tk.W)
    ttk.Entry(frame, textvariable=interval_var, width=10).grid(row=2, column=1, sticky=tk.W)
    
    ttk.Label(frame, text="Langues (ex: eng+fra):").grid(row=3, column=0, sticky=tk.W)
    ttk.Entry(frame, textvariable=languages_var, width=15).grid(row=3, column=1, sticky=tk.W)
    
    ttk.Label(frame, text="Amélioration:").grid(row=4, column=0, sticky=tk.W)
    enhancement_combo = ttk.Combobox(frame, textvariable=enhancement_var, width=12)
    enhancement_combo['values'] = ('light', 'medium', 'strong', 'custom')
    enhancement_combo.grid(row=4, column=1, sticky=tk.W)
    
    ttk.Checkbutton(frame, text="Afficher les boîtes", variable=show_boxes_var).grid(row=5, column=0, columnspan=2, sticky=tk.W)
    ttk.Checkbutton(frame, text="Sauvegarder les frames", variable=save_frames_var).grid(row=6, column=0, columnspan=2, sticky=tk.W)
    
    def on_ok():
        try:
            config.camera_id = int(camera_id_var.get())
            config.confidence_threshold = float(confidence_var.get())
            config.ocr_interval = float(interval_var.get())
            config.languages = languages_var.get().split('+')
            config.enhancement_level = EnhancementLevel(enhancement_var.get())
            config.show_boxes = show_boxes_var.get()
            config.save_frames = save_frames_var.get()
            
            result[0] = config
            root.destroy()
        except ValueError as e:
            messagebox.showerror("Erreur", f"Valeur invalide: {e}")
    
    def on_cancel():
        root.destroy()
    
    # Boutons
    button_frame = ttk.Frame(frame)
    button_frame.grid(row=7, column=0, columnspan=2, pady=10)
    
    ttk.Button(button_frame, text="OK", command=on_ok).grid(row=0, column=0, padx=5)
    ttk.Button(button_frame, text="Annuler", command=on_cancel).grid(row=0, column=1, padx=5)
    
    # Centrage de la fenêtre
    root.update_idletasks()
    root.mainloop()
    
    return result[0]


def main():
    """Fonction principale."""
    print("=== OCR Camera Reader ===")
    print("Lecteur de texte en temps réel avec caméra")
    print()
    
    # Vérification de l'installation OCR
    try:
        from ocr import verify_installation
        if not verify_installation():
            print("ERREUR: Installation OCR incomplète")
            return
    except ImportError:
        print("ERREUR: Module OCR non trouvé")
        return
    
    # Configuration
    print("Configuration du système...")
    print("1. Utiliser la configuration par défaut")
    print("2. Configuration via interface graphique")
    print("3. Configuration manuelle")
    
    try:
        choice = input("Votre choix (1-3): ").strip()
        
        if choice == "2":
            config = create_gui_config()
            if config is None:
                print("Configuration annulée")
                return
        elif choice == "3":
            config = CameraConfig()
            # Configuration manuelle simplifiée
            config.camera_id = int(input(f"ID caméra [{config.camera_id}]: ") or config.camera_id)
            config.confidence_threshold = float(input(f"Seuil confiance [{config.confidence_threshold}]: ") or config.confidence_threshold)
            config.ocr_interval = float(input(f"Intervalle OCR [{config.ocr_interval}]: ") or config.ocr_interval)
            
            # S'assurer que languages est une liste
            languages = config.languages if isinstance(config.languages, list) else [config.languages]
            langs = input(f"Langues [{'+'.join(languages)}]: ") or '+'.join(languagaes)
            config.languages = langs.split('+')
        else:
            config = CameraConfig()
        
        # Création et lancement du lecteur
        reader = OCRCameraReader(config)
        reader.run()
        
    except KeyboardInterrupt:
        print("\nArrêt demandé par l'utilisateur")
    except Exception as e:
        logger.error(f"Erreur: {e}")
    finally:
        print("Fin du programme")


if __name__ == "__main__":
    main()