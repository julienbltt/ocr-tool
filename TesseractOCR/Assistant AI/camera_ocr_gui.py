#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Camera OCR avec Interface Graphique Tkinter
Version alternative avec boutons et fenêtres séparées
Inclut la sélection de caméra

Compatible Snapdragon X Plus - Windows ARM64
Utilise la classe TesseractOCR

Auteur: Assistant IA
Date: 2025
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import cv2
from PIL import Image, ImageTk
import threading
import time
from datetime import datetime
from pathlib import Path
import queue
import numpy as np

# Import de la classe TesseractOCR
try:
    from tesseract_complete_ocr import TesseractOCR
except ImportError:
    print("❌ Impossible d'importer TesseractOCR")
    print("   Assurez-vous que tesseract_complete_ocr.py est dans le même répertoire")
    exit(1)

class CameraOCRGUI:
    """Interface graphique pour Camera OCR"""
    
    def __init__(self):
        """Initialiser l'interface graphique"""
        
        # Fenêtre principale
        self.root = tk.Tk()
        self.root.title("Camera OCR - Snapdragon X Plus")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2b2b2b')
        
        # Variables
        self.camera_active = False
        self.cap = None
        self.current_frame = None
        self.ocr_processing = False
        self.capture_counter = 0
        
        # Configuration OCR
        self.ocr = None
        self.ocr_language = tk.StringVar(value='eng')
        self.use_preprocessing = tk.BooleanVar(value=True)
        self.confidence_threshold = tk.IntVar(value=50)
        
        # Configuration caméra
        self.camera_index = tk.IntVar(value=0)
        self.available_cameras = []
        
        # Dossier de sauvegarde
        self.save_folder = "camera_ocr_gui_captures"
        Path(self.save_folder).mkdir(exist_ok=True)
        
        # Queue pour communication threads
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue()
        
        # Détecter les caméras disponibles
        self.detect_cameras()
        
        # Initialiser OCR
        self.init_ocr()
        
        # Créer l'interface
        self.create_interface()
        
        # Démarrer la mise à jour des frames
        self.update_display()
    
    def detect_cameras(self):
        """Détecter les caméras disponibles"""
        print("🔍 Détection des caméras...")
        self.available_cameras = []
        
        # Tester jusqu'à 10 indices de caméra
        for i in range(10):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    # Tester si on peut lire une frame
                    ret, frame = cap.read()
                    if ret:
                        # Essayer d'obtenir les propriétés de la caméra
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = int(cap.get(cv2.CAP_PROP_FPS))
                        
                        camera_info = {
                            'index': i,
                            'name': f"Caméra {i}",
                            'resolution': f"{width}x{height}",
                            'fps': fps
                        }
                        
                        # Essayer d'obtenir le nom de la caméra (Windows)
                        try:
                            backend = cap.getBackendName()
                            if backend:
                                camera_info['name'] = f"Caméra {i} ({backend})"
                        except:
                            pass
                        
                        self.available_cameras.append(camera_info)
                        print(f"✅ Caméra trouvée: Index {i} - {width}x{height}@{fps}fps")
                
                cap.release()
                
            except Exception as e:
                # Cette caméra n'est pas disponible
                continue
        
        if not self.available_cameras:
            print("❌ Aucune caméra détectée")
            self.available_cameras = [{'index': 0, 'name': 'Caméra par défaut', 'resolution': 'Inconnue', 'fps': 0}]
        else:
            print(f"✅ {len(self.available_cameras)} caméra(s) détectée(s)")
    
    def init_ocr(self):
        """Initialiser Tesseract OCR"""
        try:
            self.ocr = TesseractOCR()
            self.available_languages = self.ocr.available_languages
            print(f"✅ OCR initialisé, langues: {self.available_languages}")
        except Exception as e:
            messagebox.showerror("Erreur OCR", f"Impossible d'initialiser Tesseract:\n{e}")
            self.root.quit()
    
    def create_interface(self):
        """Créer l'interface utilisateur"""
        
        # Style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configuration des couleurs
        self.root.configure(bg='#2b2b2b')
        
        # Frame principal
        main_frame = tk.Frame(self.root, bg='#2b2b2b')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # === PARTIE GAUCHE: CAMÉRA ===
        left_frame = tk.Frame(main_frame, bg='#2b2b2b')
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Titre caméra
        camera_title = tk.Label(left_frame, text="📷 Flux Caméra", 
                               font=('Arial', 16, 'bold'), fg='white', bg='#2b2b2b')
        camera_title.pack(pady=(0, 10))
        
        # Zone d'affichage caméra
        self.camera_frame = tk.Frame(left_frame, bg='black', relief=tk.SUNKEN, bd=2)
        self.camera_frame.pack(fill=tk.BOTH, expand=True)
        
        self.camera_label = tk.Label(self.camera_frame, text="Caméra non activée", 
                                    font=('Arial', 14), fg='white', bg='black')
        self.camera_label.pack(expand=True)
        
        # Contrôles caméra
        camera_controls = tk.Frame(left_frame, bg='#2b2b2b')
        camera_controls.pack(fill=tk.X, pady=10)
        
        self.start_camera_btn = tk.Button(camera_controls, text="🔴 Démarrer Caméra",
                                         command=self.toggle_camera, font=('Arial', 12),
                                         bg='#4CAF50', fg='white', relief=tk.FLAT, padx=20)
        self.start_camera_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.capture_btn = tk.Button(camera_controls, text="📸 Capturer + OCR",
                                    command=self.capture_and_ocr, font=('Arial', 12),
                                    bg='#2196F3', fg='white', relief=tk.FLAT, padx=20,
                                    state=tk.DISABLED)
        self.capture_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.save_btn = tk.Button(camera_controls, text="💾 Sauvegarder",
                                 command=self.save_results, font=('Arial', 12),
                                 bg='#FF9800', fg='white', relief=tk.FLAT, padx=20,
                                 state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT)
        
        # === PARTIE DROITE: CONFIGURATION ET RÉSULTATS ===
        right_frame = tk.Frame(main_frame, bg='#2b2b2b', width=400)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y)
        right_frame.pack_propagate(False)
        
        # Configuration Caméra
        camera_config_frame = tk.LabelFrame(right_frame, text="🎥 Configuration Caméra", 
                                           font=('Arial', 12, 'bold'), fg='white', bg='#3b3b3b')
        camera_config_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Sélection de caméra
        cam_select_frame = tk.Frame(camera_config_frame, bg='#3b3b3b')
        cam_select_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(cam_select_frame, text="Caméra:", font=('Arial', 10), 
                fg='white', bg='#3b3b3b').pack(side=tk.LEFT)
        
        # Créer les options pour la combobox
        camera_options = []
        for cam in self.available_cameras:
            option_text = f"{cam['name']} - {cam['resolution']}"
            camera_options.append(option_text)
        
        self.camera_combo = ttk.Combobox(cam_select_frame, values=camera_options,
                                        width=25, state='readonly')
        self.camera_combo.pack(side=tk.RIGHT)
        if camera_options:
            self.camera_combo.current(0)  # Sélectionner la première caméra
        
        # Lier l'événement de changement de sélection
        self.camera_combo.bind('<<ComboboxSelected>>', self.on_camera_changed)
        
        # Bouton refresh caméras
        refresh_cam_frame = tk.Frame(camera_config_frame, bg='#3b3b3b')
        refresh_cam_frame.pack(fill=tk.X, padx=10, pady=5)
        
        refresh_btn = tk.Button(refresh_cam_frame, text="🔄 Actualiser caméras",
                               command=self.refresh_cameras, font=('Arial', 9),
                               bg='#607D8B', fg='white', relief=tk.FLAT)
        refresh_btn.pack(side=tk.LEFT)
        
        # Informations caméra sélectionnée
        self.camera_info_label = tk.Label(camera_config_frame, text="",
                                         font=('Arial', 9), fg='#cccccc', bg='#3b3b3b',
                                         justify=tk.LEFT)
        self.camera_info_label.pack(fill=tk.X, padx=10, pady=5)
        
        # Mettre à jour les infos de la caméra sélectionnée
        self.update_camera_info()
        
        # Configuration OCR
        config_frame = tk.LabelFrame(right_frame, text="⚙️ Configuration OCR", 
                                    font=('Arial', 12, 'bold'), fg='white', bg='#3b3b3b')
        config_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Langue
        lang_frame = tk.Frame(config_frame, bg='#3b3b3b')
        lang_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(lang_frame, text="Langue:", font=('Arial', 10), 
                fg='white', bg='#3b3b3b').pack(side=tk.LEFT)
        
        lang_combo = ttk.Combobox(lang_frame, textvariable=self.ocr_language,
                                 values=self.available_languages, width=10, state='readonly')
        lang_combo.pack(side=tk.RIGHT)
        
        # Préprocessing
        preprocessing_cb = tk.Checkbutton(config_frame, text="Préprocessing d'image",
                                         variable=self.use_preprocessing, font=('Arial', 10),
                                         fg='white', bg='#3b3b3b', selectcolor='#2b2b2b')
        preprocessing_cb.pack(anchor=tk.W, padx=10, pady=5)
        
        # Seuil de confiance
        conf_frame = tk.Frame(config_frame, bg='#3b3b3b')
        conf_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(conf_frame, text="Seuil confiance:", font=('Arial', 10),
                fg='white', bg='#3b3b3b').pack(side=tk.LEFT)
        
        conf_scale = tk.Scale(conf_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                             variable=self.confidence_threshold, bg='#3b3b3b',
                             fg='white', highlightthickness=0, length=150)
        conf_scale.pack(side=tk.RIGHT)
        
        # Statistiques
        stats_frame = tk.LabelFrame(right_frame, text="📊 Statistiques", 
                                   font=('Arial', 12, 'bold'), fg='white', bg='#3b3b3b')
        stats_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.stats_text = tk.Text(stats_frame, height=6, font=('Courier', 9),
                                 bg='#1e1e1e', fg='#00ff00', relief=tk.FLAT)
        self.stats_text.pack(fill=tk.X, padx=10, pady=5)
        
        # Résultats OCR
        results_frame = tk.LabelFrame(right_frame, text="📝 Texte Détecté", 
                                     font=('Arial', 12, 'bold'), fg='white', bg='#3b3b3b')
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Zone de texte avec scrollbar
        text_frame = tk.Frame(results_frame, bg='#3b3b3b')
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.text_result = scrolledtext.ScrolledText(text_frame, font=('Arial', 10),
                                                    bg='#1e1e1e', fg='white',
                                                    relief=tk.FLAT, wrap=tk.WORD)
        self.text_result.pack(fill=tk.BOTH, expand=True)
        
        # Boutons résultats
        result_buttons = tk.Frame(results_frame, bg='#3b3b3b')
        result_buttons.pack(fill=tk.X, padx=10, pady=5)
        
        clear_btn = tk.Button(result_buttons, text="🗑️ Effacer",
                             command=self.clear_results, font=('Arial', 10),
                             bg='#f44336', fg='white', relief=tk.FLAT)
        clear_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        copy_btn = tk.Button(result_buttons, text="📋 Copier",
                            command=self.copy_text, font=('Arial', 10),
                            bg='#9C27B0', fg='white', relief=tk.FLAT)
        copy_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        export_btn = tk.Button(result_buttons, text="📄 Exporter",
                              command=self.export_text, font=('Arial', 10),
                              bg='#607D8B', fg='white', relief=tk.FLAT)
        export_btn.pack(side=tk.LEFT)
        
        # Barre de statut
        self.status_bar = tk.Label(self.root, text="✅ Prêt - Sélectionnez une caméra et cliquez sur 'Démarrer Caméra'",
                                  relief=tk.SUNKEN, anchor=tk.W, font=('Arial', 10),
                                  bg='#1e1e1e', fg='#00ff00')
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Initialiser les statistiques
        self.update_stats()
    
    def on_camera_changed(self, event=None):
        """Gestionnaire de changement de caméra sélectionnée"""
        if self.camera_active:
            messagebox.showwarning("Attention", "Arrêtez d'abord la caméra actuelle avant de changer de caméra")
            return
        
        # Mettre à jour l'index de la caméra sélectionnée
        selected_index = self.camera_combo.current()
        if selected_index >= 0 and selected_index < len(self.available_cameras):
            self.camera_index.set(self.available_cameras[selected_index]['index'])
            self.update_camera_info()
    
    def update_camera_info(self):
        """Mettre à jour les informations de la caméra sélectionnée"""
        selected_index = self.camera_combo.current() if hasattr(self, 'camera_combo') else 0
        if selected_index >= 0 and selected_index < len(self.available_cameras):
            cam_info = self.available_cameras[selected_index]
            info_text = f"Index: {cam_info['index']} | Résolution: {cam_info['resolution']} | FPS: {cam_info['fps']}"
            if hasattr(self, 'camera_info_label'):
                self.camera_info_label.config(text=info_text)
    
    def refresh_cameras(self):
        """Actualiser la liste des caméras disponibles"""
        if self.camera_active:
            messagebox.showwarning("Attention", "Arrêtez d'abord la caméra actuelle avant d'actualiser")
            return
        
        # Sauvegarder la sélection actuelle
        current_selection = self.camera_combo.current() if hasattr(self, 'camera_combo') else 0
        
        # Détecter à nouveau les caméras
        self.detect_cameras()
        
        # Mettre à jour la combobox
        camera_options = []
        for cam in self.available_cameras:
            option_text = f"{cam['name']} - {cam['resolution']}"
            camera_options.append(option_text)
        
        self.camera_combo['values'] = camera_options
        
        # Restaurer la sélection si possible
        if current_selection < len(camera_options):
            self.camera_combo.current(current_selection)
        else:
            self.camera_combo.current(0)
        
        self.update_camera_info()
        self.status_bar.config(text=f"🔄 Caméras actualisées: {len(self.available_cameras)} trouvée(s)")
    
    def toggle_camera(self):
        """Démarrer/arrêter la caméra"""
        if not self.camera_active:
            self.start_camera()
        else:
            self.stop_camera()
    
    def start_camera(self):
        """Démarrer la caméra"""
        try:
            # Obtenir l'index de la caméra sélectionnée
            selected_index = self.camera_combo.current()
            if selected_index >= 0 and selected_index < len(self.available_cameras):
                camera_index = self.available_cameras[selected_index]['index']
            else:
                camera_index = 0
            
            self.cap = cv2.VideoCapture(camera_index)
            
            if not self.cap.isOpened():
                raise Exception(f"Impossible d'ouvrir la caméra {camera_index}")
            
            # Configuration caméra
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            self.camera_active = True
            
            # Mettre à jour l'interface
            self.start_camera_btn.config(text="⏹️ Arrêter Caméra", bg='#f44336')
            self.capture_btn.config(state=tk.NORMAL)
            
            # Obtenir le nom de la caméra pour le statut
            camera_name = self.available_cameras[selected_index]['name'] if selected_index >= 0 else f"Caméra {camera_index}"
            self.status_bar.config(text=f"📷 {camera_name} active - Prêt à capturer")
            
            # Démarrer la lecture des frames
            self.read_camera_thread = threading.Thread(target=self.read_camera, daemon=True)
            self.read_camera_thread.start()
            
        except Exception as e:
            messagebox.showerror("Erreur Caméra", f"Impossible de démarrer la caméra:\n{e}")
    
    def stop_camera(self):
        """Arrêter la caméra"""
        self.camera_active = False
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # Réinitialiser l'affichage
        self.camera_label.config(image='', text="Caméra arrêtée")
        
        # Mettre à jour l'interface
        self.start_camera_btn.config(text="🔴 Démarrer Caméra", bg='#4CAF50')
        self.capture_btn.config(state=tk.DISABLED)
        self.status_bar.config(text="⏹️ Caméra arrêtée")
    
    def read_camera(self):
        """Lire les frames de la caméra (thread séparé)"""
        while self.camera_active and self.cap:
            ret, frame = self.cap.read()
            if ret:
                # Redimensionner pour l'affichage
                display_frame = cv2.resize(frame, (640, 480))
                
                # Mettre dans la queue (non-bloquant)
                try:
                    self.frame_queue.put(display_frame, block=False)
                    self.current_frame = frame.copy()  # Garder frame originale pour OCR
                except queue.Full:
                    # Queue pleine, ignorer cette frame
                    pass
            
            time.sleep(0.033)  # ~30 FPS
    
    def update_display(self):
        """Mettre à jour l'affichage (thread principal)"""
        try:
            # Récupérer la dernière frame
            while not self.frame_queue.empty():
                frame = self.frame_queue.get_nowait()
                
                # Convertir pour Tkinter
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                photo = ImageTk.PhotoImage(image)
                
                # Afficher
                self.camera_label.config(image=photo, text='')
                self.camera_label.image = photo  # Garder une référence
            
            # Vérifier les résultats OCR
            self.check_ocr_results()
            
        except queue.Empty:
            pass
        except Exception as e:
            print(f"Erreur affichage: {e}")
        
        # Programmer la prochaine mise à jour
        self.root.after(33, self.update_display)  # ~30 FPS
    
    def capture_and_ocr(self):
        """Capturer une image et lancer l'OCR"""
        if not self.camera_active or self.current_frame is None:
            messagebox.showwarning("Attention", "Caméra non active ou pas de frame disponible")
            return
        
        if self.ocr_processing:
            messagebox.showinfo("Information", "OCR en cours, veuillez attendre...")
            return
        
        self.ocr_processing = True
        self.capture_counter += 1
        
        # Mettre à jour l'interface
        self.capture_btn.config(state=tk.DISABLED, text="🔄 Traitement...")
        self.status_bar.config(text="🔍 OCR en cours...")
        
        # Lancer OCR en arrière-plan
        ocr_thread = threading.Thread(target=self.process_ocr, 
                                     args=(self.current_frame.copy(),), daemon=True)
        ocr_thread.start()
    
    def process_ocr(self, frame):
        """Traiter l'OCR (thread séparé)"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Convertir en image PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Préprocessing si activé
            if self.use_preprocessing.get():
                processed_image = self.ocr.preprocess_image(pil_image, enhancement_level='medium')
            else:
                processed_image = pil_image
            
            # OCR
            start_time = time.time()
            
            # Texte simple
            text = self.ocr.extract_text_simple(processed_image, lang=self.ocr_language.get())
            
            # Détections avec coordonnées
            boxes = self.ocr.extract_text_with_boxes(
                processed_image,
                lang=self.ocr_language.get(),
                confidence_threshold=self.confidence_threshold.get()
            )
            
            processing_time = time.time() - start_time
            
            # Calculer statistiques
            total_words = len(text.split()) if text else 0
            avg_confidence = np.mean([b['confidence'] for b in boxes]) if boxes else 0
            
            # Préparer le résultat
            result = {
                'text': text.strip(),
                'boxes': boxes,
                'stats': {
                    'processing_time': processing_time,
                    'word_count': total_words,
                    'detection_count': len(boxes),
                    'avg_confidence': avg_confidence,
                    'timestamp': timestamp
                },
                'image': frame.copy()
            }
            
            # Mettre le résultat dans la queue
            self.result_queue.put(result)
            
        except Exception as e:
            error_result = {
                'text': f"Erreur OCR: {e}",
                'error': True,
                'stats': {'processing_time': 0, 'word_count': 0, 'detection_count': 0, 'avg_confidence': 0}
            }
            self.result_queue.put(error_result)
    
    def check_ocr_results(self):
        """Vérifier et traiter les résultats OCR"""
        try:
            while not self.result_queue.empty():
                result = self.result_queue.get_nowait()
                
                # Mettre à jour le texte
                self.text_result.delete(1.0, tk.END)
                self.text_result.insert(1.0, result['text'])
                
                # Mettre à jour les statistiques
                self.last_result = result
                self.update_stats(result['stats'])
                
                # Réactiver les boutons
                self.ocr_processing = False
                self.capture_btn.config(state=tk.NORMAL, text="📸 Capturer + OCR")
                self.save_btn.config(state=tk.NORMAL)
                
                if result.get('error', False):
                    self.status_bar.config(text="❌ Erreur OCR")
                else:
                    stats = result['stats']
                    self.status_bar.config(text=f"✅ OCR terminé: {stats['word_count']} mots, {stats['avg_confidence']:.1f}% confiance")
                
                break
                
        except queue.Empty:
            pass
    
    def update_stats(self, stats=None):
        """Mettre à jour les statistiques"""
        self.stats_text.delete(1.0, tk.END)
        
        if stats:
            stats_info = f"""Capture #{self.capture_counter}
Temps OCR: {stats['processing_time']:.2f}s
Mots détectés: {stats['word_count']}
Détections: {stats['detection_count']}
Confiance moy.: {stats['avg_confidence']:.1f}%
Timestamp: {stats.get('timestamp', 'N/A')}"""
        else:
            selected_cam = "Non sélectionnée"
            if hasattr(self, 'camera_combo'):
                selected_index = self.camera_combo.current()
                if selected_index >= 0 and selected_index < len(self.available_cameras):
                    selected_cam = self.available_cameras[selected_index]['name']
            
            stats_info = f"""Caméras: {len(self.available_cameras)} détectée(s)
Caméra active: {selected_cam}
Captures totales: {self.capture_counter}
Langue OCR: {self.ocr_language.get().upper()}
Préprocessing: {'Activé' if self.use_preprocessing.get() else 'Désactivé'}
Seuil confiance: {self.confidence_threshold.get()}%"""
        
        self.stats_text.insert(1.0, stats_info)
    
    def clear_results(self):
        """Effacer les résultats"""
        self.text_result.delete(1.0, tk.END)
        self.status_bar.config(text="🗑️ Résultats effacés")
    
    def copy_text(self):
        """Copier le texte dans le presse-papiers"""
        text = self.text_result.get(1.0, tk.END).strip()
        if text:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.status_bar.config(text="📋 Texte copié dans le presse-papiers")
        else:
            messagebox.showinfo("Information", "Aucun texte à copier")
    
    def export_text(self):
        """Exporter le texte vers un fichier"""
        text = self.text_result.get(1.0, tk.END).strip()
        if not text:
            messagebox.showinfo("Information", "Aucun texte à exporter")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Fichiers texte", "*.txt"), ("Tous les fichiers", "*.*")],
            title="Exporter le texte"
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(f"Export OCR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Langue: {self.ocr_language.get()}\n")
                    f.write(f"Préprocessing: {'Activé' if self.use_preprocessing.get() else 'Désactivé'}\n\n")
                    f.write(text)
                
                self.status_bar.config(text=f"📄 Texte exporté: {filename}")
            except Exception as e:
                messagebox.showerror("Erreur", f"Impossible d'exporter:\n{e}")
    
    def save_results(self):
        """Sauvegarder les résultats complets"""
        if not hasattr(self, 'last_result'):
            messagebox.showinfo("Information", "Aucun résultat à sauvegarder")
            return
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Sauvegarder l'image
            image_path = Path(self.save_folder) / f"capture_{timestamp}.png"
            cv2.imwrite(str(image_path), self.last_result['image'])
            
            # Sauvegarder le texte
            text_path = Path(self.save_folder) / f"text_{timestamp}.txt"
            with open(text_path, 'w', encoding='utf-8') as f:
                # Informations caméra
                selected_index = self.camera_combo.current()
                camera_name = "Inconnue"
                if selected_index >= 0 and selected_index < len(self.available_cameras):
                    camera_name = self.available_cameras[selected_index]['name']
                
                f.write(f"Capture OCR - {timestamp}\n")
                f.write(f"Caméra: {camera_name}\n")
                f.write(f"Langue: {self.ocr_language.get()}\n")
                f.write(f"Préprocessing: {'Activé' if self.use_preprocessing.get() else 'Désactivé'}\n")
                f.write(f"Seuil confiance: {self.confidence_threshold.get()}%\n\n")
                f.write("Statistiques:\n")
                stats = self.last_result['stats']
                f.write(f"  Temps traitement: {stats['processing_time']:.2f}s\n")
                f.write(f"  Mots détectés: {stats['word_count']}\n")
                f.write(f"  Détections: {stats['detection_count']}\n")
                f.write(f"  Confiance moyenne: {stats['avg_confidence']:.1f}%\n\n")
                f.write("Texte détecté:\n")
                f.write("-" * 40 + "\n")
                f.write(self.last_result['text'])
            
            self.status_bar.config(text=f"💾 Sauvegardé: {image_path.name}")
            messagebox.showinfo("Succès", f"Résultats sauvegardés:\n📷 {image_path}\n📄 {text_path}")
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de sauvegarder:\n{e}")
    
    def on_closing(self):
        """Gérer la fermeture de l'application"""
        if self.camera_active:
            self.stop_camera()
        self.root.quit()
    
    def run(self):
        """Lancer l'application"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()


def main():
    """Fonction principale"""
    print("🚀 DÉMARRAGE CAMERA OCR GUI")
    print("=" * 50)
    
    try:
        app = CameraOCRGUI()
        app.run()
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()