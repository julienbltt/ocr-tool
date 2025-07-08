#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Surveillance OCR en Temps Réel
Surveille un dossier et traite automatiquement les nouvelles images

Compatible Snapdragon X Plus - Windows ARM64
Utilise la classe TesseractOCR

Fonctionnalités:
- Surveillance de dossier en temps réel
- Traitement automatique des nouvelles images
- Interface de monitoring avec statistiques
- Notifications et alertes
- Logs détaillés

Auteur: Assistant IA
Date: 2025
"""

import os
import sys
import time
import json
import threading
import queue
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import psutil

# Import de la classe TesseractOCR
try:
    from tesseract_complete_ocr import TesseractOCR
except ImportError:
    print("❌ Impossible d'importer TesseractOCR")
    print("   Assurez-vous que tesseract_complete_ocr.py est dans le même répertoire")
    sys.exit(1)

class OCRFileHandler(FileSystemEventHandler):
    """Gestionnaire d'événements pour surveillance de fichiers"""
    
    def __init__(self, monitor_instance):
        super().__init__()
        self.monitor = monitor_instance
        self.supported_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.gif', '.pdf'}
    
    def on_created(self, event):
        """Fichier créé"""
        if not event.is_directory:
            file_path = Path(event.src_path)
            if file_path.suffix.lower() in self.supported_extensions:
                self.monitor.queue_file_for_processing(file_path, 'created')
    
    def on_modified(self, event):
        """Fichier modifié"""
        if not event.is_directory:
            file_path = Path(event.src_path)
            if file_path.suffix.lower() in self.supported_extensions:
                # Attendre que le fichier soit complètement écrit
                self.monitor.queue_file_for_processing(file_path, 'modified', delay=2.0)
    
    def on_moved(self, event):
        """Fichier déplacé"""
        if not event.is_directory:
            file_path = Path(event.dest_path)
            if file_path.suffix.lower() in self.supported_extensions:
                self.monitor.queue_file_for_processing(file_path, 'moved')

class OCRMonitor:
    """Moniteur OCR en temps réel"""
    
    def __init__(self, watch_folder: str, output_folder: str = None):
        """Initialiser le moniteur"""
        
        self.watch_folder = Path(watch_folder)
        
        if not self.watch_folder.exists():
            raise ValueError(f"Dossier à surveiller non trouvé: {watch_folder}")
        
        if output_folder:
            self.output_folder = Path(output_folder)
        else:
            self.output_folder = self.watch_folder / "ocr_monitor_output"
        
        self.output_folder.mkdir(exist_ok=True)
        
        # Configuration
        self.languages = ['eng']
        self.config = '--oem 3 --psm 6'
        self.auto_delete_processed = False
        self.max_queue_size = 50
        
        # État du monitoring
        self.is_monitoring = False
        self.observer = None
        self.processing_queue = queue.Queue(maxsize=self.max_queue_size)
        self.processed_files = set()
        self.processing_thread = None
        
        # Statistiques
        self.stats = {
            'files_detected': 0,
            'files_processed': 0,
            'files_failed': 0,
            'total_processing_time': 0,
            'total_text_extracted': 0,
            'start_time': None,
            'last_activity': None,
            'queue_size': 0,
            'current_status': 'Arrêté'
        }
        
        # OCR
        print("🔧 Initialisation OCR...")
        self.ocr = TesseractOCR()
        print(f"✅ OCR initialisé")
        
        # Interface
        self.gui = None
        self.create_gui()
        
        print(f"👁️  Moniteur OCR initialisé")
        print(f"📁 Dossier surveillé: {self.watch_folder}")
        print(f"📁 Dossier de sortie: {self.output_folder}")
    
    def create_gui(self):
        """Créer l'interface graphique de monitoring"""
        
        self.root = tk.Tk()
        self.root.title("OCR Monitor - Surveillance Temps Réel")
        self.root.geometry("1000x700")
        self.root.configure(bg='#2b2b2b')
        
        # Style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Frame principal
        main_frame = tk.Frame(self.root, bg='#2b2b2b')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # === SECTION CONTRÔLES ===
        control_frame = tk.LabelFrame(main_frame, text="🎛️ Contrôles", 
                                     font=('Arial', 12, 'bold'), fg='white', bg='#3b3b3b')
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Boutons de contrôle
        btn_frame = tk.Frame(control_frame, bg='#3b3b3b')
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.start_btn = tk.Button(btn_frame, text="▶️ Démarrer Surveillance",
                                  command=self.start_monitoring, font=('Arial', 11),
                                  bg='#4CAF50', fg='white', relief=tk.FLAT, padx=20)
        self.start_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_btn = tk.Button(btn_frame, text="⏹️ Arrêter",
                                 command=self.stop_monitoring, font=('Arial', 11),
                                 bg='#f44336', fg='white', relief=tk.FLAT, padx=20,
                                 state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.clear_btn = tk.Button(btn_frame, text="🗑️ Effacer Logs",
                                  command=self.clear_logs, font=('Arial', 11),
                                  bg='#FF9800', fg='white', relief=tk.FLAT, padx=20)
        self.clear_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Configuration
        config_frame = tk.Frame(control_frame, bg='#3b3b3b')
        config_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        tk.Label(config_frame, text="Langue:", font=('Arial', 10),
                fg='white', bg='#3b3b3b').pack(side=tk.LEFT)
        
        self.lang_var = tk.StringVar(value='eng')
        lang_combo = ttk.Combobox(config_frame, textvariable=self.lang_var,
                                 values=self.ocr.available_languages, width=10, state='readonly')
        lang_combo.pack(side=tk.LEFT, padx=(5, 20))
        
        self.auto_delete_var = tk.BooleanVar()
        auto_delete_cb = tk.Checkbutton(config_frame, text="Supprimer fichiers traités",
                                       variable=self.auto_delete_var, font=('Arial', 10),
                                       fg='white', bg='#3b3b3b', selectcolor='#2b2b2b')
        auto_delete_cb.pack(side=tk.LEFT)
        
        # === SECTION STATISTIQUES ===
        stats_frame = tk.LabelFrame(main_frame, text="📊 Statistiques", 
                                   font=('Arial', 12, 'bold'), fg='white', bg='#3b3b3b')
        stats_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Grille de statistiques
        stats_grid = tk.Frame(stats_frame, bg='#3b3b3b')
        stats_grid.pack(fill=tk.X, padx=10, pady=10)
        
        # Variables pour les statistiques
        self.status_var = tk.StringVar(value="Arrêté")
        self.detected_var = tk.StringVar(value="0")
        self.processed_var = tk.StringVar(value="0")
        self.failed_var = tk.StringVar(value="0")
        self.queue_var = tk.StringVar(value="0")
        self.uptime_var = tk.StringVar(value="00:00:00")
        
        # Créer les labels de statistiques
        stats_labels = [
            ("Statut:", self.status_var, 0, 0),
            ("Détectés:", self.detected_var, 0, 2),
            ("Traités:", self.processed_var, 0, 4),
            ("Échecs:", self.failed_var, 1, 0),
            ("File d'attente:", self.queue_var, 1, 2),
            ("Durée:", self.uptime_var, 1, 4)
        ]
        
        for label_text, var, row, col in stats_labels:
            tk.Label(stats_grid, text=label_text, font=('Arial', 10),
                    fg='#ccc', bg='#3b3b3b').grid(row=row, column=col, sticky='w', padx=(0, 5))
            
            tk.Label(stats_grid, textvariable=var, font=('Arial', 10, 'bold'),
                    fg='white', bg='#3b3b3b').grid(row=row, column=col+1, sticky='w', padx=(0, 20))
        
        # === SECTION LOGS ===
        log_frame = tk.LabelFrame(main_frame, text="📋 Journal d'Activité", 
                                 font=('Arial', 12, 'bold'), fg='white', bg='#3b3b3b')
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        # Zone de logs avec scrollbar
        self.log_text = scrolledtext.ScrolledText(log_frame, height=15, font=('Courier', 9),
                                                 bg='#1e1e1e', fg='#00ff00', relief=tk.FLAT,
                                                 wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Démarrer la mise à jour de l'interface
        self.update_gui()
        
        # Logs initiaux
        self.log_message("💫 Moniteur OCR initialisé")
        self.log_message(f"📁 Surveillance: {self.watch_folder}")
        self.log_message(f"📁 Sortie: {self.output_folder}")
    
    def log_message(self, message: str, level: str = "INFO"):
        """Ajouter un message au log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Couleurs selon le niveau
        colors = {
            'INFO': '#00ff00',    # Vert
            'WARNING': '#ffff00', # Jaune
            'ERROR': '#ff0000',   # Rouge
            'SUCCESS': '#00ffff'  # Cyan
        }
        
        color = colors.get(level, '#00ff00')
        
        if hasattr(self, 'log_text'):
            self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
            self.log_text.see(tk.END)
            
            # Limiter le nombre de lignes
            lines = self.log_text.get("1.0", tk.END).split('\n')
            if len(lines) > 1000:
                # Garder seulement les 800 dernières lignes
                self.log_text.delete("1.0", f"{len(lines)-800}.0")
        
        # Aussi imprimer en console
        print(f"[{timestamp}] {message}")
    
    def queue_file_for_processing(self, file_path: Path, event_type: str, delay: float = 0):
        """Ajouter un fichier à la queue de traitement"""
        
        if delay > 0:
            # Attendre que le fichier soit complètement écrit
            def delayed_queue():
                time.sleep(delay)
                if file_path.exists():
                    self._add_to_queue(file_path, event_type)
            
            threading.Thread(target=delayed_queue, daemon=True).start()
        else:
            self._add_to_queue(file_path, event_type)
    
    def _add_to_queue(self, file_path: Path, event_type: str):
        """Ajouter effectivement à la queue"""
        
        # Éviter les doublons
        file_key = str(file_path)
        if file_key in self.processed_files:
            return
        
        try:
            task = {
                'file_path': file_path,
                'event_type': event_type,
                'timestamp': datetime.now(),
                'retry_count': 0
            }
            
            self.processing_queue.put(task, block=False)
            self.stats['files_detected'] += 1
            self.stats['queue_size'] = self.processing_queue.qsize()
            
            self.log_message(f"📄 Nouveau fichier détecté: {file_path.name} ({event_type})")
            
        except queue.Full:
            self.log_message(f"⚠️ Queue pleine, fichier ignoré: {file_path.name}", "WARNING")
    
    def process_files_worker(self):
        """Worker pour traiter les fichiers en queue"""
        
        while self.is_monitoring:
            try:
                # Récupérer une tâche (timeout pour permettre l'arrêt)
                task = self.processing_queue.get(timeout=1.0)
                
                file_path = task['file_path']
                event_type = task['event_type']
                
                self.log_message(f"🔄 Traitement: {file_path.name}")
                
                # Vérifier que le fichier existe encore
                if not file_path.exists():
                    self.log_message(f"⚠️ Fichier non trouvé: {file_path.name}", "WARNING")
                    self.processing_queue.task_done()
                    continue
                
                # Traiter le fichier
                success = self.process_single_file(file_path)
                
                if success:
                    self.stats['files_processed'] += 1
                    self.processed_files.add(str(file_path))
                    self.log_message(f"✅ Traité avec succès: {file_path.name}", "SUCCESS")
                    
                    # Supprimer le fichier si demandé
                    if self.auto_delete_var.get():
                        try:
                            file_path.unlink()
                            self.log_message(f"🗑️ Fichier supprimé: {file_path.name}")
                        except Exception as e:
                            self.log_message(f"⚠️ Erreur suppression: {e}", "WARNING")
                else:
                    self.stats['files_failed'] += 1
                    self.log_message(f"❌ Échec traitement: {file_path.name}", "ERROR")
                
                self.stats['last_activity'] = datetime.now()
                self.stats['queue_size'] = self.processing_queue.qsize()
                self.processing_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.log_message(f"❌ Erreur worker: {e}", "ERROR")
    
    def process_single_file(self, file_path: Path) -> bool:
        """Traiter un seul fichier"""
        
        try:
            start_time = time.time()
            
            # Charger l'image
            from PIL import Image
            image = Image.open(file_path)
            
            # OCR
            text = self.ocr.extract_text_simple(
                image,
                lang=self.lang_var.get(),
                config=self.config
            )
            
            processing_time = time.time() - start_time
            self.stats['total_processing_time'] += processing_time
            
            # Statistiques du texte
            word_count = len(text.split()) if text else 0
            self.stats['total_text_extracted'] += len(text)
            
            # Sauvegarder les résultats
            self.save_ocr_results(file_path, text, processing_time, word_count)
            
            return True
            
        except Exception as e:
            self.log_message(f"❌ Erreur traitement {file_path.name}: {e}", "ERROR")
            return False
    
    def save_ocr_results(self, file_path: Path, text: str, processing_time: float, word_count: int):
        """Sauvegarder les résultats OCR"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{file_path.stem}_{timestamp}"
        
        # Fichier texte
        text_file = self.output_folder / f"{base_name}.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(f"OCR Monitor - Traitement Automatique\n")
            f.write(f"="*40 + "\n")
            f.write(f"Fichier source: {file_path.name}\n")
            f.write(f"Chemin complet: {file_path}\n")
            f.write(f"Date traitement: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Temps traitement: {processing_time:.2f}s\n")
            f.write(f"Langue OCR: {self.lang_var.get()}\n")
            f.write(f"Nombre de mots: {word_count}\n")
            f.write(f"Longueur texte: {len(text)} caractères\n")
            f.write(f"\n" + "="*40 + "\n")
            f.write("TEXTE EXTRAIT:\n")
            f.write("="*40 + "\n\n")
            f.write(text)
        
        # Fichier JSON pour traitement automatique
        json_file = self.output_folder / f"{base_name}.json"
        json_data = {
            'source_file': str(file_path),
            'processing_date': datetime.now().isoformat(),
            'processing_time': processing_time,
            'language': self.lang_var.get(),
            'word_count': word_count,
            'text_length': len(text),
            'text': text
        }
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    def start_monitoring(self):
        """Démarrer la surveillance"""
        
        if self.is_monitoring:
            return
        
        try:
            # Mettre à jour la configuration
            self.languages = [self.lang_var.get()]
            self.auto_delete_processed = self.auto_delete_var.get()
            
            # Créer l'observateur de fichiers
            self.observer = Observer()
            event_handler = OCRFileHandler(self)
            self.observer.schedule(event_handler, str(self.watch_folder), recursive=True)
            
            # Démarrer l'observateur
            self.observer.start()
            
            # Démarrer le thread de traitement
            self.is_monitoring = True
            self.processing_thread = threading.Thread(target=self.process_files_worker, daemon=True)
            self.processing_thread.start()
            
            # Réinitialiser les statistiques
            self.stats['start_time'] = datetime.now()
            self.stats['current_status'] = 'Actif'
            
            # Mettre à jour l'interface
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            
            self.log_message("🚀 Surveillance démarrée", "SUCCESS")
            self.log_message(f"👁️ Surveillance du dossier: {self.watch_folder}")
            
        except Exception as e:
            self.log_message(f"❌ Erreur démarrage: {e}", "ERROR")
            messagebox.showerror("Erreur", f"Impossible de démarrer la surveillance:\n{e}")
    
    def stop_monitoring(self):
        """Arrêter la surveillance"""
        
        if not self.is_monitoring:
            return
        
        try:
            # Arrêter l'observateur
            if self.observer:
                self.observer.stop()
                self.observer.join()
                self.observer = None
            
            # Arrêter le monitoring
            self.is_monitoring = False
            
            # Attendre que la queue soit vidée (max 10 secondes)
            if not self.processing_queue.empty():
                self.log_message("⏳ Traitement des fichiers restants...")
                start_wait = time.time()
                while not self.processing_queue.empty() and (time.time() - start_wait) < 10:
                    time.sleep(0.1)
            
            # Mettre à jour les statistiques
            self.stats['current_status'] = 'Arrêté'
            
            # Mettre à jour l'interface
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            
            self.log_message("⏹️ Surveillance arrêtée", "SUCCESS")
            
        except Exception as e:
            self.log_message(f"❌ Erreur arrêt: {e}", "ERROR")
    
    def clear_logs(self):
        """Effacer les logs"""
        if hasattr(self, 'log_text'):
            self.log_text.delete(1.0, tk.END)
            self.log_message("🗑️ Logs effacés")
    
    def update_gui(self):
        """Mettre à jour l'interface graphique"""
        
        try:
            # Mettre à jour les statistiques
            self.status_var.set(self.stats['current_status'])
            self.detected_var.set(str(self.stats['files_detected']))
            self.processed_var.set(str(self.stats['files_processed']))
            self.failed_var.set(str(self.stats['files_failed']))
            self.queue_var.set(str(self.stats['queue_size']))
            
            # Calculer la durée de fonctionnement
            if self.stats['start_time']:
                uptime = datetime.now() - self.stats['start_time']
                uptime_str = str(uptime).split('.')[0]  # Retirer les microsecondes
                self.uptime_var.set(uptime_str)
            else:
                self.uptime_var.set("00:00:00")
            
        except Exception as e:
            print(f"Erreur mise à jour GUI: {e}")
        
        # Programmer la prochaine mise à jour
        if hasattr(self, 'root'):
            self.root.after(1000, self.update_gui)  # Mise à jour chaque seconde
    
    def run(self):
        """Lancer l'interface graphique"""
        
        def on_closing():
            if self.is_monitoring:
                if messagebox.askokcancel("Quitter", "La surveillance est active. Voulez-vous vraiment quitter?"):
                    self.stop_monitoring()
                    time.sleep(1)  # Laisser le temps d'arrêter proprement
                    self.root.quit()
            else:
                self.root.quit()
        
        self.root.protocol("WM_DELETE_WINDOW", on_closing)
        self.root.mainloop()


class OCRMonitorCLI:
    """Version ligne de commande du moniteur"""
    
    def __init__(self, watch_folder: str, output_folder: str = None, languages: List[str] = None):
        self.watch_folder = Path(watch_folder)
        self.output_folder = Path(output_folder) if output_folder else self.watch_folder / "ocr_monitor_output"
        self.output_folder.mkdir(exist_ok=True)
        
        self.languages = languages or ['eng']
        self.is_monitoring = False
        self.observer = None
        self.stats = {'processed': 0, 'failed': 0}
        
        # OCR
        self.ocr = TesseractOCR()
        
        print(f"👁️ Moniteur OCR CLI initialisé")
        print(f"📁 Surveillance: {self.watch_folder}")
        print(f"📁 Sortie: {self.output_folder}")
    
    def process_file(self, file_path: Path):
        """Traiter un fichier"""
        try:
            print(f"🔄 Traitement: {file_path.name}")
            
            # OCR
            from PIL import Image
            image = Image.open(file_path)
            text = self.ocr.extract_text_simple(image, lang=self.languages[0])
            
            # Sauvegarder
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.output_folder / f"{file_path.stem}_{timestamp}.txt"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"Source: {file_path}\n")
                f.write(f"Date: {datetime.now()}\n")
                f.write(f"Langue: {self.languages[0]}\n\n")
                f.write(text)
            
            self.stats['processed'] += 1
            print(f"✅ Traité: {file_path.name}")
            
        except Exception as e:
            self.stats['failed'] += 1
            print(f"❌ Erreur {file_path.name}: {e}")
    
    def run_cli(self):
        """Lancer en mode CLI"""
        
        class CLIHandler(FileSystemEventHandler):
            def __init__(self, monitor):
                self.monitor = monitor
                self.supported_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.gif'}
            
            def on_created(self, event):
                if not event.is_directory:
                    file_path = Path(event.src_path)
                    if file_path.suffix.lower() in self.supported_extensions:
                        time.sleep(1)  # Attendre que le fichier soit complètement écrit
                        self.monitor.process_file(file_path)
        
        print("🚀 Démarrage de la surveillance...")
        
        try:
            self.observer = Observer()
            event_handler = CLIHandler(self)
            self.observer.schedule(event_handler, str(self.watch_folder), recursive=True)
            self.observer.start()
            self.is_monitoring = True
            
            print("👁️ Surveillance active. Appuyez sur Ctrl+C pour arrêter.")
            
            while self.is_monitoring:
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n🛑 Arrêt demandé...")
        finally:
            if self.observer:
                self.observer.stop()
                self.observer.join()
            
            print(f"📊 Statistiques finales:")
            print(f"   Traités: {self.stats['processed']}")
            print(f"   Échecs: {self.stats['failed']}")
            print("✅ Surveillance arrêtée")


def main():
    """Fonction principale"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Surveillance OCR en temps réel")
    parser.add_argument("watch_folder", help="Dossier à surveiller")
    parser.add_argument("-o", "--output", help="Dossier de sortie")
    parser.add_argument("-l", "--languages", nargs='+', default=['eng'], help="Langues OCR")
    parser.add_argument("--cli", action='store_true', help="Mode ligne de commande")
    
    args = parser.parse_args()
    
    print(f"👁️ MONITEUR OCR - SURVEILLANCE TEMPS RÉEL")
    print(f"=" * 50)
    
    try:
        if args.cli:
            # Mode CLI
            monitor = OCRMonitorCLI(args.watch_folder, args.output, args.languages)
            monitor.run_cli()
        else:
            # Mode GUI
            monitor = OCRMonitor(args.watch_folder, args.output)
            monitor.run()
            
    except KeyboardInterrupt:
        print(f"\n🛑 Surveillance interrompue")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Mode interactif si aucun argument
        print("👁️ Mode interactif")
        watch_folder = input("Dossier à surveiller: ").strip()
        
        if not watch_folder:
            print("❌ Dossier requis")
            sys.exit(1)
        
        try:
            monitor = OCRMonitor(watch_folder)
            monitor.run()
        except Exception as e:
            print(f"❌ Erreur: {e}")
    else:
        main()