#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Traitement OCR par Lots
Traite automatiquement plusieurs images dans un dossier

Compatible Snapdragon X Plus - Windows ARM64
Utilise la classe TesseractOCR

Fonctionnalités:
- Traitement massif de dossiers
- Formats multiples (PNG, JPG, PDF, TIFF)
- Rapports détaillés
- Reprise après interruption
- Filtrage et tri automatique

Auteur: Assistant IA
Date: 2025
"""

import os
import sys
import json
import time
import hashlib
import threading
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional
import argparse

# Imports pour différents formats
from PIL import Image
import fitz  # PyMuPDF pour PDF
import cv2
import numpy as np

# Import de la classe TesseractOCR
try:
    from tesseract_complete_ocr import TesseractOCR
except ImportError:
    print("❌ Impossible d'importer TesseractOCR")
    print("   Assurez-vous que tesseract_complete_ocr.py est dans le même répertoire")
    sys.exit(1)

class BatchOCRProcessor:
    """Processeur OCR pour traitement par lots"""
    
    def __init__(self, 
                 input_folder: str,
                 output_folder: str = None,
                 max_workers: int = 2,
                 languages: List[str] = None,
                 config: str = None):
        """
        Initialiser le processeur par lots
        
        Args:
            input_folder: Dossier contenant les images à traiter
            output_folder: Dossier de sortie (par défaut: input_folder + "_ocr")
            max_workers: Nombre de threads simultanés
            languages: Langues OCR à utiliser
            config: Configuration Tesseract personnalisée
        """
        
        self.input_folder = Path(input_folder)
        
        if not self.input_folder.exists():
            raise ValueError(f"Dossier d'entrée non trouvé: {input_folder}")
        
        # Dossier de sortie
        if output_folder:
            self.output_folder = Path(output_folder)
        else:
            self.output_folder = self.input_folder.parent / f"{self.input_folder.name}_ocr"
        
        self.output_folder.mkdir(exist_ok=True)
        
        # Configuration
        self.max_workers = min(max_workers, 4)  # Limiter pour ARM
        self.languages = languages or ['eng']
        self.config = config or '--oem 3 --psm 6'
        
        # Extensions supportées
        self.supported_extensions = {
            '.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.gif',
            '.pdf'  # Support PDF basique
        }
        
        # Statistiques
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'successful_files': 0,
            'failed_files': 0,
            'skipped_files': 0,
            'total_text_length': 0,
            'total_processing_time': 0,
            'start_time': None,
            'end_time': None
        }
        
        # État du traitement
        self.progress_file = self.output_folder / "progress.json"
        self.processed_hashes = self.load_progress()
        
        # Initialiser OCR
        print("🔧 Initialisation OCR...")
        self.ocr = TesseractOCR()
        print(f"✅ OCR initialisé, langues: {self.ocr.available_languages}")
        
        # Validation des langues
        for lang in self.languages:
            if lang not in self.ocr.available_languages and '+' not in lang:
                print(f"⚠️  Langue '{lang}' non disponible, utilisation de 'eng'")
                self.languages = ['eng']
                break
        
        print(f"📁 Dossier d'entrée: {self.input_folder}")
        print(f"📁 Dossier de sortie: {self.output_folder}")
        print(f"🔧 Configuration: {self.max_workers} workers, langues: {self.languages}")
    
    def load_progress(self) -> set:
        """Charger l'état de progression précédent"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    data = json.load(f)
                    return set(data.get('processed_hashes', []))
            except Exception as e:
                print(f"⚠️  Impossible de charger la progression: {e}")
        
        return set()
    
    def save_progress(self):
        """Sauvegarder l'état de progression"""
        try:
            progress_data = {
                'processed_hashes': list(self.processed_hashes),
                'last_update': datetime.now().isoformat(),
                'stats': self.stats
            }
            
            with open(self.progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
                
        except Exception as e:
            print(f"⚠️  Impossible de sauvegarder la progression: {e}")
    
    def get_file_hash(self, file_path: Path) -> str:
        """Calculer le hash d'un fichier pour éviter le retraitement"""
        hasher = hashlib.md5()
        
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            
            # Inclure aussi la taille et la date de modification
            stat = file_path.stat()
            hasher.update(f"{stat.st_size}_{stat.st_mtime}".encode())
            
            return hasher.hexdigest()
            
        except Exception:
            return f"error_{file_path.name}_{time.time()}"
    
    def find_files(self) -> List[Path]:
        """Trouver tous les fichiers à traiter"""
        files = []
        
        print("🔍 Recherche des fichiers...")
        
        for ext in self.supported_extensions:
            pattern = f"**/*{ext}"
            found_files = list(self.input_folder.glob(pattern))
            files.extend(found_files)
        
        # Supprimer les doublons et trier
        files = sorted(list(set(files)))
        
        print(f"📄 {len(files)} fichiers trouvés")
        
        # Filtrer les fichiers déjà traités
        new_files = []
        for file_path in files:
            file_hash = self.get_file_hash(file_path)
            if file_hash not in self.processed_hashes:
                new_files.append(file_path)
            else:
                self.stats['skipped_files'] += 1
        
        if len(new_files) != len(files):
            print(f"📋 {len(files) - len(new_files)} fichiers déjà traités (ignorés)")
        
        self.stats['total_files'] = len(new_files)
        return new_files
    
    def convert_pdf_to_images(self, pdf_path: Path) -> List[np.ndarray]:
        """Convertir PDF en images"""
        images = []
        
        try:
            # Ouvrir le PDF
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Convertir en image (300 DPI pour qualité OCR)
                mat = fitz.Matrix(300/72, 300/72)  # 300 DPI
                pix = page.get_pixmap(matrix=mat)
                
                # Convertir en array numpy
                img_data = pix.tobytes("png")
                img_array = np.frombuffer(img_data, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                if img is not None:
                    images.append(img)
            
            doc.close()
            print(f"📖 PDF converti: {len(images)} pages")
            
        except Exception as e:
            print(f"❌ Erreur conversion PDF {pdf_path.name}: {e}")
        
        return images
    
    def process_single_file(self, file_path: Path) -> Dict:
        """Traiter un seul fichier"""
        result = {
            'file_path': str(file_path),
            'file_name': file_path.name,
            'file_size': 0,
            'processing_time': 0,
            'success': False,
            'error': None,
            'text_length': 0,
            'word_count': 0,
            'detection_count': 0,
            'avg_confidence': 0,
            'languages_used': [],
            'output_files': []
        }
        
        start_time = time.time()
        
        try:
            # Informations du fichier
            result['file_size'] = file_path.stat().st_size
            file_ext = file_path.suffix.lower()
            
            print(f"🔄 Traitement: {file_path.name}")
            
            # Traitement selon le type de fichier
            if file_ext == '.pdf':
                images = self.convert_pdf_to_images(file_path)
                if not images:
                    raise Exception("Aucune page convertible trouvée dans le PDF")
                
                # Traiter chaque page
                all_texts = []
                all_boxes = []
                
                for page_num, img_array in enumerate(images):
                    # Convertir en PIL Image
                    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(img_rgb)
                    
                    # OCR sur cette page
                    page_results = self.process_image_ocr(pil_image, f"{file_path.stem}_page_{page_num+1}")
                    all_texts.append(page_results['text'])
                    all_boxes.extend(page_results['boxes'])
                
                # Combiner les résultats
                combined_text = '\n\n'.join(all_texts)
                result['text'] = combined_text
                result['boxes'] = all_boxes
                
            else:
                # Fichier image standard
                image = Image.open(file_path)
                ocr_results = self.process_image_ocr(image, file_path.stem)
                result['text'] = ocr_results['text']
                result['boxes'] = ocr_results['boxes']
            
            # Statistiques
            result['text_length'] = len(result['text'])
            result['word_count'] = len(result['text'].split())
            result['detection_count'] = len(result['boxes'])
            result['avg_confidence'] = np.mean([b['confidence'] for b in result['boxes']]) if result['boxes'] else 0
            result['languages_used'] = self.languages
            
            # Sauvegarder les résultats
            output_files = self.save_results(file_path, result)
            result['output_files'] = output_files
            
            result['success'] = True
            
            # Marquer comme traité
            file_hash = self.get_file_hash(file_path)
            self.processed_hashes.add(file_hash)
            
        except Exception as e:
            result['error'] = str(e)
            print(f"❌ Erreur {file_path.name}: {e}")
        
        result['processing_time'] = time.time() - start_time
        
        return result
    
    def process_image_ocr(self, image: Image.Image, base_name: str) -> Dict:
        """Effectuer l'OCR sur une image"""
        results = {
            'text': '',
            'boxes': []
        }
        
        # Préprocessing
        processed_image = self.ocr.preprocess_image(image, enhancement_level='medium')
        
        # OCR pour chaque langue
        all_texts = []
        all_boxes = []
        
        for lang in self.languages:
            try:
                # Texte simple
                text = self.ocr.extract_text_simple(
                    processed_image,
                    lang=lang,
                    config=self.config
                )
                
                if text.strip():
                    all_texts.append(f"[{lang.upper()}] {text}")
                
                # Boîtes de détection
                boxes = self.ocr.extract_text_with_boxes(
                    processed_image,
                    lang=lang,
                    confidence_threshold=30
                )
                
                # Ajouter la langue aux boîtes
                for box in boxes:
                    box['language'] = lang
                
                all_boxes.extend(boxes)
                
            except Exception as e:
                print(f"⚠️  Erreur OCR {lang} pour {base_name}: {e}")
        
        # Combiner les résultats
        results['text'] = '\n\n'.join(all_texts) if all_texts else 'Aucun texte détecté'
        results['boxes'] = all_boxes
        
        return results
    
    def save_results(self, original_file: Path, result: Dict) -> List[str]:
        """Sauvegarder les résultats d'un fichier"""
        output_files = []
        
        try:
            base_name = original_file.stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 1. Fichier texte
            text_file = self.output_folder / f"{base_name}_text.txt"
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(f"OCR Results - {original_file.name}\n")
                f.write(f"Processed: {timestamp}\n")
                f.write(f"Languages: {', '.join(self.languages)}\n")
                f.write(f"Processing time: {result['processing_time']:.2f}s\n")
                f.write(f"File size: {result['file_size']} bytes\n")
                f.write(f"Word count: {result['word_count']}\n")
                f.write(f"Detection count: {result['detection_count']}\n")
                f.write(f"Average confidence: {result['avg_confidence']:.1f}%\n")
                f.write("\n" + "="*50 + "\n")
                f.write("EXTRACTED TEXT:\n")
                f.write("="*50 + "\n\n")
                f.write(result['text'])
            
            output_files.append(str(text_file))
            
            # 2. Fichier JSON détaillé
            json_file = self.output_folder / f"{base_name}_details.json"
            json_data = {
                'original_file': str(original_file),
                'processing_date': timestamp,
                'processing_time': result['processing_time'],
                'file_info': {
                    'name': original_file.name,
                    'size': result['file_size'],
                    'extension': original_file.suffix
                },
                'ocr_config': {
                    'languages': self.languages,
                    'tesseract_config': self.config
                },
                'results': {
                    'text': result['text'],
                    'statistics': {
                        'text_length': result['text_length'],
                        'word_count': result['word_count'],
                        'detection_count': result['detection_count'],
                        'avg_confidence': result['avg_confidence']
                    },
                    'detections': result['boxes'][:50]  # Limiter pour la taille du fichier
                }
            }
            
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            output_files.append(str(json_file))
            
            # 3. CSV simple pour import
            csv_file = self.output_folder / f"{base_name}_simple.csv"
            with open(csv_file, 'w', encoding='utf-8') as f:
                f.write("file_name,processing_time,word_count,avg_confidence,text_preview\n")
                text_preview = result['text'].replace('\n', ' ')[:200]
                f.write(f'"{original_file.name}",{result["processing_time"]:.2f},{result["word_count"]},{result["avg_confidence"]:.1f},"{text_preview}"\n')
            
            output_files.append(str(csv_file))
            
        except Exception as e:
            print(f"⚠️  Erreur sauvegarde pour {original_file.name}: {e}")
        
        return output_files
    
    def generate_report(self, results: List[Dict]):
        """Générer un rapport global"""
        
        # Calculs statistiques
        successful_results = [r for r in results if r['success']]
        failed_results = [r for r in results if not r['success']]
        
        total_processing_time = sum(r['processing_time'] for r in results)
        total_text_length = sum(r['text_length'] for r in successful_results)
        total_words = sum(r['word_count'] for r in successful_results)
        
        avg_confidence = np.mean([r['avg_confidence'] for r in successful_results]) if successful_results else 0
        avg_processing_time = np.mean([r['processing_time'] for r in successful_results]) if successful_results else 0
        
        # Rapport détaillé
        report_file = self.output_folder / "batch_ocr_report.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("RAPPORT DE TRAITEMENT OCR PAR LOTS\n")
            f.write("=" * 50 + "\n\n")
            
            # Résumé général
            f.write("RÉSUMÉ GÉNÉRAL\n")
            f.write("-" * 20 + "\n")
            f.write(f"Dossier traité: {self.input_folder}\n")
            f.write(f"Date de traitement: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Langues OCR: {', '.join(self.languages)}\n")
            f.write(f"Configuration: {self.config}\n")
            f.write(f"Workers: {self.max_workers}\n\n")
            
            # Statistiques
            f.write("STATISTIQUES\n")
            f.write("-" * 20 + "\n")
            f.write(f"Fichiers totaux: {len(results)}\n")
            f.write(f"Fichiers réussis: {len(successful_results)}\n")
            f.write(f"Fichiers échoués: {len(failed_results)}\n")
            f.write(f"Taux de réussite: {len(successful_results)/len(results)*100:.1f}%\n")
            f.write(f"Temps total: {total_processing_time:.2f}s\n")
            f.write(f"Temps moyen/fichier: {avg_processing_time:.2f}s\n")
            f.write(f"Texte total extrait: {total_text_length} caractères\n")
            f.write(f"Mots totaux: {total_words}\n")
            f.write(f"Confiance moyenne: {avg_confidence:.1f}%\n\n")
            
            # Top 10 des fichiers les plus longs à traiter
            f.write("FICHIERS LES PLUS LONGS À TRAITER\n")
            f.write("-" * 40 + "\n")
            sorted_by_time = sorted(successful_results, key=lambda x: x['processing_time'], reverse=True)
            for i, result in enumerate(sorted_by_time[:10], 1):
                f.write(f"{i:2d}. {result['file_name']:30s} {result['processing_time']:6.2f}s\n")
            
            f.write(f"\n")
            
            # Fichiers avec le plus de texte
            f.write("FICHIERS AVEC LE PLUS DE TEXTE\n")
            f.write("-" * 35 + "\n")
            sorted_by_text = sorted(successful_results, key=lambda x: x['word_count'], reverse=True)
            for i, result in enumerate(sorted_by_text[:10], 1):
                f.write(f"{i:2d}. {result['file_name']:30s} {result['word_count']:6d} mots\n")
            
            f.write(f"\n")
            
            # Erreurs
            if failed_results:
                f.write("FICHIERS EN ERREUR\n")
                f.write("-" * 20 + "\n")
                for result in failed_results:
                    f.write(f"• {result['file_name']}: {result['error']}\n")
            
        print(f"📊 Rapport généré: {report_file}")
        
        # Rapport CSV pour analyse
        csv_report = self.output_folder / "batch_ocr_summary.csv"
        with open(csv_report, 'w', encoding='utf-8') as f:
            f.write("file_name,success,processing_time,file_size,word_count,text_length,avg_confidence,error\n")
            
            for result in results:
                error_msg = result.get('error', '').replace(',', ';') if result.get('error') else ''
                f.write(f'"{result["file_name"]}",{result["success"]},{result["processing_time"]:.2f},'
                       f'{result["file_size"]},{result["word_count"]},{result["text_length"]},'
                       f'{result["avg_confidence"]:.1f},"{error_msg}"\n')
        
        print(f"📊 Résumé CSV: {csv_report}")
    
    def process_batch(self):
        """Traiter tous les fichiers par lots"""
        
        print(f"🚀 DÉMARRAGE DU TRAITEMENT PAR LOTS")
        print(f"=" * 50)
        
        self.stats['start_time'] = time.time()
        
        # Trouver les fichiers
        files_to_process = self.find_files()
        
        if not files_to_process:
            print(f"✅ Aucun nouveau fichier à traiter")
            return
        
        print(f"📋 {len(files_to_process)} fichiers à traiter")
        print(f"🔧 {self.max_workers} workers simultanés")
        
        results = []
        
        try:
            # Traitement en parallèle
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Soumettre tous les fichiers
                future_to_file = {
                    executor.submit(self.process_single_file, file_path): file_path 
                    for file_path in files_to_process
                }
                
                # Traiter les résultats au fur et à mesure
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    
                    try:
                        result = future.result()
                        results.append(result)
                        
                        self.stats['processed_files'] += 1
                        
                        if result['success']:
                            self.stats['successful_files'] += 1
                            self.stats['total_text_length'] += result['text_length']
                            print(f"✅ {result['file_name']} - {result['word_count']} mots, {result['processing_time']:.2f}s")
                        else:
                            self.stats['failed_files'] += 1
                            print(f"❌ {result['file_name']} - {result['error']}")
                        
                        # Sauvegarder la progression périodiquement
                        if self.stats['processed_files'] % 5 == 0:
                            self.save_progress()
                            
                            # Afficher le progrès
                            progress = self.stats['processed_files'] / len(files_to_process) * 100
                            print(f"📊 Progrès: {self.stats['processed_files']}/{len(files_to_process)} ({progress:.1f}%)")
                        
                    except Exception as e:
                        print(f"❌ Erreur inattendue pour {file_path.name}: {e}")
                        results.append({
                            'file_path': str(file_path),
                            'file_name': file_path.name,
                            'success': False,
                            'error': f"Erreur inattendue: {e}",
                            'processing_time': 0,
                            'file_size': 0,
                            'text_length': 0,
                            'word_count': 0,
                            'detection_count': 0,
                            'avg_confidence': 0
                        })
                        self.stats['failed_files'] += 1
            
        except KeyboardInterrupt:
            print(f"\n🛑 Traitement interrompu par l'utilisateur")
            
        finally:
            self.stats['end_time'] = time.time()
            self.stats['total_processing_time'] = self.stats['end_time'] - self.stats['start_time']
            
            # Sauvegarder la progression finale
            self.save_progress()
            
            # Générer le rapport
            if results:
                self.generate_report(results)
            
            # Afficher le résumé final
            self.print_final_summary()
    
    def print_final_summary(self):
        """Afficher le résumé final"""
        
        print(f"\n🎉 TRAITEMENT TERMINÉ")
        print(f"=" * 50)
        print(f"📄 Fichiers traités: {self.stats['processed_files']}")
        print(f"✅ Réussites: {self.stats['successful_files']}")
        print(f"❌ Échecs: {self.stats['failed_files']}")
        print(f"⏭️  Ignorés: {self.stats['skipped_files']}")
        
        if self.stats['processed_files'] > 0:
            success_rate = self.stats['successful_files'] / self.stats['processed_files'] * 100
            print(f"📊 Taux de réussite: {success_rate:.1f}%")
        
        print(f"⏱️  Temps total: {self.stats['total_processing_time']:.2f}s")
        
        if self.stats['successful_files'] > 0:
            avg_time = self.stats['total_processing_time'] / self.stats['successful_files']
            print(f"⏱️  Temps moyen/fichier: {avg_time:.2f}s")
        
        print(f"📝 Texte total extrait: {self.stats['total_text_length']} caractères")
        print(f"📁 Résultats dans: {self.output_folder}")


def main():
    """Fonction principale avec arguments de ligne de commande"""
    
    parser = argparse.ArgumentParser(description="Traitement OCR par lots")
    parser.add_argument("input_folder", help="Dossier contenant les images à traiter")
    parser.add_argument("-o", "--output", help="Dossier de sortie")
    parser.add_argument("-w", "--workers", type=int, default=2, help="Nombre de workers (défaut: 2)")
    parser.add_argument("-l", "--languages", nargs='+', default=['eng'], help="Langues OCR (défaut: eng)")
    parser.add_argument("-c", "--config", default='--oem 3 --psm 6', help="Configuration Tesseract")
    parser.add_argument("--resume", action='store_true', help="Reprendre un traitement interrompu")
    
    args = parser.parse_args()
    
    print(f"📋 TRAITEMENT OCR PAR LOTS - SNAPDRAGON X PLUS")
    print(f"=" * 50)
    print(f"📁 Dossier d'entrée: {args.input_folder}")
    print(f"🔧 Workers: {args.workers}")
    print(f"🌍 Langues: {args.languages}")
    
    try:
        processor = BatchOCRProcessor(
            input_folder=args.input_folder,
            output_folder=args.output,
            max_workers=args.workers,
            languages=args.languages,
            config=args.config
        )
        
        processor.process_batch()
        
    except KeyboardInterrupt:
        print(f"\n🛑 Processus interrompu")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Mode interactif si aucun argument
        print("📋 Mode interactif")
        input_folder = input("Dossier d'entrée: ").strip()
        
        if not input_folder:
            print("❌ Dossier requis")
            sys.exit(1)
        
        try:
            processor = BatchOCRProcessor(input_folder)
            processor.process_batch()
        except Exception as e:
            print(f"❌ Erreur: {e}")
    else:
        main()