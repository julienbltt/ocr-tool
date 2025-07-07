#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostic et Optimisation OCR
Analyse le système et optimise automatiquement les paramètres OCR

Compatible Snapdragon X Plus - Windows ARM64
Utilise la classe TesseractOCR

Fonctionnalités:
- Diagnostic complet du système
- Test de performance automatique
- Optimisation des paramètres
- Recommandations personnalisées
- Maintenance et nettoyage

Auteur: Assistant IA
Date: 2025
"""

import os
import sys
import time
import json
import platform
import psutil
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

# Import de la classe TesseractOCR
try:
    from tesseract_complete_ocr import TesseractOCR
except ImportError:
    print("❌ Impossible d'importer TesseractOCR")
    print("   Assurez-vous que tesseract_complete_ocr.py est dans le même répertoire")
    sys.exit(1)

class OCRDiagnosticOptimizer:
    """Diagnostic et optimisation OCR pour Snapdragon X Plus"""
    
    def __init__(self, output_dir: str = "ocr_diagnostic"):
        """Initialiser le diagnostic"""
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.system_info = {}
        self.ocr_info = {}
        self.performance_results = {}
        self.recommendations = []
        
        print(f"🔍 Diagnostic OCR initialisé")
        print(f"📁 Résultats dans: {self.output_dir}")
    
    def get_system_info(self) -> Dict:
        """Collecter les informations système détaillées"""
        
        print("🖥️  Collecte des informations système...")
        
        info = {
            'timestamp': datetime.now().isoformat(),
            'platform': {
                'system': platform.system(),
                'release': platform.release(),
                'version': platform.version(),
                'machine': platform.machine(),
                'processor': platform.processor(),
                'architecture': platform.architecture(),
                'python_version': platform.python_version()
            },
            'hardware': {
                'cpu_count': psutil.cpu_count(logical=False),
                'cpu_count_logical': psutil.cpu_count(logical=True),
                'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
                'memory_available_gb': round(psutil.virtual_memory().available / (1024**3), 2),
                'memory_percent': psutil.virtual_memory().percent
            },
            'environment': {
                'temp_folder': os.environ.get('TEMP', 'N/A'),
                'user_profile': os.environ.get('USERPROFILE', 'N/A'),
                'path_env': len(os.environ.get('PATH', '').split(';')),
                'omp_threads': os.environ.get('OMP_NUM_THREADS', 'auto'),
                'mkl_threads': os.environ.get('MKL_NUM_THREADS', 'auto')
            }
        }
        
        # Vérifications spécifiques Snapdragon X Plus
        is_arm = platform.machine().lower() in ['arm64', 'aarch64', 'arm']
        is_snapdragon = 'snapdragon' in platform.processor().lower()
        
        info['snapdragon_detection'] = {
            'is_arm_architecture': is_arm,
            'is_snapdragon_processor': is_snapdragon,
            'likely_snapdragon_x': is_arm and 'windows' in platform.system().lower()
        }
        
        # Test de performance CPU basique
        start_time = time.time()
        
        # Test calcul simple
        test_array = np.random.rand(1000, 1000)
        result = np.dot(test_array, test_array.T)
        
        cpu_test_time = time.time() - start_time
        info['cpu_performance'] = {
            'matrix_multiply_1000x1000_time': cpu_test_time,
            'estimated_performance': 'fast' if cpu_test_time < 1.0 else 'moderate' if cpu_test_time < 3.0 else 'slow'
        }
        
        self.system_info = info
        return info
    
    def test_tesseract_installation(self) -> Dict:
        """Tester l'installation Tesseract"""
        
        print("🔧 Test de l'installation Tesseract...")
        
        ocr_info = {
            'installation_status': 'unknown',
            'tesseract_found': False,
            'tesseract_version': None,
            'tesseract_path': None,
            'languages': [],
            'ocr_class_working': False,
            'basic_ocr_test': False,
            'errors': []
        }
        
        try:
            # Test 1: Import de la classe OCR
            ocr = TesseractOCR()
            ocr_info['ocr_class_working'] = True
            ocr_info['tesseract_found'] = True
            ocr_info['languages'] = ocr.available_languages
            
            # Test 2: Version Tesseract
            try:
                import pytesseract
                version = pytesseract.get_tesseract_version()
                ocr_info['tesseract_version'] = str(version)
                ocr_info['tesseract_path'] = pytesseract.pytesseract.tesseract_cmd
            except Exception as e:
                ocr_info['errors'].append(f"Version check failed: {e}")
            
            # Test 3: OCR basique
            try:
                test_image = Image.new('RGB', (200, 50), 'white')
                draw = ImageDraw.Draw(test_image)
                draw.text((10, 15), "TEST", fill='black')
                
                text = ocr.extract_text_simple(test_image)
                
                if 'test' in text.lower():
                    ocr_info['basic_ocr_test'] = True
                    ocr_info['installation_status'] = 'working'
                else:
                    ocr_info['installation_status'] = 'partial'
                    ocr_info['errors'].append(f"OCR returned: '{text}' instead of 'TEST'")
                    
            except Exception as e:
                ocr_info['errors'].append(f"Basic OCR test failed: {e}")
                ocr_info['installation_status'] = 'broken'
            
        except Exception as e:
            ocr_info['errors'].append(f"TesseractOCR class import failed: {e}")
            ocr_info['installation_status'] = 'not_found'
        
        self.ocr_info = ocr_info
        return ocr_info
    
    def create_test_images(self) -> Dict[str, Path]:
        """Créer des images de test pour le benchmark"""
        
        print("🖼️  Création des images de test...")
        
        test_images = {}
        
        # 1. Texte simple haute qualité
        img1 = Image.new('RGB', (400, 100), 'white')
        draw1 = ImageDraw.Draw(img1)
        try:
            font1 = ImageFont.truetype("arial.ttf", 24)
        except:
            font1 = ImageFont.load_default()
        
        draw1.text((20, 30), "Simple Quality Text 123", fill='black', font=font1)
        
        path1 = self.output_dir / "test_simple.png"
        img1.save(path1)
        test_images['simple'] = path1
        
        # 2. Texte complexe
        img2 = Image.new('RGB', (600, 200), 'white')
        draw2 = ImageDraw.Draw(img2)
        try:
            font2 = ImageFont.truetype("arial.ttf", 16)
        except:
            font2 = ImageFont.load_default()
        
        complex_text = [
            "Complex Document Analysis",
            "Email: test@example.com",
            "Phone: +33 1 23 45 67 89",
            "Date: 07/01/2025 - Amount: €1,234.56"
        ]
        
        y_pos = 20
        for line in complex_text:
            draw2.text((20, y_pos), line, fill='black', font=font2)
            y_pos += 30
        
        path2 = self.output_dir / "test_complex.png"
        img2.save(path2)
        test_images['complex'] = path2
        
        # 3. Texte avec bruit
        img3 = Image.new('RGB', (400, 100), 'white')
        draw3 = ImageDraw.Draw(img3)
        draw3.text((20, 30), "Noisy Text ABC123", fill='black', font=font1)
        
        # Ajouter du bruit
        img3_array = np.array(img3)
        noise = np.random.normal(0, 15, img3_array.shape).astype(np.int16)
        img3_noisy = np.clip(img3_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        img3_final = Image.fromarray(img3_noisy)
        
        path3 = self.output_dir / "test_noisy.png"
        img3_final.save(path3)
        test_images['noisy'] = path3
        
        print(f"✅ {len(test_images)} images de test créées")
        return test_images
    
    def benchmark_configurations(self, test_images: Dict[str, Path]) -> Dict:
        """Benchmark des différentes configurations"""
        
        print("⚡ Benchmark des configurations OCR...")
        
        if not self.ocr_info.get('ocr_class_working', False):
            print("❌ OCR non fonctionnel, benchmark impossible")
            return {}
        
        ocr = TesseractOCR()
        
        # Configurations à tester
        configurations = {
            'default': {
                'config': '--oem 3 --psm 6',
                'preprocessing': 'medium',
                'description': 'Configuration par défaut'
            },
            'fast': {
                'config': '--oem 3 --psm 8',
                'preprocessing': 'light',
                'description': 'Configuration rapide'
            },
            'accurate': {
                'config': '--oem 3 --psm 3',
                'preprocessing': 'strong',
                'description': 'Configuration précise'
            },
            'line': {
                'config': '--oem 3 --psm 7',
                'preprocessing': 'medium',
                'description': 'Ligne unique'
            }
        }
        
        results = {}
        
        for config_name, config_data in configurations.items():
            config_results = {}
            
            for image_name, image_path in test_images.items():
                print(f"  🧪 Test {config_name} sur {image_name}...")
                
                try:
                    # Charger et préprocesser l'image
                    image = Image.open(image_path)
                    
                    if config_data['preprocessing'] != 'none':
                        processed_image = ocr.preprocess_image(
                            image, 
                            enhancement_level=config_data['preprocessing']
                        )
                    else:
                        processed_image = image
                    
                    # Mesurer le temps d'OCR
                    start_time = time.time()
                    
                    text = ocr.extract_text_simple(
                        processed_image,
                        lang='eng',
                        config=config_data['config']
                    )
                    
                    boxes = ocr.extract_text_with_boxes(
                        processed_image,
                        lang='eng',
                        confidence_threshold=30
                    )
                    
                    processing_time = time.time() - start_time
                    
                    # Analyser les résultats
                    word_count = len(text.split()) if text else 0
                    avg_confidence = np.mean([b['confidence'] for b in boxes]) if boxes else 0
                    
                    config_results[image_name] = {
                        'processing_time': processing_time,
                        'word_count': word_count,
                        'text_length': len(text),
                        'detection_count': len(boxes),
                        'avg_confidence': avg_confidence,
                        'text_preview': text[:50],
                        'success': True
                    }
                    
                except Exception as e:
                    config_results[image_name] = {
                        'processing_time': 0,
                        'word_count': 0,
                        'text_length': 0,
                        'detection_count': 0,
                        'avg_confidence': 0,
                        'error': str(e),
                        'success': False
                    }
            
            results[config_name] = {
                'config_data': config_data,
                'image_results': config_results,
                'summary': self.calculate_config_summary(config_results)
            }
        
        self.performance_results = results
        return results
    
    def calculate_config_summary(self, config_results: Dict) -> Dict:
        """Calculer le résumé d'une configuration"""
        
        successful_tests = {k: v for k, v in config_results.items() if v.get('success', False)}
        
        if not successful_tests:
            return {
                'success_rate': 0,
                'avg_processing_time': 0,
                'avg_word_count': 0,
                'avg_confidence': 0,
                'performance_score': 0
            }
        
        avg_time = np.mean([r['processing_time'] for r in successful_tests.values()])
        avg_words = np.mean([r['word_count'] for r in successful_tests.values()])
        avg_confidence = np.mean([r['avg_confidence'] for r in successful_tests.values()])
        
        # Score de performance (combinaison vitesse/précision)
        speed_score = max(0, (2.0 - avg_time) / 2.0)  # Meilleur si < 2s
        accuracy_score = avg_confidence / 100.0
        word_score = min(avg_words / 10.0, 1.0)  # Meilleur si beaucoup de mots
        
        performance_score = (speed_score * 0.3 + accuracy_score * 0.5 + word_score * 0.2)
        
        return {
            'success_rate': len(successful_tests) / len(config_results),
            'avg_processing_time': avg_time,
            'avg_word_count': avg_words,
            'avg_confidence': avg_confidence,
            'performance_score': performance_score
        }
    
    def analyze_system_bottlenecks(self) -> List[str]:
        """Analyser les goulots d'étranglement du système"""
        
        print("🔍 Analyse des goulots d'étranglement...")
        
        bottlenecks = []
        
        # 1. Mémoire
        memory_percent = self.system_info['hardware']['memory_percent']
        if memory_percent > 80:
            bottlenecks.append(f"Mémoire RAM très utilisée ({memory_percent:.1f}%)")
        elif memory_percent > 60:
            bottlenecks.append(f"Mémoire RAM modérément utilisée ({memory_percent:.1f}%)")
        
        # 2. CPU
        cpu_perf = self.system_info['cpu_performance']['estimated_performance']
        if cpu_perf == 'slow':
            bottlenecks.append("Performance CPU lente détectée")
        
        # 3. Architecture ARM
        if self.system_info['snapdragon_detection']['is_arm_architecture']:
            bottlenecks.append("Architecture ARM détectée - optimisations spéciales recommandées")
        
        # 4. Threads
        omp_threads = self.system_info['environment']['omp_threads']
        if omp_threads == 'auto':
            cpu_count = self.system_info['hardware']['cpu_count_logical']
            if cpu_count > 8:
                bottlenecks.append(f"Trop de threads possibles ({cpu_count}) - limitation recommandée")
        
        # 5. Performance OCR
        if self.performance_results:
            avg_times = []
            for config_data in self.performance_results.values():
                summary = config_data['summary']
                if summary['avg_processing_time'] > 0:
                    avg_times.append(summary['avg_processing_time'])
            
            if avg_times:
                overall_avg_time = np.mean(avg_times)
                if overall_avg_time > 3.0:
                    bottlenecks.append(f"OCR lent (moyenne {overall_avg_time:.2f}s)")
        
        return bottlenecks
    
    def generate_recommendations(self) -> List[Dict]:
        """Générer des recommandations personnalisées"""
        
        print("💡 Génération des recommandations...")
        
        recommendations = []
        
        # 1. Recommandations système
        if self.system_info['snapdragon_detection']['is_arm_architecture']:
            recommendations.append({
                'category': 'système',
                'priority': 'haute',
                'title': 'Optimisation pour ARM',
                'description': 'Limitez les threads pour améliorer les performances sur Snapdragon X Plus',
                'actions': [
                    'set OMP_NUM_THREADS=4',
                    'set MKL_NUM_THREADS=4',
                    'Utiliser max 2-4 workers pour traitement parallèle'
                ]
            })
        
        memory_percent = self.system_info['hardware']['memory_percent']
        if memory_percent > 70:
            recommendations.append({
                'category': 'mémoire',
                'priority': 'moyenne',
                'title': 'Optimisation mémoire',
                'description': f'Utilisation mémoire élevée ({memory_percent:.1f}%)',
                'actions': [
                    'Fermer les applications non nécessaires',
                    'Traiter les images par plus petits lots',
                    'Réduire la résolution des images avant OCR'
                ]
            })
        
        # 2. Recommandations OCR
        if self.performance_results:
            # Trouver la meilleure configuration
            best_config = None
            best_score = 0
            
            for config_name, config_data in self.performance_results.items():
                score = config_data['summary']['performance_score']
                if score > best_score:
                    best_score = score
                    best_config = config_name
            
            if best_config:
                config_data = self.performance_results[best_config]
                recommendations.append({
                    'category': 'ocr',
                    'priority': 'haute',
                    'title': f'Configuration OCR optimale: {best_config}',
                    'description': f"Score de performance: {best_score:.2f}",
                    'actions': [
                        f"Utiliser config: {config_data['config_data']['config']}",
                        f"Préprocessing: {config_data['config_data']['preprocessing']}",
                        f"Temps moyen: {config_data['summary']['avg_processing_time']:.2f}s"
                    ]
                })
            
            # Recommandations par type d'image
            for config_name, config_data in self.performance_results.items():
                image_results = config_data['image_results']
                
                # Trouver les points forts de chaque config
                if 'simple' in image_results and image_results['simple'].get('success'):
                    simple_time = image_results['simple']['processing_time']
                    if simple_time < 1.0:
                        recommendations.append({
                            'category': 'usage',
                            'priority': 'basse',
                            'title': f'Pour texte simple: {config_name}',
                            'description': f'Très rapide ({simple_time:.2f}s) pour texte de qualité',
                            'actions': [f"Config: {config_data['config_data']['config']}"]
                        })
        
        # 3. Recommandations maintenance
        if not self.ocr_info.get('basic_ocr_test', False):
            recommendations.append({
                'category': 'maintenance',
                'priority': 'critique',
                'title': 'Problème OCR détecté',
                'description': 'L\'OCR ne fonctionne pas correctement',
                'actions': [
                    'Vérifier l\'installation Tesseract',
                    'Vérifier les variables d\'environnement PATH',
                    'Réinstaller pytesseract'
                ]
            })
        
        # 4. Recommandations langues
        available_langs = self.ocr_info.get('languages', [])
        if 'fra' not in available_langs and len(available_langs) == 1:
            recommendations.append({
                'category': 'langues',
                'priority': 'basse',
                'title': 'Ajouter support français',
                'description': 'Seul l\'anglais est disponible',
                'actions': [
                    'Télécharger fra.traineddata',
                    'Copier dans tessdata/',
                    'Redémarrer l\'application'
                ]
            })
        
        self.recommendations = recommendations
        return recommendations
    
    def optimize_environment(self):
        """Optimiser automatiquement l'environnement"""
        
        print("⚙️  Optimisation automatique de l'environnement...")
        
        optimizations_applied = []
        
        # 1. Variables d'environnement pour ARM
        if self.system_info['snapdragon_detection']['is_arm_architecture']:
            try:
                os.environ['OMP_NUM_THREADS'] = '4'
                os.environ['MKL_NUM_THREADS'] = '4'
                os.environ['NUMEXPR_NUM_THREADS'] = '4'
                optimizations_applied.append("Variables d'environnement ARM configurées")
            except Exception as e:
                print(f"⚠️  Erreur config threads: {e}")
        
        # 2. Créer dossiers de travail optimaux
        temp_ocr_dir = Path(os.environ.get('TEMP', '.')) / 'ocr_temp'
        temp_ocr_dir.mkdir(exist_ok=True)
        optimizations_applied.append(f"Dossier temp OCR: {temp_ocr_dir}")
        
        # 3. Nettoyage des fichiers temporaires anciens
        try:
            import glob
            import time
            
            temp_files = glob.glob(str(temp_ocr_dir / '*'))
            now = time.time()
            
            cleaned_count = 0
            for temp_file in temp_files:
                file_path = Path(temp_file)
                if file_path.is_file():
                    # Supprimer fichiers > 24h
                    if now - file_path.stat().st_mtime > 24 * 3600:
                        file_path.unlink()
                        cleaned_count += 1
            
            if cleaned_count > 0:
                optimizations_applied.append(f"Nettoyés {cleaned_count} fichiers temporaires")
                
        except Exception as e:
            print(f"⚠️  Erreur nettoyage: {e}")
        
        print(f"✅ {len(optimizations_applied)} optimisations appliquées:")
        for opt in optimizations_applied:
            print(f"   • {opt}")
        
        return optimizations_applied
    
    def generate_report(self):
        """Générer un rapport de diagnostic complet"""
        
        print("📊 Génération du rapport de diagnostic...")
        
        report = {
            'diagnostic_info': {
                'timestamp': datetime.now().isoformat(),
                'version': '1.0',
                'system': 'snapdragon_x_plus'
            },
            'system_info': self.system_info,
            'ocr_info': self.ocr_info,
            'performance_results': self.performance_results,
            'bottlenecks': self.analyze_system_bottlenecks(),
            'recommendations': self.recommendations
        }
        
        # Rapport JSON détaillé
        json_report = self.output_dir / "diagnostic_report.json"
        with open(json_report, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Rapport texte lisible
        text_report = self.output_dir / "diagnostic_report.txt"
        with open(text_report, 'w', encoding='utf-8') as f:
            f.write("RAPPORT DE DIAGNOSTIC OCR - SNAPDRAGON X PLUS\n")
            f.write("=" * 60 + "\n\n")
            
            # Résumé exécutif
            f.write("RÉSUMÉ EXÉCUTIF\n")
            f.write("-" * 20 + "\n")
            f.write(f"Date du diagnostic: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Système: {self.system_info['platform']['system']} {self.system_info['platform']['machine']}\n")
            f.write(f"Processeur: {self.system_info['platform']['processor']}\n")
            f.write(f"Mémoire: {self.system_info['hardware']['memory_total_gb']} GB\n")
            f.write(f"OCR Status: {self.ocr_info.get('installation_status', 'unknown')}\n\n")
            
            # État OCR
            f.write("ÉTAT DE L'INSTALLATION OCR\n")
            f.write("-" * 30 + "\n")
            f.write(f"Tesseract trouvé: {'✅' if self.ocr_info.get('tesseract_found') else '❌'}\n")
            f.write(f"Version: {self.ocr_info.get('tesseract_version', 'Inconnue')}\n")
            f.write(f"Langues: {', '.join(self.ocr_info.get('languages', []))}\n")
            f.write(f"Test basique: {'✅' if self.ocr_info.get('basic_ocr_test') else '❌'}\n")
            
            if self.ocr_info.get('errors'):
                f.write("Erreurs détectées:\n")
                for error in self.ocr_info['errors']:
                    f.write(f"  • {error}\n")
            f.write("\n")
            
            # Performance
            if self.performance_results:
                f.write("RÉSULTATS DE PERFORMANCE\n")
                f.write("-" * 25 + "\n")
                
                for config_name, config_data in self.performance_results.items():
                    summary = config_data['summary']
                    f.write(f"\nConfiguration: {config_name}\n")
                    f.write(f"  Score: {summary['performance_score']:.2f}\n")
                    f.write(f"  Temps moyen: {summary['avg_processing_time']:.2f}s\n")
                    f.write(f"  Confiance: {summary['avg_confidence']:.1f}%\n")
                    f.write(f"  Taux de réussite: {summary['success_rate']:.1f}%\n")
                
                f.write("\n")
            
            # Goulots d'étranglement
            bottlenecks = report['bottlenecks']
            if bottlenecks:
                f.write("GOULOTS D'ÉTRANGLEMENT DÉTECTÉS\n")
                f.write("-" * 35 + "\n")
                for bottleneck in bottlenecks:
                    f.write(f"⚠️  {bottleneck}\n")
                f.write("\n")
            
            # Recommandations
            f.write("RECOMMANDATIONS\n")
            f.write("-" * 15 + "\n")
            
            # Grouper par priorité
            for priority in ['critique', 'haute', 'moyenne', 'basse']:
                priority_recs = [r for r in self.recommendations if r['priority'] == priority]
                if priority_recs:
                    f.write(f"\nPriorité {priority.upper()}:\n")
                    for rec in priority_recs:
                        f.write(f"  📌 {rec['title']}\n")
                        f.write(f"     {rec['description']}\n")
                        for action in rec['actions']:
                            f.write(f"     • {action}\n")
        
        print(f"📄 Rapport JSON: {json_report}")
        print(f"📄 Rapport texte: {text_report}")
        
        return json_report, text_report
    
    def run_full_diagnostic(self):
        """Exécuter le diagnostic complet"""
        
        print(f"🚀 DIAGNOSTIC COMPLET OCR - SNAPDRAGON X PLUS")
        print(f"=" * 60)
        
        start_time = time.time()
        
        try:
            # 1. Informations système
            self.get_system_info()
            
            # 2. Test Tesseract
            self.test_tesseract_installation()
            
            # 3. Benchmark performance (si OCR fonctionne)
            if self.ocr_info.get('ocr_class_working', False):
                test_images = self.create_test_images()
                self.benchmark_configurations(test_images)
            else:
                print("⚠️  OCR non fonctionnel, benchmark ignoré")
            
            # 4. Analyse et recommandations
            self.generate_recommendations()
            
            # 5. Optimisation automatique
            self.optimize_environment()
            
            # 6. Rapport final
            json_report, text_report = self.generate_report()
            
            end_time = time.time()
            
            print(f"\n🎉 DIAGNOSTIC TERMINÉ")
            print(f"⏱️  Temps total: {end_time - start_time:.2f} secondes")
            print(f"📊 {len(self.recommendations)} recommandations générées")
            print(f"📁 Rapports dans: {self.output_dir}")
            
            # Afficher le résumé des recommandations critiques
            critical_recs = [r for r in self.recommendations if r['priority'] == 'critique']
            if critical_recs:
                print(f"\n🚨 ACTIONS CRITIQUES REQUISES:")
                for rec in critical_recs:
                    print(f"   • {rec['title']}")
            
        except Exception as e:
            print(f"❌ Erreur durant le diagnostic: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Fonction principale"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Diagnostic et optimisation OCR")
    parser.add_argument("-o", "--output", default="ocr_diagnostic", help="Dossier de sortie")
    parser.add_argument("--optimize", action='store_true', help="Appliquer les optimisations automatiquement")
    
    args = parser.parse_args()
    
    try:
        diagnostic = OCRDiagnosticOptimizer(args.output)
        diagnostic.run_full_diagnostic()
        
    except KeyboardInterrupt:
        print(f"\n🛑 Diagnostic interrompu")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()