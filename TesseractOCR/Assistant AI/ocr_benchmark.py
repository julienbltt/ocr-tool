#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark et Test de Performance OCR
Teste différentes configurations et mesure les performances

Compatible Snapdragon X Plus - Windows ARM64
Utilise la classe TesseractOCR

Auteur: Assistant IA
Date: 2025
"""

import time
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import cv2
from datetime import datetime
import sys

# Import de la classe TesseractOCR
try:
    from tesseract_complete_ocr import TesseractOCR
except ImportError:
    print("❌ Impossible d'importer TesseractOCR")
    print("   Assurez-vous que tesseract_complete_ocr.py est dans le même répertoire")
    sys.exit(1)

class OCRBenchmark:
    """Benchmark complet pour OCR"""
    
    def __init__(self, output_dir="benchmark_results"):
        """Initialiser le benchmark"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.ocr = TesseractOCR()
        self.results = {
            'system_info': self.get_system_info(),
            'tests': [],
            'summary': {}
        }
        
        print(f"🧪 Benchmark OCR initialisé")
        print(f"📁 Résultats dans: {self.output_dir}")
    
    def get_system_info(self):
        """Récupérer les informations système"""
        import platform
        import psutil
        
        return {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'machine': platform.machine(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'tesseract_languages': self.ocr.available_languages,
            'timestamp': datetime.now().isoformat()
        }
    
    def create_test_images(self):
        """Créer des images de test standardisées"""
        test_images = {}
        
        print("🖼️  Création des images de test...")
        
        # 1. Texte simple haute qualité
        img1 = Image.new('RGB', (800, 200), 'white')
        draw1 = ImageDraw.Draw(img1)
        try:
            font1 = ImageFont.truetype("arial.ttf", 32)
        except:
            font1 = ImageFont.load_default()
        
        draw1.text((20, 50), "Hello World! This is a simple test.", fill='black', font=font1)
        draw1.text((20, 100), "Numbers: 123456789 Email: test@example.com", fill='black', font=font1)
        
        test_path1 = self.output_dir / "test_simple.png"
        img1.save(test_path1)
        test_images['simple'] = {
            'path': test_path1,
            'description': 'Texte simple, haute qualité',
            'expected_words': 11,
            'difficulty': 'facile'
        }
        
        # 2. Texte petit et dense
        img2 = Image.new('RGB', (800, 600), 'white')
        draw2 = ImageDraw.Draw(img2)
        try:
            font2 = ImageFont.truetype("arial.ttf", 16)
        except:
            font2 = ImageFont.load_default()
        
        dense_text = [
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
            "Sed do eiusmod tempor incididunt ut labore et dolore magna.",
            "Ut enim ad minim veniam, quis nostrud exercitation ullamco.",
            "Duis aute irure dolor in reprehenderit in voluptate velit.",
            "Excepteur sint occaecat cupidatat non proident, sunt in.",
            "Contact: +33 1 23 45 67 89 | email@company.com",
            "Date: 07/01/2025 | Reference: REF-2025-001",
            "Amount: €1,234.56 | Status: APPROVED"
        ]
        
        y_pos = 30
        for line in dense_text:
            draw2.text((20, y_pos), line, fill='black', font=font2)
            y_pos += 25
        
        test_path2 = self.output_dir / "test_dense.png"
        img2.save(test_path2)
        test_images['dense'] = {
            'path': test_path2,
            'description': 'Texte dense, police petite',
            'expected_words': 50,
            'difficulty': 'moyen'
        }
        
        # 3. Texte avec bruit et artefacts
        img3 = Image.new('RGB', (800, 400), 'white')
        draw3 = ImageDraw.Draw(img3)
        try:
            font3 = ImageFont.truetype("arial.ttf", 28)
        except:
            font3 = ImageFont.load_default()
        
        # Ajouter du texte
        draw3.text((50, 100), "Noisy Document Text", fill='black', font=font3)
        draw3.text((50, 150), "Phone: +33 1 23 45 67 89", fill='black', font=font3)
        draw3.text((50, 200), "Serial: ABC-123-XYZ", fill='black', font=font3)
        
        # Ajouter du bruit
        img3_array = np.array(img3)
        noise = np.random.normal(0, 25, img3_array.shape).astype(np.uint8)
        img3_noisy = Image.fromarray(np.clip(img3_array + noise, 0, 255).astype(np.uint8))
        
        # Ajouter des lignes parasites
        draw3_noisy = ImageDraw.Draw(img3_noisy)
        for i in range(10):
            x1, y1 = np.random.randint(0, 800, 2)
            x2, y2 = np.random.randint(0, 400, 2)
            draw3_noisy.line([(x1, y1), (x2, y2)], fill='gray', width=1)
        
        test_path3 = self.output_dir / "test_noisy.png"
        img3_noisy.save(test_path3)
        test_images['noisy'] = {
            'path': test_path3,
            'description': 'Texte avec bruit et artefacts',
            'expected_words': 8,
            'difficulty': 'difficile'
        }
        
        # 4. Texte multilingue
        img4 = Image.new('RGB', (800, 300), 'white')
        draw4 = ImageDraw.Draw(img4)
        try:
            font4 = ImageFont.truetype("arial.ttf", 24)
        except:
            font4 = ImageFont.load_default()
        
        multilingual_text = [
            "English: Hello World!",
            "French: Bonjour le monde!",
            "Spanish: ¡Hola Mundo!",
            "Numbers: 123,456.78 € - $987.65",
            "Date: 07/01/2025 - Time: 14:30:45"
        ]
        
        y_pos = 40
        for line in multilingual_text:
            draw4.text((30, y_pos), line, fill='black', font=font4)
            y_pos += 35
        
        test_path4 = self.output_dir / "test_multilingual.png"
        img4.save(test_path4)
        test_images['multilingual'] = {
            'path': test_path4,
            'description': 'Texte multilingue avec caractères spéciaux',
            'expected_words': 20,
            'difficulty': 'moyen'
        }
        
        print(f"✅ {len(test_images)} images de test créées")
        return test_images
    
    def benchmark_configurations(self, test_images):
        """Tester différentes configurations OCR"""
        
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
            'sparse': {
                'config': '--oem 3 --psm 11',
                'preprocessing': 'strong',
                'description': 'Texte sparse'
            },
            'line': {
                'config': '--oem 3 --psm 7',
                'preprocessing': 'medium',
                'description': 'Ligne unique'
            }
        }
        
        languages = ['eng']
        if 'fra' in self.ocr.available_languages:
            languages.append('fra')
            languages.append('eng+fra')
        
        print(f"🧪 Test de {len(configurations)} configurations x {len(languages)} langues x {len(test_images)} images")
        
        total_tests = len(configurations) * len(languages) * len(test_images)
        current_test = 0
        
        for config_name, config_data in configurations.items():
            for lang in languages:
                for image_name, image_data in test_images.items():
                    current_test += 1
                    print(f"📊 Test {current_test}/{total_tests}: {config_name}-{lang}-{image_name}")
                    
                    # Préparation de l'image
                    image = Image.open(image_data['path'])
                    
                    if config_data['preprocessing'] != 'none':
                        processed_image = self.ocr.preprocess_image(
                            image, 
                            enhancement_level=config_data['preprocessing']
                        )
                    else:
                        processed_image = image
                    
                    # Mesure du temps
                    start_time = time.time()
                    
                    try:
                        # OCR simple
                        text = self.ocr.extract_text_simple(
                            processed_image,
                            lang=lang,
                            config=config_data['config']
                        )
                        
                        # OCR avec boîtes
                        boxes = self.ocr.extract_text_with_boxes(
                            processed_image,
                            lang=lang,
                            confidence_threshold=30
                        )
                        
                        end_time = time.time()
                        processing_time = end_time - start_time
                        
                        # Analyser les résultats
                        word_count = len(text.split()) if text else 0
                        char_count = len(text) if text else 0
                        detection_count = len(boxes)
                        avg_confidence = np.mean([b['confidence'] for b in boxes]) if boxes else 0
                        
                        # Calculer la précision (estimation basée sur les mots attendus)
                        expected_words = image_data['expected_words']
                        word_accuracy = min(word_count / expected_words, 1.0) if expected_words > 0 else 0
                        
                        success = True
                        error_msg = None
                        
                    except Exception as e:
                        end_time = time.time()
                        processing_time = end_time - start_time
                        
                        text = ""
                        word_count = 0
                        char_count = 0
                        detection_count = 0
                        avg_confidence = 0
                        word_accuracy = 0
                        success = False
                        error_msg = str(e)
                    
                    # Enregistrer les résultats
                    test_result = {
                        'configuration': config_name,
                        'language': lang,
                        'image': image_name,
                        'image_difficulty': image_data['difficulty'],
                        'processing_time': processing_time,
                        'word_count': word_count,
                        'char_count': char_count,
                        'detection_count': detection_count,
                        'avg_confidence': avg_confidence,
                        'word_accuracy': word_accuracy,
                        'expected_words': expected_words,
                        'success': success,
                        'error': error_msg,
                        'text_preview': text[:100] if text else "",
                        'config_description': config_data['description'],
                        'preprocessing': config_data['preprocessing']
                    }
                    
                    self.results['tests'].append(test_result)
                    
                    # Affichage des résultats
                    if success:
                        print(f"   ✅ {word_count} mots, {avg_confidence:.1f}% conf, {processing_time:.2f}s")
                    else:
                        print(f"   ❌ Erreur: {error_msg}")
        
        print(f"✅ Benchmark terminé: {len(self.results['tests'])} tests effectués")
    
    def analyze_results(self):
        """Analyser et résumer les résultats"""
        tests = self.results['tests']
        successful_tests = [t for t in tests if t['success']]
        
        if not successful_tests:
            print("❌ Aucun test réussi pour l'analyse")
            return
        
        print(f"📊 Analyse de {len(successful_tests)} tests réussis sur {len(tests)} total")
        
        # Analyse globale
        avg_time = np.mean([t['processing_time'] for t in successful_tests])
        avg_words = np.mean([t['word_count'] for t in successful_tests])
        avg_confidence = np.mean([t['avg_confidence'] for t in successful_tests])
        avg_accuracy = np.mean([t['word_accuracy'] for t in successful_tests])
        
        # Analyse par configuration
        config_stats = {}
        for test in successful_tests:
            config = test['configuration']
            if config not in config_stats:
                config_stats[config] = []
            config_stats[config].append(test)
        
        # Analyse par langue
        lang_stats = {}
        for test in successful_tests:
            lang = test['language']
            if lang not in lang_stats:
                lang_stats[lang] = []
            lang_stats[lang].append(test)
        
        # Analyse par difficulté
        difficulty_stats = {}
        for test in successful_tests:
            diff = test['image_difficulty']
            if diff not in difficulty_stats:
                difficulty_stats[diff] = []
            difficulty_stats[diff].append(test)
        
        # Résumé
        summary = {
            'total_tests': len(tests),
            'successful_tests': len(successful_tests),
            'success_rate': len(successful_tests) / len(tests) * 100,
            'average_processing_time': avg_time,
            'average_words_detected': avg_words,
            'average_confidence': avg_confidence,
            'average_accuracy': avg_accuracy,
            'best_configuration': self.find_best_configuration(config_stats),
            'best_language': self.find_best_language(lang_stats),
            'performance_by_difficulty': self.analyze_by_difficulty(difficulty_stats),
            'recommendations': self.generate_recommendations(successful_tests)
        }
        
        self.results['summary'] = summary
        
        # Affichage du résumé
        print(f"\n📈 RÉSUMÉ DU BENCHMARK")
        print(f"=" * 50)
        print(f"Tests réussis: {summary['successful_tests']}/{summary['total_tests']} ({summary['success_rate']:.1f}%)")
        print(f"Temps moyen: {summary['average_processing_time']:.2f}s")
        print(f"Mots moyens: {summary['average_words_detected']:.1f}")
        print(f"Confiance moyenne: {summary['average_confidence']:.1f}%")
        print(f"Précision moyenne: {summary['average_accuracy']:.1f}%")
        print(f"Meilleure config: {summary['best_configuration']['name']} ({summary['best_configuration']['score']:.2f})")
        print(f"Meilleure langue: {summary['best_language']['name']} ({summary['best_language']['score']:.2f})")
    
    def find_best_configuration(self, config_stats):
        """Trouver la meilleure configuration"""
        best_config = None
        best_score = 0
        
        for config, tests in config_stats.items():
            # Score basé sur précision, confiance et vitesse
            avg_accuracy = np.mean([t['word_accuracy'] for t in tests])
            avg_confidence = np.mean([t['avg_confidence'] for t in tests])
            avg_time = np.mean([t['processing_time'] for t in tests])
            
            # Score pondéré (précision et confiance importantes, temps inversé)
            score = (avg_accuracy * 0.4 + (avg_confidence/100) * 0.4 + (1/avg_time) * 0.2)
            
            if score > best_score:
                best_score = score
                best_config = config
        
        return {
            'name': best_config,
            'score': best_score,
            'tests_count': len(config_stats[best_config])
        }
    
    def find_best_language(self, lang_stats):
        """Trouver la meilleure langue"""
        best_lang = None
        best_score = 0
        
        for lang, tests in lang_stats.items():
            avg_accuracy = np.mean([t['word_accuracy'] for t in tests])
            avg_confidence = np.mean([t['avg_confidence'] for t in tests])
            
            score = (avg_accuracy + avg_confidence/100) / 2
            
            if score > best_score:
                best_score = score
                best_lang = lang
        
        return {
            'name': best_lang,
            'score': best_score,
            'tests_count': len(lang_stats[best_lang])
        }
    
    def analyze_by_difficulty(self, difficulty_stats):
        """Analyser par niveau de difficulté"""
        results = {}
        
        for difficulty, tests in difficulty_stats.items():
            avg_accuracy = np.mean([t['word_accuracy'] for t in tests])
            avg_confidence = np.mean([t['avg_confidence'] for t in tests])
            avg_time = np.mean([t['processing_time'] for t in tests])
            
            results[difficulty] = {
                'tests_count': len(tests),
                'avg_accuracy': avg_accuracy,
                'avg_confidence': avg_confidence,
                'avg_time': avg_time
            }
        
        return results
    
    def generate_recommendations(self, successful_tests):
        """Générer des recommandations"""
        recommendations = []
        
        # Recommandation de configuration
        fast_tests = [t for t in successful_tests if t['processing_time'] < 1.0]
        if fast_tests:
            best_fast = max(fast_tests, key=lambda x: x['word_accuracy'])
            recommendations.append(f"Pour la vitesse: {best_fast['configuration']} ({best_fast['processing_time']:.2f}s)")
        
        # Recommandation de précision
        accurate_tests = [t for t in successful_tests if t['avg_confidence'] > 80]
        if accurate_tests:
            best_accurate = max(accurate_tests, key=lambda x: x['word_accuracy'])
            recommendations.append(f"Pour la précision: {best_accurate['configuration']} ({best_accurate['avg_confidence']:.1f}% conf)")
        
        # Recommandation pour texte difficile
        difficult_tests = [t for t in successful_tests if t['image_difficulty'] == 'difficile']
        if difficult_tests:
            best_difficult = max(difficult_tests, key=lambda x: x['word_accuracy'])
            recommendations.append(f"Pour texte difficile: {best_difficult['configuration']} + preprocessing {best_difficult['preprocessing']}")
        
        return recommendations
    
    def generate_charts(self):
        """Générer des graphiques de performance"""
        successful_tests = [t for t in self.results['tests'] if t['success']]
        
        if not successful_tests:
            print("❌ Aucune donnée pour les graphiques")
            return
        
        # Configuration matplotlib
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Benchmark OCR - Résultats de Performance', fontsize=16, fontweight='bold')
        
        # 1. Temps de traitement par configuration
        configs = list(set([t['configuration'] for t in successful_tests]))
        config_times = []
        
        for config in configs:
            config_tests = [t for t in successful_tests if t['configuration'] == config]
            avg_time = np.mean([t['processing_time'] for t in config_tests])
            config_times.append(avg_time)
        
        axes[0, 0].bar(configs, config_times, color='skyblue')
        axes[0, 0].set_title('Temps de Traitement par Configuration')
        axes[0, 0].set_ylabel('Temps (secondes)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Précision par difficulté
        difficulties = ['facile', 'moyen', 'difficile']
        diff_accuracies = []
        
        for diff in difficulties:
            diff_tests = [t for t in successful_tests if t['image_difficulty'] == diff]
            if diff_tests:
                avg_accuracy = np.mean([t['word_accuracy'] for t in diff_tests]) * 100
                diff_accuracies.append(avg_accuracy)
            else:
                diff_accuracies.append(0)
        
        axes[0, 1].bar(difficulties, diff_accuracies, color='lightgreen')
        axes[0, 1].set_title('Précision par Difficulté')
        axes[0, 1].set_ylabel('Précision (%)')
        axes[0, 1].set_ylim(0, 100)
        
        # 3. Confiance par langue
        languages = list(set([t['language'] for t in successful_tests]))
        lang_confidences = []
        
        for lang in languages:
            lang_tests = [t for t in successful_tests if t['language'] == lang]
            avg_conf = np.mean([t['avg_confidence'] for t in lang_tests])
            lang_confidences.append(avg_conf)
        
        axes[1, 0].bar(languages, lang_confidences, color='orange')
        axes[1, 0].set_title('Confiance Moyenne par Langue')
        axes[1, 0].set_ylabel('Confiance (%)')
        axes[1, 0].set_ylim(0, 100)
        
        # 4. Relation Temps vs Précision
        times = [t['processing_time'] for t in successful_tests]
        accuracies = [t['word_accuracy'] * 100 for t in successful_tests]
        
        axes[1, 1].scatter(times, accuracies, alpha=0.6, color='red')
        axes[1, 1].set_title('Temps vs Précision')
        axes[1, 1].set_xlabel('Temps (secondes)')
        axes[1, 1].set_ylabel('Précision (%)')
        
        plt.tight_layout()
        
        # Sauvegarder
        chart_path = self.output_dir / "benchmark_charts.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        print(f"📊 Graphiques sauvegardés: {chart_path}")
        
        plt.show()
    
    def save_results(self):
        """Sauvegarder les résultats complets"""
        
        # JSON détaillé
        json_path = self.output_dir / "benchmark_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        # Rapport texte
        report_path = self.output_dir / "benchmark_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("RAPPORT DE BENCHMARK OCR\n")
            f.write("=" * 50 + "\n\n")
            
            # Informations système
            f.write("INFORMATIONS SYSTÈME\n")
            f.write("-" * 20 + "\n")
            for key, value in self.results['system_info'].items():
                f.write(f"{key}: {value}\n")
            
            f.write(f"\nRÉSUMÉ\n")
            f.write("-" * 20 + "\n")
            summary = self.results['summary']
            for key, value in summary.items():
                if key not in ['performance_by_difficulty', 'recommendations']:
                    f.write(f"{key}: {value}\n")
            
            f.write(f"\nRECOMMANDATIONS\n")
            f.write("-" * 20 + "\n")
            for rec in summary['recommendations']:
                f.write(f"• {rec}\n")
            
            f.write(f"\nDÉTAILS DES TESTS\n")
            f.write("-" * 20 + "\n")
            for i, test in enumerate(self.results['tests'][:10], 1):  # Top 10
                f.write(f"{i}. {test['configuration']}-{test['language']}-{test['image']}\n")
                f.write(f"   Temps: {test['processing_time']:.2f}s, ")
                f.write(f"Mots: {test['word_count']}, ")
                f.write(f"Confiance: {test['avg_confidence']:.1f}%\n")
        
        print(f"💾 Résultats sauvegardés:")
        print(f"   📄 JSON: {json_path}")
        print(f"   📄 Rapport: {report_path}")
    
    def run_full_benchmark(self):
        """Exécuter le benchmark complet"""
        print(f"🚀 DÉMARRAGE DU BENCHMARK COMPLET")
        print(f"=" * 50)
        
        start_time = time.time()
        
        try:
            # 1. Créer les images de test
            test_images = self.create_test_images()
            
            # 2. Exécuter les tests
            self.benchmark_configurations(test_images)
            
            # 3. Analyser les résultats
            self.analyze_results()
            
            # 4. Générer les graphiques
            self.generate_charts()
            
            # 5. Sauvegarder tout
            self.save_results()
            
            end_time = time.time()
            total_time = end_time - start_time
            
            print(f"\n🎉 BENCHMARK TERMINÉ")
            print(f"⏱️  Temps total: {total_time:.2f} secondes")
            print(f"📁 Résultats dans: {self.output_dir}")
            
        except Exception as e:
            print(f"❌ Erreur durante le benchmark: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Fonction principale"""
    print("🧪 BENCHMARK OCR - SNAPDRAGON X PLUS")
    print("=" * 50)
    
    try:
        benchmark = OCRBenchmark()
        benchmark.run_full_benchmark()
        
    except KeyboardInterrupt:
        print(f"\n🛑 Benchmark interrompu par l'utilisateur")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()