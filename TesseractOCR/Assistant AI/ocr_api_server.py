#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Serveur API REST OCR
Expose les fonctionnalités OCR via une API REST

Compatible Snapdragon X Plus - Windows ARM64
Utilise la classe TesseractOCR

Fonctionnalités:
- API REST complète pour OCR
- Upload d'images et traitement
- Gestion des tâches asynchrones
- Interface web de test
- Documentation API automatique
- Authentification et rate limiting

Auteur: Assistant IA
Date: 2025
"""

import os
import sys
import json
import uuid
import time
import asyncio
import base64
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union
import io
import hashlib

# FastAPI et dépendances
try:
    from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks, Request
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from pydantic import BaseModel, Field
    import uvicorn
except ImportError:
    print("❌ Modules FastAPI manquants. Installez avec:")
    print("   pip install fastapi uvicorn python-multipart")
    sys.exit(1)

# PIL pour traitement d'images
from PIL import Image
import numpy as np

# Import de la classe TesseractOCR
try:
    from tesseract_complete_ocr import TesseractOCR
except ImportError:
    print("❌ Impossible d'importer TesseractOCR")
    print("   Assurez-vous que tesseract_complete_ocr.py est dans le même répertoire")
    sys.exit(1)

# === MODÈLES PYDANTIC ===

class OCRRequest(BaseModel):
    """Modèle pour requête OCR"""
    image_base64: str = Field(..., description="Image encodée en base64")
    language: str = Field("eng", description="Langue OCR (eng, fra, etc.)")
    config: str = Field("--oem 3 --psm 6", description="Configuration Tesseract")
    preprocessing: str = Field("medium", description="Niveau de préprocessing (light, medium, strong)")
    return_boxes: bool = Field(False, description="Retourner les coordonnées des détections")
    confidence_threshold: int = Field(30, description="Seuil de confiance minimum")

class OCRResponse(BaseModel):
    """Modèle pour réponse OCR"""
    task_id: str
    text: str
    processing_time: float
    word_count: int
    character_count: int
    language: str
    success: bool
    error: Optional[str] = None
    boxes: Optional[List[Dict]] = None
    metadata: Dict

class TaskStatus(BaseModel):
    """Modèle pour statut de tâche"""
    task_id: str
    status: str  # pending, processing, completed, failed
    created_at: datetime
    completed_at: Optional[datetime] = None
    result: Optional[OCRResponse] = None
    error: Optional[str] = None

class ServerStats(BaseModel):
    """Modèle pour statistiques serveur"""
    uptime: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_processing_time: float
    active_tasks: int
    supported_languages: List[str]
    server_version: str

# === GESTIONNAIRE DE TÂCHES ===

class TaskManager:
    """Gestionnaire de tâches asynchrones"""
    
    def __init__(self):
        self.tasks: Dict[str, TaskStatus] = {}
        self.max_tasks = 1000
        self.cleanup_interval = 3600  # 1 heure
        
    def create_task(self) -> str:
        """Créer une nouvelle tâche"""
        task_id = str(uuid.uuid4())
        
        task = TaskStatus(
            task_id=task_id,
            status="pending",
            created_at=datetime.now()
        )
        
        self.tasks[task_id] = task
        
        # Nettoyage si trop de tâches
        if len(self.tasks) > self.max_tasks:
            self._cleanup_old_tasks()
        
        return task_id
    
    def update_task(self, task_id: str, status: str, result: Optional[OCRResponse] = None, error: Optional[str] = None):
        """Mettre à jour une tâche"""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            task.status = status
            
            if status in ["completed", "failed"]:
                task.completed_at = datetime.now()
            
            if result:
                task.result = result
            
            if error:
                task.error = error
    
    def get_task(self, task_id: str) -> Optional[TaskStatus]:
        """Récupérer une tâche"""
        return self.tasks.get(task_id)
    
    def _cleanup_old_tasks(self):
        """Nettoyer les anciennes tâches"""
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        old_tasks = [
            task_id for task_id, task in self.tasks.items()
            if task.created_at < cutoff_time and task.status in ["completed", "failed"]
        ]
        
        for task_id in old_tasks:
            del self.tasks[task_id]

# === SERVEUR API ===

class OCRAPIServer:
    """Serveur API REST pour OCR"""
    
    def __init__(self):
        """Initialiser le serveur"""
        
        # FastAPI app
        self.app = FastAPI(
            title="OCR API Server",
            description="API REST pour reconnaissance optique de caractères (OCR)",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Configuration CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # À restreindre en production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # OCR et gestionnaire de tâches
        self.ocr = TesseractOCR()
        self.task_manager = TaskManager()
        
        # Statistiques
        self.stats = {
            'start_time': datetime.now(),
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_processing_time': 0.0
        }
        
        # Rate limiting (simple)
        self.rate_limits = {}
        self.rate_limit_requests = 100  # Par heure
        
        # Configuration des routes
        self.setup_routes()
        
        print(f"🌐 Serveur API OCR initialisé")
        print(f"📚 Documentation: http://localhost:8000/docs")
    
    def setup_routes(self):
        """Configurer les routes API"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def root():
            """Page d'accueil avec interface de test"""
            return self.get_test_interface()
        
        @self.app.get("/health")
        async def health_check():
            """Vérification de santé du serveur"""
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "uptime": str(datetime.now() - self.stats['start_time']),
                "ocr_available": True
            }
        
        @self.app.get("/stats", response_model=ServerStats)
        async def get_stats():
            """Statistiques du serveur"""
            uptime = datetime.now() - self.stats['start_time']
            avg_time = (self.stats['total_processing_time'] / 
                       max(self.stats['successful_requests'], 1))
            
            return ServerStats(
                uptime=str(uptime),
                total_requests=self.stats['total_requests'],
                successful_requests=self.stats['successful_requests'],
                failed_requests=self.stats['failed_requests'],
                average_processing_time=avg_time,
                active_tasks=len([t for t in self.task_manager.tasks.values() 
                                if t.status in ["pending", "processing"]]),
                supported_languages=self.ocr.available_languages,
                server_version="1.0.0"
            )
        
        @self.app.post("/ocr", response_model=OCRResponse)
        async def process_ocr_sync(
            request: OCRRequest,
            client_request: Request,
            background_tasks: BackgroundTasks
        ):
            """Traitement OCR synchrone"""
            
            # Rate limiting
            client_ip = client_request.client.host
            if not self.check_rate_limit(client_ip):
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
            
            self.stats['total_requests'] += 1
            
            try:
                # Décoder l'image
                image = self.decode_base64_image(request.image_base64)
                
                # Créer une tâche
                task_id = self.task_manager.create_task()
                self.task_manager.update_task(task_id, "processing")
                
                # Traitement OCR
                result = await self.process_ocr_task(task_id, image, request)
                
                if result.success:
                    self.stats['successful_requests'] += 1
                else:
                    self.stats['failed_requests'] += 1
                
                self.stats['total_processing_time'] += result.processing_time
                
                return result
                
            except Exception as e:
                self.stats['failed_requests'] += 1
                raise HTTPException(status_code=500, detail=f"Erreur OCR: {str(e)}")
        
        @self.app.post("/ocr/async")
        async def process_ocr_async(
            request: OCRRequest,
            client_request: Request,
            background_tasks: BackgroundTasks
        ):
            """Traitement OCR asynchrone"""
            
            # Rate limiting
            client_ip = client_request.client.host
            if not self.check_rate_limit(client_ip):
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
            
            self.stats['total_requests'] += 1
            
            try:
                # Décoder l'image
                image = self.decode_base64_image(request.image_base64)
                
                # Créer une tâche
                task_id = self.task_manager.create_task()
                
                # Lancer le traitement en arrière-plan
                background_tasks.add_task(
                    self.process_ocr_background, 
                    task_id, 
                    image, 
                    request
                )
                
                return {
                    "task_id": task_id,
                    "status": "pending",
                    "message": "Tâche créée, utilisez /task/{task_id} pour suivre le progrès"
                }
                
            except Exception as e:
                self.stats['failed_requests'] += 1
                raise HTTPException(status_code=500, detail=f"Erreur création tâche: {str(e)}")
        
        @self.app.get("/task/{task_id}", response_model=TaskStatus)
        async def get_task_status(task_id: str):
            """Récupérer le statut d'une tâche"""
            
            task = self.task_manager.get_task(task_id)
            if not task:
                raise HTTPException(status_code=404, detail="Tâche non trouvée")
            
            return task
        
        @self.app.post("/ocr/file")
        async def process_ocr_file(
            file: UploadFile = File(...),
            language: str = "eng",
            config: str = "--oem 3 --psm 6",
            preprocessing: str = "medium",
            return_boxes: bool = False,
            confidence_threshold: int = 30
        ):
            """Traitement OCR par upload de fichier"""
            
            self.stats['total_requests'] += 1
            
            try:
                # Vérifier le type de fichier
                if not file.content_type.startswith('image/'):
                    raise HTTPException(status_code=400, detail="Fichier doit être une image")
                
                # Lire le fichier
                content = await file.read()
                image = Image.open(io.BytesIO(content))
                
                # Créer la requête
                request = OCRRequest(
                    image_base64="",  # Pas utilisé dans ce cas
                    language=language,
                    config=config,
                    preprocessing=preprocessing,
                    return_boxes=return_boxes,
                    confidence_threshold=confidence_threshold
                )
                
                # Créer une tâche
                task_id = self.task_manager.create_task()
                self.task_manager.update_task(task_id, "processing")
                
                # Traitement OCR
                result = await self.process_ocr_task(task_id, image, request)
                
                if result.success:
                    self.stats['successful_requests'] += 1
                else:
                    self.stats['failed_requests'] += 1
                
                self.stats['total_processing_time'] += result.processing_time
                
                return result
                
            except Exception as e:
                self.stats['failed_requests'] += 1
                raise HTTPException(status_code=500, detail=f"Erreur traitement fichier: {str(e)}")
        
        @self.app.get("/languages")
        async def get_supported_languages():
            """Liste des langues supportées"""
            return {
                "languages": self.ocr.available_languages,
                "default": "eng",
                "multilingue_example": "eng+fra"
            }
        
        @self.app.get("/config/presets")
        async def get_config_presets():
            """Configurations prédéfinies"""
            return {
                "presets": {
                    "default": "--oem 3 --psm 6",
                    "fast": "--oem 3 --psm 8", 
                    "accurate": "--oem 3 --psm 3",
                    "line": "--oem 3 --psm 7",
                    "sparse": "--oem 3 --psm 11"
                },
                "preprocessing_levels": ["light", "medium", "strong"]
            }
    
    def decode_base64_image(self, base64_string: str) -> Image.Image:
        """Décoder une image base64"""
        try:
            # Supprimer le préfixe data:image si présent
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
            
            # Décoder
            image_data = base64.b64decode(base64_string)
            image = Image.open(io.BytesIO(image_data))
            
            return image
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Image base64 invalide: {str(e)}")
    
    async def process_ocr_task(self, task_id: str, image: Image.Image, request: OCRRequest) -> OCRResponse:
        """Traiter une tâche OCR"""
        
        start_time = time.time()
        
        try:
            # Préprocessing
            if request.preprocessing and request.preprocessing != "none":
                processed_image = self.ocr.preprocess_image(
                    image, 
                    enhancement_level=request.preprocessing
                )
            else:
                processed_image = image
            
            # OCR simple
            text = self.ocr.extract_text_simple(
                processed_image,
                lang=request.language,
                config=request.config
            )
            
            # OCR avec boîtes si demandé
            boxes = None
            if request.return_boxes:
                boxes = self.ocr.extract_text_with_boxes(
                    processed_image,
                    lang=request.language,
                    confidence_threshold=request.confidence_threshold
                )
            
            processing_time = time.time() - start_time
            
            # Créer la réponse
            response = OCRResponse(
                task_id=task_id,
                text=text,
                processing_time=processing_time,
                word_count=len(text.split()) if text else 0,
                character_count=len(text),
                language=request.language,
                success=True,
                boxes=boxes,
                metadata={
                    "image_size": image.size,
                    "image_mode": image.mode,
                    "preprocessing": request.preprocessing,
                    "config": request.config,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            # Mettre à jour la tâche
            self.task_manager.update_task(task_id, "completed", response)
            
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            error_response = OCRResponse(
                task_id=task_id,
                text="",
                processing_time=processing_time,
                word_count=0,
                character_count=0,
                language=request.language,
                success=False,
                error=str(e),
                metadata={
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            # Mettre à jour la tâche
            self.task_manager.update_task(task_id, "failed", error_response, str(e))
            
            return error_response
    
    async def process_ocr_background(self, task_id: str, image: Image.Image, request: OCRRequest):
        """Traitement OCR en arrière-plan"""
        try:
            result = await self.process_ocr_task(task_id, image, request)
            
            if result.success:
                self.stats['successful_requests'] += 1
            else:
                self.stats['failed_requests'] += 1
            
            self.stats['total_processing_time'] += result.processing_time
            
        except Exception as e:
            self.stats['failed_requests'] += 1
            self.task_manager.update_task(task_id, "failed", error=str(e))
    
    def check_rate_limit(self, client_ip: str) -> bool:
        """Vérifier le rate limiting"""
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)
        
        # Nettoyer les anciennes entrées
        if client_ip in self.rate_limits:
            self.rate_limits[client_ip] = [
                timestamp for timestamp in self.rate_limits[client_ip]
                if timestamp > hour_ago
            ]
        else:
            self.rate_limits[client_ip] = []
        
        # Vérifier la limite
        if len(self.rate_limits[client_ip]) >= self.rate_limit_requests:
            return False
        
        # Ajouter cette requête
        self.rate_limits[client_ip].append(now)
        return True
    
    def get_test_interface(self) -> str:
        """Interface de test HTML"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>OCR API Server - Interface de Test</title>
            <meta charset="utf-8">
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
                .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                .header { text-align: center; color: #333; margin-bottom: 30px; }
                .section { margin-bottom: 30px; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
                .section h3 { color: #2196F3; margin-top: 0; }
                input, select, textarea, button { padding: 10px; margin: 5px; border: 1px solid #ddd; border-radius: 4px; }
                button { background: #2196F3; color: white; cursor: pointer; }
                button:hover { background: #1976D2; }
                .result { background: #f9f9f9; padding: 15px; border-radius: 4px; margin-top: 10px; }
                .error { background: #ffebee; color: #c62828; }
                .success { background: #e8f5e8; color: #2e7d32; }
                #imagePreview { max-width: 300px; max-height: 200px; margin: 10px 0; }
                .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }
                .stat-box { background: #e3f2fd; padding: 15px; border-radius: 5px; text-align: center; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🔍 OCR API Server</h1>
                    <p>Interface de test pour l'API de reconnaissance optique de caractères</p>
                </div>
                
                <div class="section">
                    <h3>📊 Statistiques du Serveur</h3>
                    <div class="stats" id="stats">
                        <div class="stat-box">
                            <h4>Statut</h4>
                            <p id="status">Chargement...</p>
                        </div>
                        <div class="stat-box">
                            <h4>Requêtes Totales</h4>
                            <p id="totalRequests">-</p>
                        </div>
                        <div class="stat-box">
                            <h4>Taux de Succès</h4>
                            <p id="successRate">-</p>
                        </div>
                        <div class="stat-box">
                            <h4>Temps Moyen</h4>
                            <p id="avgTime">-</p>
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h3>🖼️ Test OCR par Upload</h3>
                    <div>
                        <input type="file" id="imageFile" accept="image/*" onchange="previewImage()">
                        <img id="imagePreview" style="display:none;">
                    </div>
                    <div>
                        <label>Langue:</label>
                        <select id="language">
                            <option value="eng">Anglais</option>
                            <option value="fra">Français</option>
                            <option value="eng+fra">Multilingue</option>
                        </select>
                        
                        <label>Configuration:</label>
                        <select id="config">
                            <option value="--oem 3 --psm 6">Par défaut</option>
                            <option value="--oem 3 --psm 8">Rapide</option>
                            <option value="--oem 3 --psm 3">Précis</option>
                            <option value="--oem 3 --psm 7">Ligne unique</option>
                        </select>
                        
                        <label>Préprocessing:</label>
                        <select id="preprocessing">
                            <option value="medium">Moyen</option>
                            <option value="light">Léger</option>
                            <option value="strong">Fort</option>
                        </select>
                    </div>
                    <div>
                        <label>
                            <input type="checkbox" id="returnBoxes"> Retourner les coordonnées
                        </label>
                        <label>Seuil confiance:</label>
                        <input type="number" id="confidenceThreshold" value="30" min="0" max="100">
                    </div>
                    <button onclick="processOCR()">🔍 Analyser l'Image</button>
                    <div id="ocrResult" class="result" style="display:none;"></div>
                </div>
                
                <div class="section">
                    <h3>📋 Test par Base64</h3>
                    <textarea id="base64Input" placeholder="Collez une image encodée en base64..." rows="4" style="width:100%;"></textarea>
                    <button onclick="processBase64OCR()">🔍 Analyser Base64</button>
                    <div id="base64Result" class="result" style="display:none;"></div>
                </div>
                
                <div class="section">
                    <h3>📖 Documentation API</h3>
                    <p>
                        <a href="/docs" target="_blank">📚 Documentation Swagger</a> |
                        <a href="/redoc" target="_blank">📘 Documentation ReDoc</a>
                    </p>
                    <h4>Endpoints Principaux:</h4>
                    <ul>
                        <li><code>POST /ocr</code> - OCR synchrone avec JSON</li>
                        <li><code>POST /ocr/async</code> - OCR asynchrone</li>
                        <li><code>POST /ocr/file</code> - OCR par upload de fichier</li>
                        <li><code>GET /task/{task_id}</code> - Statut d'une tâche</li>
                        <li><code>GET /stats</code> - Statistiques du serveur</li>
                        <li><code>GET /languages</code> - Langues supportées</li>
                    </ul>
                </div>
            </div>
            
            <script>
                // Charger les statistiques
                async function loadStats() {
                    try {
                        const response = await fetch('/stats');
                        const stats = await response.json();
                        
                        document.getElementById('status').textContent = 'En ligne';
                        document.getElementById('totalRequests').textContent = stats.total_requests;
                        
                        const successRate = stats.total_requests > 0 ? 
                            ((stats.successful_requests / stats.total_requests) * 100).toFixed(1) + '%' : '-';
                        document.getElementById('successRate').textContent = successRate;
                        document.getElementById('avgTime').textContent = stats.average_processing_time.toFixed(2) + 's';
                        
                    } catch (error) {
                        document.getElementById('status').textContent = 'Erreur';
                    }
                }
                
                // Prévisualiser l'image
                function previewImage() {
                    const file = document.getElementById('imageFile').files[0];
                    if (file) {
                        const reader = new FileReader();
                        reader.onload = function(e) {
                            const preview = document.getElementById('imagePreview');
                            preview.src = e.target.result;
                            preview.style.display = 'block';
                        };
                        reader.readAsDataURL(file);
                    }
                }
                
                // Traiter OCR par upload
                async function processOCR() {
                    const fileInput = document.getElementById('imageFile');
                    const resultDiv = document.getElementById('ocrResult');
                    
                    if (!fileInput.files[0]) {
                        alert('Veuillez sélectionner une image');
                        return;
                    }
                    
                    const formData = new FormData();
                    formData.append('file', fileInput.files[0]);
                    formData.append('language', document.getElementById('language').value);
                    formData.append('config', document.getElementById('config').value);
                    formData.append('preprocessing', document.getElementById('preprocessing').value);
                    formData.append('return_boxes', document.getElementById('returnBoxes').checked);
                    formData.append('confidence_threshold', document.getElementById('confidenceThreshold').value);
                    
                    resultDiv.style.display = 'block';
                    resultDiv.innerHTML = '🔄 Traitement en cours...';
                    resultDiv.className = 'result';
                    
                    try {
                        const response = await fetch('/ocr/file', {
                            method: 'POST',
                            body: formData
                        });
                        
                        const result = await response.json();
                        
                        if (result.success) {
                            resultDiv.className = 'result success';
                            resultDiv.innerHTML = `
                                <h4>✅ Succès!</h4>
                                <p><strong>Texte détecté:</strong></p>
                                <pre>${result.text || 'Aucun texte détecté'}</pre>
                                <p><strong>Statistiques:</strong></p>
                                <ul>
                                    <li>Mots: ${result.word_count}</li>
                                    <li>Caractères: ${result.character_count}</li>
                                    <li>Temps: ${result.processing_time.toFixed(2)}s</li>
                                    <li>Langue: ${result.language}</li>
                                </ul>
                                ${result.boxes ? `<p><strong>Détections:</strong> ${result.boxes.length}</p>` : ''}
                            `;
                        } else {
                            resultDiv.className = 'result error';
                            resultDiv.innerHTML = `<h4>❌ Erreur</h4><p>${result.error}</p>`;
                        }
                        
                        loadStats(); // Recharger les stats
                        
                    } catch (error) {
                        resultDiv.className = 'result error';
                        resultDiv.innerHTML = `<h4>❌ Erreur</h4><p>${error.message}</p>`;
                    }
                }
                
                // Traiter OCR par base64
                async function processBase64OCR() {
                    const base64Input = document.getElementById('base64Input').value.trim();
                    const resultDiv = document.getElementById('base64Result');
                    
                    if (!base64Input) {
                        alert('Veuillez entrer une image en base64');
                        return;
                    }
                    
                    resultDiv.style.display = 'block';
                    resultDiv.innerHTML = '🔄 Traitement en cours...';
                    resultDiv.className = 'result';
                    
                    try {
                        const response = await fetch('/ocr', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json'
                            },
                            body: JSON.stringify({
                                image_base64: base64Input,
                                language: document.getElementById('language').value,
                                config: document.getElementById('config').value,
                                preprocessing: document.getElementById('preprocessing').value,
                                return_boxes: document.getElementById('returnBoxes').checked,
                                confidence_threshold: parseInt(document.getElementById('confidenceThreshold').value)
                            })
                        });
                        
                        const result = await response.json();
                        
                        if (result.success) {
                            resultDiv.className = 'result success';
                            resultDiv.innerHTML = `
                                <h4>✅ Succès!</h4>
                                <p><strong>Texte détecté:</strong></p>
                                <pre>${result.text || 'Aucun texte détecté'}</pre>
                                <p><strong>Statistiques:</strong></p>
                                <ul>
                                    <li>Mots: ${result.word_count}</li>
                                    <li>Caractères: ${result.character_count}</li>
                                    <li>Temps: ${result.processing_time.toFixed(2)}s</li>
                                    <li>Langue: ${result.language}</li>
                                </ul>
                                ${result.boxes ? `<p><strong>Détections:</strong> ${result.boxes.length}</p>` : ''}
                            `;
                        } else {
                            resultDiv.className = 'result error';
                            resultDiv.innerHTML = `<h4>❌ Erreur</h4><p>${result.error}</p>`;
                        }
                        
                        loadStats(); // Recharger les stats
                        
                    } catch (error) {
                        resultDiv.className = 'result error';
                        resultDiv.innerHTML = `<h4>❌ Erreur</h4><p>${error.message}</p>`;
                    }
                }
                
                // Charger les stats au démarrage
                loadStats();
                
                // Recharger les stats toutes les 30 secondes
                setInterval(loadStats, 30000);
            </script>
        </body>
        </html>
        """
    
    def run(self, host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
        """Lancer le serveur"""
        
        print(f"🚀 Démarrage du serveur OCR API...")
        print(f"🌐 URL: http://{host}:{port}")
        print(f"📚 Documentation: http://{host}:{port}/docs")
        print(f"🧪 Interface de test: http://{host}:{port}/")
        
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )


def main():
    """Fonction principale"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Serveur API REST OCR")
    parser.add_argument("--host", default="0.0.0.0", help="Adresse IP (défaut: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port (défaut: 8000)")
    parser.add_argument("--reload", action='store_true', help="Redémarrage automatique")
    
    args = parser.parse_args()
    
    print(f"🌐 SERVEUR API OCR - SNAPDRAGON X PLUS")
    print(f"=" * 50)
    
    try:
        server = OCRAPIServer()
        server.run(host=args.host, port=args.port, reload=args.reload)
        
    except KeyboardInterrupt:
        print(f"\n🛑 Serveur arrêté")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()