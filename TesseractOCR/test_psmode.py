#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de test pour les modes PSM avec enhancement CUSTOM
"""

from ocr import TesseractOCR, EnhancementLevel

IMAGE_PATH = "radio_frequency_lab_sign.jpg"

def main():
    ocr = TesseractOCR()
    results = ocr.extract_text_with_boxes(
        IMAGE_PATH,
        ['eng', 'glg'],
        80,
        EnhancementLevel.LIGHT
    )


if __name__ == "__main__":
    main()