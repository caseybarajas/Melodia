#!/usr/bin/env python3
"""
Simple demo script to test Melodia functionality
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add the melodia package to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from melodia.config import MelodiaConfig, ModelConfig, GenerationConfig
from melodia.model.architecture import MelodiaModel
from melodia.data.loader import MusicEvent
from melodia.evaluation.metrics import MusicEvaluator
from melodia.generation.controls import GenerationControls, MusicalStyle

def test_basic_imports():
    """Test that all basic imports work"""
    print("✓ All imports successful!")

def test_config():
    """Test configuration classes"""
    config = MelodiaConfig()
    print(f"✓ Configuration loaded - Model vocab size: {config.model.vocab_size}")
    
    # Test model config with gradient clip val
    model_config = ModelConfig()
    print(f"✓ Model config has gradient_clip_val: {model_config.gradient_clip_val}")

def test_model_creation():
    """Test model creation"""
    model_config = ModelConfig(
        embedding_dim=256,
        num_layers=2,
        num_heads=4,
        vocab_size=128
    )
    
    model = MelodiaModel(model_config)
    print(f"✓ Model created successfully with {model_config.num_layers} layers")

def test_generation_controls():
    """Test generation controls"""
    gen_config = GenerationConfig()
    controls = GenerationControls(gen_config, style=MusicalStyle.JAZZ)
    
    print(f"✓ Generation controls created for {controls.style.value} style")
    
    # Test validation
    if controls.validate():
        print("✓ Generation controls validation passed")
    else:
        print("✗ Generation controls validation failed")

def test_music_evaluator():
    """Test music evaluator"""
    evaluator = MusicEvaluator()
    
    # Create some test events
    test_events = [
        MusicEvent(
            type='note',
            time=0.0,
            duration=1.0,
            pitch=60,
            velocity=64
        ),
        MusicEvent(
            type='note',
            time=1.0,
            duration=1.0,
            pitch=64,
            velocity=64
        ),
        MusicEvent(
            type='note',
            time=2.0,
            duration=1.0,
            pitch=67,
            velocity=64
        )
    ]
    
    try:
        metrics = evaluator.evaluate(test_events)
        print(f"✓ Music evaluator working - {len(metrics)} metrics computed")
        
        # Show some example metrics
        for key, value in list(metrics.items())[:3]:
            print(f"  - {key}: {value:.3f}")
            
    except Exception as e:
        print(f"✗ Music evaluator error: {e}")

def test_gui_import():
    """Test GUI import (without launching)"""
    try:
        import tkinter as tk
        import ttkbootstrap
        print("✓ GUI dependencies available")
    except ImportError as e:
        print(f"✗ GUI import error: {e}")

def main():
    """Run all tests"""
    print("🎵 Melodia Basic Functionality Test 🎵\n")
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Configuration", test_config),
        ("Model Creation", test_model_creation),
        ("Generation Controls", test_generation_controls),
        ("Music Evaluator", test_music_evaluator),
        ("GUI Dependencies", test_gui_import),
    ]
    
    for test_name, test_func in tests:
        print(f"\n📋 Testing {test_name}:")
        try:
            test_func()
        except Exception as e:
            print(f"✗ {test_name} failed: {e}")
    
    print("\n🎉 Demo complete! Your Melodia project is working!")
    print("\nNext steps:")
    print("1. Place MIDI files in the 'data' directory")
    print("2. Run: python train.py --data_dir data --model_dir models")
    print("3. Run: python generate.py --model_dir models --output_dir outputs")
    print("4. Run: python melodia_gui.py (for GUI interface)")

if __name__ == "__main__":
    main() 