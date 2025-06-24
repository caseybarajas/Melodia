#!/usr/bin/env python3
"""
Test script to verify all Melodia imports and dependencies
"""

def test_imports():
    errors = []
    successes = []
    
    # Test basic Python modules
    try:
        import sys
        import os
        import json
        from pathlib import Path
        successes.append("✅ Standard library modules")
    except Exception as e:
        errors.append(f"❌ Standard library: {e}")
    
    # Test PyTorch
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, Dataset
        successes.append(f"✅ PyTorch {torch.__version__}")
        
        # Test CUDA
        if torch.cuda.is_available():
            successes.append(f"✅ CUDA available: {torch.cuda.device_count()} GPU(s)")
        else:
            successes.append("⚠️  CUDA not available (CPU only)")
            
    except Exception as e:
        errors.append(f"❌ PyTorch: {e}")
    
    # Test music libraries
    try:
        import mido
        successes.append("✅ mido (MIDI processing)")
    except Exception as e:
        errors.append(f"❌ mido: {e}")
    
    try:
        import pretty_midi
        successes.append("✅ pretty_midi")
    except Exception as e:
        errors.append(f"❌ pretty_midi: {e}")
    
    try:
        import music21
        successes.append("✅ music21")
    except Exception as e:
        errors.append(f"❌ music21: {e}")
    
    # Test data science libraries
    try:
        import numpy as np
        import pandas as pd
        from sklearn.metrics import accuracy_score
        successes.append("✅ Data science libraries (numpy, pandas, sklearn)")
    except Exception as e:
        errors.append(f"❌ Data science libraries: {e}")
    
    # Test GUI libraries
    try:
        import ttkbootstrap
        from ttkbootstrap import ttk
        successes.append("✅ ttkbootstrap (GUI)")
    except Exception as e:
        errors.append(f"❌ ttkbootstrap: {e}")
    
    # Test progress bars
    try:
        from tqdm import tqdm
        successes.append("✅ tqdm (progress bars)")
    except Exception as e:
        errors.append(f"❌ tqdm: {e}")
    
    # Test Melodia modules
    try:
        from melodia.config import MelodiaConfig, ModelConfig, TrainingConfig
        successes.append("✅ Melodia config")
    except Exception as e:
        errors.append(f"❌ Melodia config: {e}")
    
    try:
        from melodia.model.architecture import MelodiaModel
        successes.append("✅ Melodia model architecture")
    except Exception as e:
        errors.append(f"❌ Melodia model: {e}")
    
    try:
        from melodia.data.loader import MIDILoader
        from melodia.data.processor import DataProcessor
        successes.append("✅ Melodia data processing")
    except Exception as e:
        errors.append(f"❌ Melodia data: {e}")
    
    try:
        from melodia.training.trainer import Trainer
        successes.append("✅ Melodia training")
    except Exception as e:
        errors.append(f"❌ Melodia training: {e}")
    
    try:
        from melodia.generation.generator import MusicGenerator
        successes.append("✅ Melodia generation")
    except Exception as e:
        errors.append(f"❌ Melodia generation: {e}")
    
    try:
        from melodia.evaluation.metrics import MusicEvaluator
        successes.append("✅ Melodia evaluation")
    except Exception as e:
        errors.append(f"❌ Melodia evaluation: {e}")
    
    try:
        from melodia.utils.midi import MIDIProcessor
        from melodia.utils.music import MusicalAnalyzer
        successes.append("✅ Melodia utils")
    except Exception as e:
        errors.append(f"❌ Melodia utils: {e}")
    
    # Print results
    print("🎵 MELODIA DEPENDENCY CHECK 🎵")
    print("=" * 50)
    
    for success in successes:
        print(success)
    
    if errors:
        print("\n" + "⚠️  ERRORS FOUND:" + " " * 30)
        for error in errors:
            print(error)
        return False
    else:
        print(f"\n🎉 ALL {len(successes)} CHECKS PASSED!")
        return True

if __name__ == "__main__":
    success = test_imports()
    if success:
        print("\n✅ Melodia is ready to use!")
    else:
        print("\n❌ Please fix the import errors above")
        exit(1) 