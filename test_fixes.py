#!/usr/bin/env python3
"""
Test script to verify Melodia fixes and troubleshoot issues
Run with: python test_fixes.py
"""

import sys
from pathlib import Path
import torch
import numpy as np
import logging

# Add melodia to path
sys.path.insert(0, str(Path(__file__).parent))

def test_midi_loading():
    """Test MIDI loading with the fixed tempo extraction"""
    print("🧪 Testing MIDI loading...")
    
    try:
        from melodia.config import DataConfig
        from melodia.data.loader import MIDILoader
        
        data_config = DataConfig()
        loader = MIDILoader(data_config)
        
        # Test with actual MIDI files
        data_dir = Path("data")
        midi_files = list(data_dir.glob("**/*.mid"))
        
        if not midi_files:
            print("❌ No MIDI files found in data/ directory")
            return False
        
        print(f"📁 Found {len(midi_files)} MIDI files")
        
        successful_loads = 0
        for midi_file in midi_files[:3]:  # Test first 3 files
            try:
                events = loader.load_midi(midi_file)
                if events:
                    print(f"   ✅ {midi_file.name}: {len(events)} events loaded")
                    successful_loads += 1
                else:
                    print(f"   ⚠️  {midi_file.name}: No events loaded")
            except Exception as e:
                print(f"   ❌ {midi_file.name}: {str(e)}")
        
        if successful_loads > 0:
            print(f"✅ MIDI loading test passed ({successful_loads}/{len(midi_files[:3])} files)")
            return True
        else:
            print("❌ MIDI loading test failed")
            return False
            
    except Exception as e:
        print(f"❌ MIDI loading test error: {str(e)}")
        return False

def test_tokenizer():
    """Test the improved tokenizer"""
    print("\n🧪 Testing tokenizer...")
    
    try:
        from melodia.config import DataConfig
        from melodia.data.processor import EventTokenizer
        from melodia.data.loader import MusicEvent
        
        data_config = DataConfig()
        tokenizer = EventTokenizer(data_config)
        
        print(f"📚 Vocabulary size: {tokenizer.vocab_size}")
        
        # Create test events
        test_events = [
            MusicEvent(type='tempo', time=0.0, tempo=120.0),
            MusicEvent(type='note', time=0.0, pitch=60, velocity=80, duration=1.0),
            MusicEvent(type='note', time=1.0, pitch=64, velocity=75, duration=0.5),
            MusicEvent(type='note', time=1.5, pitch=67, velocity=70, duration=0.5),
        ]
        
        # Test encoding
        tokens = tokenizer.encode_events(test_events)
        print(f"🔤 Encoded {len(test_events)} events to {len(tokens)} tokens")
        
        # Test decoding
        decoded_events = tokenizer.decode_tokens(tokens)
        print(f"🔄 Decoded back to {len(decoded_events)} events")
        
        if len(decoded_events) > 0:
            print("✅ Tokenizer test passed")
            return True
        else:
            print("❌ Tokenizer test failed - no events decoded")
            return False
            
    except Exception as e:
        print(f"❌ Tokenizer test error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_model_creation():
    """Test model creation with correct dimensions"""
    print("\n🧪 Testing model creation...")
    
    try:
        from melodia.config import ModelConfig, DataConfig
        from melodia.data.processor import DataProcessor
        from melodia.model.architecture import MelodiaModel
        
        # Create proper config with correct vocab size
        data_config = DataConfig()
        data_processor = DataProcessor(data_config)
        vocab_size = data_processor.tokenizer.vocab_size
        
        model_config = ModelConfig(
            embedding_dim=128,
            num_layers=2,
            num_heads=4,
            vocab_size=vocab_size
        )
        
        print(f"🧠 Creating model with vocab_size={vocab_size}")
        
        model = MelodiaModel(model_config)
        
        # Test forward pass
        batch_size = 2
        seq_length = 64
        test_input = torch.randint(0, vocab_size, (batch_size, seq_length))
        
        with torch.no_grad():
            output = model(test_input)
        
        expected_shape = (batch_size, seq_length, vocab_size)
        if output.shape == expected_shape:
            print(f"✅ Model test passed - output shape: {output.shape}")
            return True
        else:
            print(f"❌ Model test failed - expected {expected_shape}, got {output.shape}")
            return False
            
    except Exception as e:
        print(f"❌ Model creation test error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_data_processing():
    """Test data processing pipeline"""
    print("\n🧪 Testing data processing...")
    
    try:
        from melodia.config import DataConfig, TrainingConfig
        from melodia.data.loader import MIDILoader
        from melodia.data.processor import DataProcessor
        
        data_config = DataConfig(max_sequence_length=256, min_sequence_length=32)
        training_config = TrainingConfig(batch_size=4)
        
        # Load some MIDI data
        loader = MIDILoader(data_config)
        data_dir = Path("data")
        midi_files = list(data_dir.glob("**/*.mid"))[:2]  # Just test with 2 files
        
        if not midi_files:
            print("❌ No MIDI files found for testing")
            return False
        
        events_list = []
        for midi_file in midi_files:
            events = loader.load_midi(midi_file)
            if events:
                events_list.append(events)
        
        if not events_list:
            print("❌ No events loaded from MIDI files")
            return False
        
        print(f"📊 Loaded {len(events_list)} sequences")
        
        # Process data
        data_processor = DataProcessor(data_config)
        processed_events = data_processor.process_events(events_list, augment=False)
        
        print(f"📊 Processed to {len(processed_events)} sequences")
        
        # Create dataset
        dataloader = data_processor.prepare_dataset(
            processed_events,
            batch_size=training_config.batch_size,
            shuffle=True
        )
        
        # Test one batch
        for batch_inputs, batch_targets in dataloader:
            print(f"✅ Data processing test passed - batch shape: {batch_inputs.shape}")
            return True
        
        print("❌ Data processing test failed - no batches created")
        return False
        
    except Exception as e:
        print(f"❌ Data processing test error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_training_setup():
    """Test training setup"""
    print("\n🧪 Testing training setup...")
    
    try:
        from melodia.config import ModelConfig, TrainingConfig, DataConfig
        from melodia.data.processor import DataProcessor
        from melodia.training.trainer import Trainer
        
        # Create minimal configs
        data_config = DataConfig(max_sequence_length=128, min_sequence_length=16)
        data_processor = DataProcessor(data_config)
        
        model_config = ModelConfig(
            embedding_dim=64,
            num_layers=1,
            num_heads=2,
            max_sequence_length=128,
            vocab_size=data_processor.tokenizer.vocab_size
        )
        
        training_config = TrainingConfig(
            batch_size=2,
            learning_rate=0.001,
            max_epochs=1
        )
        
        trainer = Trainer(
            model_config=model_config,
            training_config=training_config,
            data_processor=data_processor,
            model_dir="test_models"
        )
        
        print(f"✅ Training setup test passed")
        print(f"📊 Model parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")
        return True
        
    except Exception as e:
        print(f"❌ Training setup test error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🔧 Testing Melodia Fixes")
    print("=" * 50)
    
    tests = [
        ("MIDI Loading", test_midi_loading),
        ("Tokenizer", test_tokenizer),
        ("Model Creation", test_model_creation),
        ("Data Processing", test_data_processing),
        ("Training Setup", test_training_setup),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*50}")
    print("🔧 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("\n🎉 All tests passed! Your Melodia setup should work now.")
        print("\n💡 Next steps:")
        print("   1. Run: python train.py")
        print("   2. Use smaller model if you get memory errors")
        print("   3. Run: python generate.py after training")
    else:
        print(f"\n⚠️  {len(tests) - passed} tests failed. Please fix the issues above.")
        print("\n💡 Common fixes:")
        print("   - Make sure you have MIDI files in the data/ directory")
        print("   - Check that all dependencies are installed")
        print("   - Try reducing model size if you get memory errors")

if __name__ == "__main__":
    main() 