# 🎵 Melodia - AI Music Composition System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.13+](https://img.shields.io/badge/tensorflow-2.13+-orange.svg)](https://tensorflow.org/)
[![GPU Accelerated](https://img.shields.io/badge/GPU-accelerated-green.svg)](https://developer.nvidia.com/cuda-downloads)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **A sophisticated AI-powered music composition system using transformer neural networks for generating beautiful melodies.**

## ✨ **What's New in v2.0**

🚀 **Major Updates:**
- ✅ **Interactive Training** - Just run `python train.py` and configure everything interactively
- ✅ **GPU Auto-Detection** - Automatically detects and configures GPU for 10x speed boost
- ✅ **Beautiful Progress Bars** - Real-time training progress in both CLI and GUI
- ✅ **Performance Optimizations** - Mixed precision, XLA compilation, optimized data pipeline
- ✅ **Modern GUI** - Redesigned interface with live progress tracking
- ✅ **Smart Defaults** - Automatically adjusts model complexity based on hardware

## 🎯 **Features**

### 🧠 **AI Model**
- **Transformer Architecture** with multi-head attention
- **Relative Position Encoding** for musical context
- **GPU Acceleration** with automatic mixed precision
- **Configurable Model Size** (2-6 layers, 128-512D embeddings)

### 🎼 **Music Processing**
- **MIDI File Support** - Train on your own MIDI collections
- **Event-Based Tokenization** - Handles notes, timing, dynamics
- **Multiple Time Signatures** - 4/4, 3/4, 6/8, and more
- **Key Signature Detection** - Automatic musical key analysis

### 🖥️ **User Experience**
- **Interactive Training Setup** - No command-line arguments needed
- **Real-Time Progress Bars** - See exactly what's happening
- **Modern GUI Interface** - Beautiful ttkbootstrap design
- **Smart Hardware Detection** - Optimizes for CPU or GPU automatically

### ⚡ **Performance**
- **GPU Training**: ~2 minutes per epoch (10x faster than CPU)
- **CPU Training**: ~20 minutes per epoch (optimized for CPU-only systems)
- **Automatic Optimization** - XLA, mixed precision, memory growth
- **Efficient Data Pipeline** - Caching, prefetching, parallel processing

## 🚀 **Quick Start**

### **1. Installation**
```bash
# Clone the repository
git clone https://github.com/yourusername/Melodia.git
cd Melodia

# Install with GPU support (recommended)
pip install -r requirements.txt

# Verify GPU detection
python -c "import tensorflow as tf; print(f'GPUs: {len(tf.config.list_physical_devices(\"GPU\"))}')"
```

### **2. Interactive Training (Easiest Way)**
```bash
# Just run this - it will ask you everything!
python train.py
```

**The script will guide you through:**
- 📁 Data directory selection
- 🧠 Model configuration (auto-optimized for your hardware)
- 📚 Training parameters with smart defaults
- ⚡ Performance optimizations
- 🚀 Real-time training with progress bars

### **3. Quick Test Training**
```bash
# Fast 3-epoch test with minimal model
python quick_train.py
```

### **4. GUI Interface**
```bash
# Launch the modern GUI
python melodia_gui.py
```

## 📊 **Training Performance**

| Hardware | Time/Epoch | 10 Epochs | Model Size | Notes |
|----------|------------|-----------|------------|-------|
| **GPU** | ~2 min | ~20 min | 4 layers, 256D | Recommended |
| **CPU** | ~20 min | ~3.3 hours | 2 layers, 128D | Slower but works |

**GPU Benefits:**
- 🚀 **10x faster training**
- 🧠 **Larger models** (better quality)
- ⚡ **Mixed precision** (2x memory efficiency)
- 🔥 **XLA compilation** (additional speedup)

## 🎼 **Usage Examples**

### **Interactive Training Session**
```bash
$ python train.py

🎵 Welcome to Melodia Interactive Training! 🎵

🎵 ============================================ 🎵
    MELODIA INTERACTIVE TRAINING SETUP
🎵 ============================================ 🎵

📁 DATA CONFIGURATION
------------------------------
Training data directory [data]: <Enter>
✅ Found 7 MIDI files

🧠 MODEL CONFIGURATION
------------------------------
🔍 Checking GPU availability...
✅ Found 1 GPU(s): GPU 0: /physical_device:GPU:0
🚀 GPU detected - you can use larger models!
Embedding dimension [256]: <Enter>
Number of transformer layers [4]: <Enter>
...

🚀 ESTIMATED TRAINING TIME:
   GPU: ~40 minutes (0.7 hours)

▶️  Start training? [Y/n]: y

🚀 STARTING TRAINING...
Epochs: 20%|██████     | 2/10 [04:32<18:08, 136.05s/epoch, loss=0.7234, acc=0.812, time=2.3m]
Training: 67%|██████▋  | 402/603 [01:43<00:51, loss=0.7234]
```

### **Music Generation**
```bash
# Generate music from trained model
python generate.py --model_dir models --output_dir outputs --num_samples 3 --style jazz

# Output: Generated 3 MIDI files in outputs/
```

### **GUI Training**
```bash
python melodia_gui.py
# Click "Start Training" to see beautiful progress bars!
```

## 📁 **Project Structure**

```
Melodia/
├── 📄 train.py              # Interactive training (main entry point)
├── 📄 quick_train.py        # Fast test training
├── 📄 generate.py           # Music generation
├── 📄 melodia_gui.py        # Modern GUI interface
├── 📄 demo.py               # Basic functionality demo
├── 📁 data/                 # Training MIDI files
├── 📁 melodia/              # Core library
│   ├── 📁 model/            # Neural network architecture
│   ├── 📁 training/         # Training pipeline with progress bars
│   ├── 📁 data/             # Data processing & MIDI loading
│   ├── 📁 generation/       # Music generation
│   └── 📁 evaluation/       # Model evaluation metrics
├── 📁 models/               # Saved model checkpoints
├── 📁 outputs/              # Generated music files
└── 📁 tests/                # Unit tests
```

## 🎵 **Training Your Own Model**

### **1. Prepare Your Data**
```bash
# Place MIDI files in the data directory
data/
├── song1.mid
├── song2.mid
└── ...
```

### **2. Run Interactive Training**
```bash
python train.py
# Follow the interactive prompts!
```

### **3. Monitor Progress**
Watch the beautiful progress bars show:
- ✅ **Epoch Progress** - Overall training progress
- ✅ **Batch Progress** - Within-epoch progress  
- ✅ **Loss/Accuracy** - Real-time metrics
- ✅ **Time Estimates** - How long remaining

### **4. Generate Music**
```bash
python generate.py --model_dir models --num_samples 5
```

## ⚙️ **Configuration Options**

### **Model Architecture**
- **Embedding Dimension**: 128 (fast) to 512 (high quality)
- **Transformer Layers**: 2 (fast) to 6 (complex)
- **Attention Heads**: 4 (simple) to 12 (detailed)
- **Sequence Length**: 256 (short) to 1024 (long contexts)

### **Training Parameters**
- **Batch Size**: 8 (CPU) to 32 (GPU)
- **Learning Rate**: 0.0001 to 0.01
- **Epochs**: 10 (quick) to 200 (thorough)
- **Validation Split**: 0.1 (10% validation)

### **Hardware Optimization**
- **GPU**: Automatic mixed precision, XLA compilation
- **CPU**: Optimized threading, smaller models
- **Memory**: Dynamic GPU memory growth

## 🎨 **GUI Features**

### **Training Tab**
- 📁 **Data Directory Selection** with file browser
- 🧠 **Model Configuration** with intelligent defaults
- 📚 **Training Parameters** with validation
- 📊 **Live Progress Bars** with real-time metrics
- ⏱️ **Time Estimates** and completion tracking

### **Generation Tab**
- 🎼 **Model Selection** with file browser
- 🎵 **Style Selection** (Classical, Jazz, Folk, Blues)
- ⚙️ **Generation Parameters** (temperature, samples)
- 🎯 **Output Configuration** with preview

## 🔧 **Advanced Usage**

### **Custom Training Configuration**
```python
from melodia.config import ModelConfig, TrainingConfig
from melodia.training.trainer import Trainer

# Custom model configuration
model_config = ModelConfig(
    embedding_dim=256,
    num_layers=4,
    num_heads=8,
    max_sequence_length=512
)

# Custom training configuration  
training_config = TrainingConfig(
    batch_size=16,
    learning_rate=0.001,
    max_epochs=50
)
```

### **GPU Memory Configuration**
```python
import tensorflow as tf

# Configure GPU memory growth (done automatically)
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

## 🧪 **Testing**

```bash
# Run unit tests
python -m pytest tests/ -v

# Test specific components
python -m pytest tests/test_model.py -v
python -m pytest tests/test_training.py -v
```

## 📈 **Performance Tips**

### **For GPU Users**
- ✅ Use batch sizes 16-32
- ✅ Enable mixed precision (automatic)
- ✅ Use larger models (4-6 layers)
- ✅ Train for more epochs (50-200)

### **For CPU Users**
- ✅ Use batch sizes 4-8
- ✅ Use smaller models (2-3 layers)
- ✅ Reduce sequence length (256-512)
- ✅ Train for fewer epochs (10-30)

### **Data Optimization**
- ✅ Use multiple MIDI files for variety
- ✅ Ensure consistent time signatures
- ✅ Include diverse musical styles
- ✅ Preprocess long pieces into chunks

## 🐛 **Troubleshooting**

### **Common Issues**

**"No GPU found"**
```bash
# Install GPU support
pip install tensorflow[and-cuda]

# Verify CUDA installation
nvidia-smi
```

**"Training too slow"**
```bash
# Use quick training for testing
python quick_train.py

# Or reduce model size in interactive setup
python train.py
# Choose smaller embedding_dim and num_layers
```

**"Out of memory"**
```bash
# Reduce batch size in training configuration
# GPU: Try batch_size 8 instead of 16
# CPU: Try batch_size 4 instead of 8
```

**"MIDI files not loading"**
```bash
# Check file format
# Ensure files are standard MIDI (.mid or .midi)
# Check data directory contains MIDI files
```

## 🤝 **Contributing**

We welcome contributions! Here's how to get started:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Make your changes** with tests
4. **Run the tests** (`python -m pytest`)
5. **Submit a pull request**

### **Development Setup**
```bash
# Clone for development
git clone https://github.com/yourusername/Melodia.git
cd Melodia

# Install in development mode
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt
```

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- **TensorFlow** team for the amazing ML framework
- **music21** for comprehensive music analysis tools
- **ttkbootstrap** for the beautiful modern GUI components
- **tqdm** for the excellent progress bar library

## 📞 **Support**

- 📧 **Email**: your.email@example.com
- 🐛 **Issues**: [GitHub Issues](https://github.com/yourusername/Melodia/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/yourusername/Melodia/discussions)

---

**Made with ❤️ for music and AI enthusiasts**

*Transform your MIDI collections into AI-powered music generation with beautiful progress tracking and modern interfaces!* 🎵✨
