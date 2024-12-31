# 🎵 Melodia: Algorithmic Music Composition 🎶

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.6%2B-blue.svg)](https://www.python.org/downloads/)

Melodia is a Python-based tool for generating musical compositions using algorithmic techniques. It leverages machine learning and deep learning to create unique and creative musical pieces with control over notes, durations, instruments, and articulations. 🎹

## ✨ Features

- 🎼 Generate unique musical compositions using your own MIDI files
- 🤖 LSTM networks for learning complex musical patterns
- 🎵 Control over musical elements:
  - Notes and chords
  - Duration patterns
  - Instrument selection
  - Articulation styles
- 🎛️ Adjustable generation parameters
- 💾 Save and load trained models (soon)

## 📦 Installation

1. Clone this repository:
    ```sh
    git clone https://github.com/caseybarajas/melodia.git
    ```
2. Navigate to the cloned repository:
    ```sh
    cd melodia
    ```
3. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## 🛠️ Usage

1. Place your MIDI training files in the `data` directory.

2. Train the model:
    ```sh
    python train.py
    ```

3. Generate music with custom parameters:
    ```sh
    python play.py --length 500 --temperature 1.0 --output output.mid
    ```

### Generation Parameters

- `--length`: Number of notes to generate (default: 500)
- `--temperature`: Controls randomness (0.1-2.0, default: 1.0)
  - Lower values = more conservative/predictable
  - Higher values = more experimental/random
- `--output`: Output MIDI filename (default: output.mid)

## 👥 Contributors

<a href="https://github.com/caseybarajas33/melodia/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=caseybarajas33/melodia" />
</a>

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
