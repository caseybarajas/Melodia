# 🎵 Melodia: Algorithmic Music Composition in Python 🎶

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.6%2B-blue.svg)](https://www.python.org/downloads/)

Melodia is a Python-based tool for generating musical compositions using algorithmic techniques. It leverages machine learning and deep learning to create unique and creative musical pieces. Dive into the fusion of code and music! 🎹

## 🚀 CURRENTLY WORKING ON:

- Enhancing machine learning algorithms for better performance
- Developing a user-friendly GUI

## ✨ Features

- 🎼 Generate unique musical compositions using your own MIDI files.
- 🤖 Utilizes LSTM (Long Short-Term Memory) networks to learn and generate musical patterns.
- 💾 Save and load trained models for future use.
- 🎶 Export generated compositions as MIDI files.

## 📦 Installation

1. Clone this repository:
    ```sh
    git clone https://github.com/caseybarajas33/melodia.git
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

1. Prepare your example MIDI files and place them in the `examples` directory.
2. Run the training script:
    ```sh
    python train.py
    ```
3. Update the model name in `play.py` to the model you just trained.
4. Generate a new composition:
    ```sh
    python play.py
    ```
5. The generated composition will be saved as a MIDI file.

## 👥 Contributors

<a href="https://github.com/caseybarajas33/melodia/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=caseybarajas33/melodia" />
</a>

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

Some of the data used to train this model is partly built upon the MAESTRO dataset, introduced in the following paper:

Curtis Hawthorne, Andriy Stasyuk, Adam Roberts, Ian Simon, Cheng-Zhi Anna Huang, Sander Dieleman, Erich Elsen, Jesse Engel, and Douglas Eck. "Enabling Factorized Piano Music Modeling and Generation with the MAESTRO Dataset." In International Conference on Learning Representations, 2019.

The MAESTRO dataset can be found at: [MAESTRO Dataset](https://magenta.tensorflow.org/datasets/maestro)
