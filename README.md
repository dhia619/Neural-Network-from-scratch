## Handwritten digits recognition

![Demo](demo.gif)

This project includes a simple yet powerful **handwritten digits recognition tool**. It features an interactive drawing app that uses an artificial neural network implemented from scratch and trained on the **MNIST** dataset, achieving an impressive **91% accuracy** on the test set.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/dhia619/Neural-Network-from-scratch
   cd Neural-Network-from-scratch/src
   ```

2. Install requirements:
    ```bash
    pip install -r requirements.txt
    ```
   
## Usage

```bash
python drawingTool.py
```

## Project structure
- notebooks/: Contains the Jupyter notebook used for training the model on the MNIST dataset.
- src/: Contains the source code, including the drawing tool app for digit recognition.

## How it Works
1. The neural network is fully implemented from scratch in Python, using no external deep learning libraries.
2. The drawing tool allows you to draw digits, which the neural network classifies in real-time.

Feel free to explore the notebook to understand the model’s structure and the training process.