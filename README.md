****# MNIST Studio 🖊️🔢

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A modern **handwritten digit recognition app** using a custom CNN trained on MNIST dataset. Draw digits on a sleek canvas and get instant predictions with confidence scores!

## ✨ Features

- 🎨 **Interactive Drawing Canvas** - Draw digits naturally with mouse
- ⚡ **Real-time Predictions** - Instant CNN inference
- 📊 **Confidence Visualization** - Progress bar + percentage
- 🎯 **Smart Preprocessing** - Auto-crop, resize, normalize (MNIST standards)
- 🖥️ **Modern GUI** - Clean, responsive Tkinter interface
- 💾 **Production-ready Model** - Trained CNN with dropout regularization

## 📁 File Structure

```
Tp4/
├── README.md                 # This file
├── requirements.txt          # Dependencies
├── mnist_cnn_training.ipynb  # Model training notebook
├── models/
│   └── best_model_v2.pth     # Trained CNN weights (~best accuracy)
├── MNIST/
│   └── MnistCNN.py           # CNN model architecture
└── src/
    ├── app.py                # Main GUI application
    ├── predictor.py          # Model loader & inference
    └── pretraitement.py      # Image preprocessing pipeline
```

## 🛠️ Quick Start

### 1. Clone & Install

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Model

Ensure `models/best_model_v2.pth` exists (trained via notebook).

### 3. Run the App

```bash
python src/app.py
```

### 4. Draw & Predict!

- Draw a digit (0-9) on the canvas
- Click **Predict Number** → Get result + confidence
- **Clear Canvas** to try again

## 🎯 How It Works

```
Canvas → preprocess_image() → Predictor.predict() → CNN → Result + Confidence
```

**Preprocessing Pipeline:**

1. Grayscale + Invert
2. Auto-crop bounding box
3. Resize to 20x20 (aspect-preserving)
4. Center in 28×28 canvas
5. Normalize: `(pixel - 0.1307) / 0.3081`
6. Shape: `(1, 1, 28, 28)`

**Model:** Custom CNN (Conv2D → MaxPool → Dropout → FC) trained on MNIST.

## 🚀 Example Usage

```
Draw: '7' → Prediction: 7 (94.2% confidence)
Draw: '3' → Prediction: 3 (87.5% confidence)
```

## 🔍 Model Performance

- **Accuracy**: ~99% on MNIST test set (version 2 improvements)
- **Input**: 28×28 grayscale
- **Output**: 10-class softmax probabilities

## 🧪 Development

### Train Your Own Model

```bash
jupyter notebook mnist_cnn_training.ipynb
```

### Customize

- Edit `MNIST/MnistCNN.py` for architecture changes
- Adjust preprocessing in `src/pretraitement.py`
- Model path in `src/predictor.py`

## 🤝 Contributing

1. Fork & clone
2. Install dev deps
3. Create feature branch
4. Submit PR!

## 📄 License

MIT License - see [LICENSE](LICENSE) (create if needed).

## 🙏 Acknowledgments

- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- PyTorch Team

---

**⭐ Star if useful!** Questions? Open an issue.
