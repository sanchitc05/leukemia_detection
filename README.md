# 🧬 Leukemia Detection using Deep Learning

This project focuses on detecting leukemia from microscopic blood smear images using deep learning, combining custom CNNs and transfer learning techniques with robust evaluation and modular configuration.

---

## 📁 Project Structure

```

leukemia\_detection/
├── data/                    # Raw and processed dataset
│   ├── raw/                # Original downloaded dataset
│   ├── processed/          # Preprocessed and split dataset
│
├── models/                 # Saved models, checkpoints, and exports
│
├── results/                # Logs, plots, and metrics
│
├── scripts/                # CLI scripts for training, evaluation, prediction
│
├── src/                    # Source code
│   ├── **init**.py
│   ├── data\_preprocessing.py
│   ├── data\_augmentation.py
│   ├── evaluation.py
│   ├── model\_architecture.py
│   ├── model\_config.py
│   ├── training.py
│   ├── utils.py
│
├── tests/                  # Unit tests
│   ├── **init**.py
│   ├── test\_preprocessing.py
│   ├── test\_model.py
│   ├── test\_inference.py
│   ├── test\_utils.py
│   └── test\_data\_augmentation.py
│
├── config.yaml             # Full project configuration
├── main.py                 # Entry-point to train and evaluate model
├── setup.py                # Setup for pip installation
└── README.md

````

---

## 🧪 Dataset

- **Source:** [ALL-IDB1 Leukemia Dataset](https://www.kaggle.com/datasets/andrewmvd/leukemia-classification)
- **Classes:** `Healthy` vs `Leukemia`
- **Image Type:** Microscopic blood smear images

---

## 🚀 Quickstart

### 1. 🔧 Setup Environment

```bash
git clone https://github.com/sanchitc05/leukemia-detection.git
cd leukemia_detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Or venv\Scripts\activate on Windows

# Install dependencies
pip install -e .
````

### 2. 📦 Prepare Data

* Place the extracted dataset in: `data/raw/ALL_IDB1/`
* Then preprocess it:

```bash
python scripts/preprocess_data.py
```

### 3. 🏋️‍♀️ Train the Model

```bash
python scripts/train_model.py
```

### 4. 📈 Evaluate Model

```bash
python scripts/evaluate_model.py
```

### 5. 🔍 Predict

```bash
python scripts/predict.py --image path/to/image.jpg
```

---

## ⚙️ Configuration

All hyperparameters and paths are configurable via `config.yaml`. Examples:

```yaml
model:
  architecture: "efficientnet_b0"
  dropout_rate: 0.3
  pretrained_weights: "imagenet"

training:
  optimizer: "adam"
  epochs: 100
  initial_learning_rate: 0.001
  loss_function: "binary_crossentropy"
```

---

## 🧠 Model Options

Supports:

* ✅ Custom CNN
* ✅ EfficientNetB0, B1
* ✅ ResNet50
* ✅ DenseNet121
* ✅ MobileNetV2
* ✅ VGG16
* ✅ Fine-tuning top layers

---

## 📊 Evaluation Metrics

* Accuracy
* Precision / Recall
* F1 Score
* AUC (ROC)
* Confusion Matrix
* Classification Report

---

## 🧪 Testing

```bash
pytest tests/
```

---

## 📈 Logging & Monitoring

* TensorBoard logs stored at: `results/logs/`
* Plots for training history and confusion matrix are saved in: `results/plots/`

---

## ✨ Highlights

* ✅ Modular and clean architecture
* ✅ Easy configuration using YAML
* ✅ Transfer learning with fine-tuning
* ✅ Reproducibility with seed setting
* ✅ Visualizations for training and evaluation
* ✅ Class balancing support
* ✅ Unit tests for all components

---

## 🧪 Example Results

| Metric    | Value |
| --------- | ----- |
| Accuracy  | 97.5% |
| Precision | 97.1% |
| Recall    | 98.2% |
| AUC       | 0.99  |

*Note: These are indicative. Results vary based on architecture and training setup.*

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 🤝 Contributing

Pull requests, issues, and feature suggestions are welcome!

---

## 👨‍💻 Author

**Sanchit Chauhan**
Reach out via GitHub or LinkedIn.

```

---