# 🐾 Oxford-IIIT Pet Breed Classifier

A high-end deep learning pipeline for classifying **37 cat and dog breeds** from the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) using **EfficientNetV2B2** with transfer learning and fine-tuning.

---

## 📁 Project Structure

```
oxford-iiit-pet/
├── images/                   # 7390 pet images (.jpg)
├── pet_classifier.ipynb      # Main high-end pipeline notebook
├── pet123.ipynb              # Original exploratory notebook
├── saved_models/             # Exported SavedModel + TFLite
│   ├── pet_breed_classifier/ # TensorFlow SavedModel
│   ├── pet_breed_classifier.tflite
│   └── class_labels.txt      # 37 breed names
├── logs/                     # Training curves, confusion matrix, Grad-CAM
└── README.md
```

---

## 🚀 Features

| Feature | Detail |
|---|---|
| **Backbone** | EfficientNetV2B2 pretrained on ImageNet |
| **Classes** | 37 pet breeds (cats & dogs) |
| **Input Size** | 260 × 260 px |
| **Data Pipeline** | `tf.data` — memory-efficient, no full dataset in RAM |
| **Augmentation** | Random Flip, Rotation, Zoom, Brightness, Contrast |
| **Training** | 2-phase: frozen head → fine-tune top backbone layers |
| **Loss** | Categorical Cross-Entropy with Label Smoothing (0.1) |
| **Metrics** | Top-1 Accuracy, Top-5 Accuracy |
| **Explainability** | Grad-CAM heatmaps |
| **Evaluation** | Confusion matrix, per-breed classification report |
| **Export** | TensorFlow SavedModel + TFLite (mobile-ready) |

---

## 🧠 Model Architecture

```
Input (260×260×3)
    └── EfficientNetV2B2 (ImageNet weights, partially frozen)
        └── GlobalAveragePooling2D
            └── Dense(512, relu)
                └── Dropout(0.4)
                    └── Dense(256, relu)
                        └── Dropout(0.2)
                            └── Dense(37, softmax)
```

---

## ⚙️ Setup

### Requirements

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn seaborn tqdm pillow
```

### Run

1. Clone the repo and place the `images/` folder in the project root
2. Open `pet_classifier.ipynb` in Jupyter or VS Code
3. Run all cells top to bottom

---

## 📊 Training Pipeline

### Phase 1 — Head Training
- Backbone frozen
- Only the custom Dense head is trained
- Learning rate: `1e-3`
- Early stopping on `val_accuracy` (patience=5)

### Phase 2 — Fine-tuning
- Top layers of EfficientNetV2B2 unfrozen (after layer index 200)
- Lower learning rate: `1e-5`
- Early stopping (patience=6)
- Best weights restored automatically

---

## 🔍 Grad-CAM Visualisation

The notebook generates Grad-CAM overlays showing which regions of the image the model focuses on when making predictions.

![Grad-CAM](logs/gradcam.png)

---

## 📈 Results

Training curves, confusion matrix, and per-breed metrics are saved to the `logs/` folder after running the notebook.

---

## 🗃️ Dataset

**Oxford-IIIT Pet Dataset**  
37 categories, ~200 images per breed  
Source: [https://www.robots.ox.ac.uk/~vgg/data/pets/](https://www.robots.ox.ac.uk/~vgg/data/pets/)

> The images folder is not included in this repo due to size. Download it from the link above and place it at `./images/`.

---

## 📦 Export

After training, the model is exported in two formats:

- **SavedModel** → `saved_models/pet_breed_classifier/` (for TensorFlow Serving)
- **TFLite** → `saved_models/pet_breed_classifier.tflite` (for mobile/edge deployment)

---

## 👤 Author

**Rupaji Balaji**  
[GitHub](https://github.com/RupajiBalaji)
