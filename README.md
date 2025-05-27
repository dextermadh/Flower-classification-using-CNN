# 🌸 Flower Classification with TensorFlow and MobileNetV2

This project uses the [TensorFlow Flower Photos dataset](http://download.tensorflow.org/example_images/flower_photos.tgz) to build a convolutional neural network that classifies flower images into five categories. It leverages transfer learning using MobileNetV2 and includes regularization techniques like Dropout, Batch Normalization, and L2 Regularization to reduce overfitting and improve generalization.

---

## 📁 Dataset

The dataset includes 5 flower classes:
- Daisy 🌼  
- Dandelion 🌾  
- Roses 🌹  
- Sunflowers 🌻  
- Tulips 🌷  

📥 **Download Link:**  
[http://download.tensorflow.org/example_images/flower_photos.tgz](http://download.tensorflow.org/example_images/flower_photos.tgz)

### Extract Dataset

```bash
wget http://download.tensorflow.org/example_images/flower_photos.tgz
tar -xvzf flower_photos.tgz
mkdir data
mv flower_photos data/
```

---

## 🧠 Model Overview

- ✅ Data Augmentation (Flip, Rotation, Zoom, Contrast)
- ✅ Transfer Learning using MobileNetV2
- ✅ Batch Normalization & Dropout
- ✅ L2 Regularization
- ✅ tf.data pipeline for performance

---

## 📊 Results

| Metric           | Value   |
|------------------|---------|
| Test Accuracy    | 0.8420  |

---

## 🖥️ System Requirements

- Python 3.8+
- TensorFlow 2.11+ (GPU recommended)
- NVIDIA GPU with ≥6GB (e.g., RTX 3050 Laptop)
- pip packages: `tensorflow`, `numpy`, `matplotlib`

---

## 💻 Installation

### 1. Clone the repo

```bash
git clone https://github.com/yourname/Flower-classification-using-CNN
cd flower-classifier
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

**Or install manually:**

```bash
pip install tensorflow numpy matplotlib Pillow opencv-python pandas
```

---

## 🧪 GPU Errors & Fixes

### ❌ Out Of Memory (OOM) Error

If you get an OOM error:
- Reduce batch size (e.g., from 32 → 8)
- Restart your runtime
- Close other GPU apps using:  
  ```bash
  nvidia-smi
  ```
  and kill unused processes

- Use efficient prefetching:
```python
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
```

- You can also reset your GPU in code (not always needed):
```python
import tensorflow.keras.backend as K
K.clear_session()
```

---

## 📂 Project Structure

```
.
├── README.md
├── notebook.ipynb
├── requirements.txt
├── data/
│   └── flower_photos/

```

## 🙌 Acknowledgements

- [TensorFlow Datasets](https://www.tensorflow.org/datasets)
- [Flower Photos Dataset](http://download.tensorflow.org/example_images/flower_photos.tgz)
- MobileNetV2 paper: [https://arxiv.org/abs/1801.04381](https://arxiv.org/abs/1801.04381)

---


