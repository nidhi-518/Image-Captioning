# 🖼️ Image Captioning Model (With and Without Attention Mechanism)

## 📌 Overview
This project implements **two versions** of an image captioning model using **deep learning**:
1. **Without Attention Mechanism**: Uses a **CNN** (DenseNet201 or MobileNetV3Small) for **feature extraction** and an **LSTM-based RNN** for text generation.
2. **With Attention Mechanism**: Enhances caption quality by incorporating an **attention layer**, which allows the model to **focus on different parts** of an image while generating captions.

---

## 🌟 Features

### ✅ Without Attention:
- Uses **DenseNet201** or **MobileNetV3Small** for **image feature extraction**.
- Implements **LSTM-based RNN** for **text generation**.
- Utilizes **Tokenization and Padding** for **text preprocessing**.
- Includes **callbacks** like `EarlyStopping` and `ReduceLROnPlateau` to optimize training.

### 🔥 With Attention:
- Builds on the previous model but **introduces an Attention Layer**.
- Allows the model to **dynamically focus** on relevant parts of an image **during caption generation**.
- **Improves caption accuracy and relevance**.

---

## 📂 Dataset
- The dataset consists of **images and corresponding captions**.
- Ensure that images are stored in an **Images/** directory.
- Captions should be available in a **structured text format** (e.g., CSV or JSON).

---

## 🛠 Installation

### 📌 Prerequisites
Ensure you have Python installed along with the following libraries:

```sh
pip install tensorflow numpy pandas matplotlib seaborn tqdm
```

### 📥 Clone the Repository
```sh
git clone https://github.com/yourusername/Image-Captioning.git
cd Image-Captioning
```

---

## 🚀 Usage

### 🔹 Running the Model

#### **Without Attention:**
Run the Jupyter Notebook **step by step** to train the model.

```sh
jupyter notebook Image_Captioning.ipynb
```

#### **With Attention:**
Run the Jupyter Notebook for training the **enhanced model**.

```sh
jupyter notebook Image_Captioning_Attention.ipynb
```

---

### 🔹 Running Predictions
Once trained, use the model to **generate captions**:

```python
image = load_img('example.jpg', target_size=(224, 224))
caption = generate_caption(image, model)
print("Generated Caption:", caption)
```

---

## 🔮 Future Improvements
- Experiment with **Transformer-based models** (e.g., **BLIP, ViT, GPT-based captioning**).
- **Deploy as a Web App** using **Flask, FastAPI, or Streamlit**.
- Improve **Evaluation Metrics** (e.g., **BLEU, METEOR, CIDEr**).
- **Hyperparameter Tuning** for better accuracy.

---

## 🤝 Contributing
Feel free to **fork this repository** and submit **pull requests**. Any improvements are welcome! 🚀
