# 🗣️ Accent Classification AI Agent

This project is an end-to-end AI solution that classifies English speech accents from audio or video inputs. It includes a fine-tuned model trained on real-world speech data and a deployable Streamlit app that allows users to classify accents directly from file uploads or public YouTube links.

## 🌍 Supported Accents
- 🇺🇸 American  
- 🇬🇧 British  
- 🇮🇳 Indian  
- 🇦🇺 Australian  
- 🇿🇦 South African  
- 🌐 Others (any accent outside the above categories)

---

## 📂 Repository Structure

├── app/ # Streamlit app for deployment
│ └── app.py
├── notebook/ # Jupyter playground for model training & experiments
│ └── speech_accent_multiclass_classification (1).ipynb
├── package/ # Core reusable code or helpers
├── requirements.txt # Python dependencies
└── README.md


---

## 🧠 Model Training Overview

### 📦 Dataset
We used the [CommonVoice 16th dataset](https://commonvoice.mozilla.org/en/datasets) by Mozilla Foundation. It contains thousands of labeled audio samples in various English accents.

### ⚙️ Preprocessing
- Filtered out languages not in our target set.
- Grouped all non-target accents under a single **"Other"** label.
- Normalized sampling rates and truncated samples >60s.

### 🧪 Fine-tuning
- Base model: [`facebook/wav2vec2-base`](https://huggingface.co/facebook/wav2vec2-base)  
- Training duration: **5 epochs** (limited by compute constraints)  
- Framework: Hugging Face Transformers + PyTorch

### 📊 Results

| Metric        | Score     |
|---------------|-----------|
| Accuracy      | 0.71      |
| ROC AUC       | 0.91      |
| Loss (final)  | ~0.59     |

📌 **Model Checkpoint:**  
🔗 [`Abdelrahman2865/Accent_Classifier`](https://huggingface.co/Abdelrahman2865/Accent_Classifier)

---

## 🚀 App Features

The app is built using **Streamlit** and supports:
- Uploading local `.mp3`, `.mp4`, `.wav`, `.m4a` files
- Pasting YouTube links for automatic audio extraction
- Interactive model selection from available Hugging Face pipelines
- Visual bar chart for **Top-N predictions**
- Confidence threshold filtering for advanced users

🔗 **Try the App Live:**  
[https://accentclassifier-9qcjbmurgjnbgyegfcf4ar.streamlit.app/](https://accentclassifier-9qcjbmurgjnbgyegfcf4ar.streamlit.app/)

---

## 🧪 Notebooks
In `notebook/Accent_Classifier_Playground.ipynb`, you'll find:

- Full data preprocessing workflow  
- Visualizations & EDA  
- Model fine-tuning loop with Hugging Face Trainer  
- Evaluation metrics & ROC curves  

## 🤝 Contributions
Contributions are welcome! If you'd like to improve model accuracy, UI experience, or extend it to more accents or languages, feel free to fork and PR.

## 📜 License
This project is open-source and free to use under the MIT License.

## 🔗 Links
- 🔥 [Hugging Face Model: Abdelrahman2865/Accent_Classifier](https://huggingface.co/Abdelrahman2865/Accent_Classifier)  
- 🧠 [Base Model: facebook/wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base)  
- 🎧 [Dataset: Common Voice by Mozilla](https://commonvoice.mozilla.org/en/datasets)  
- 🧪 [Demo App: Accent Classifier Web App](https://accentclassifier-9qcjbmurgjnbgyegfcf4ar.streamlit.app/)
