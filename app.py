import streamlit as st
import os
import subprocess
import tempfile
import torchaudio
import numpy as np
from pytube import YouTube
from transformers import pipeline

# === App Header ===
st.title("🌍 Accent Classifier")
st.write("""
This application classifies spoken accents from video or audio using Hugging Face models.

**Choose from two models:**

- **`Abdelrahman2865/Accent_Classifier`** *(Custom model)*  
  Developed by Abdelrahman, this model distinguishes between:
  🇺🇸 American, 🇬🇧 British, 🇮🇳 Indian, 🇦🇺 Australian, 🇿🇦 South African, Other

- **`dima806/multiple_accent_classification`** *(Multilingual model)*  
  A general-purpose model trained to classify:
  🇸🇦 Arabic, 🇬🇧 English, 🇪🇸 Spanish, 🇫🇷 French, 🇷🇺 Russian, 🇨🇳 Mandarin, Other
""")


# === Model Selector ===
model_option = st.selectbox(
    "Choose a model for accent classification:",
    ("Abdelrahman2865/Accent_Classifier", "dima806/multiple_accent_classification")
)

@st.cache_resource
def load_model(model_name):
    return pipeline("audio-classification", model=model_name)

clf_pipeline = load_model(model_option)

# === Utilities ===
def convert_to_wav(input_path, output_path):
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", input_path, "-ar", "16000", "-ac", "1", output_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    if not os.path.exists(output_path):
        raise RuntimeError("❌ ffmpeg failed — output.wav was not created.")

def is_remote_url(path):
    return path.startswith("http://") or path.startswith("https://")

def extract_audio_from_youtube(youtube_url):
    yt = YouTube(youtube_url)
    stream = yt.streams.filter(only_audio=True).first()
    temp_dir = tempfile.mkdtemp()
    audio_path = os.path.join(temp_dir, "audio.mp4")
    stream.download(filename=audio_path)
    wav_path = os.path.join(temp_dir, "output.wav")
    convert_to_wav(audio_path, wav_path)
    return wav_path

def prepare_local_audio(uploaded_file):
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())
    wav_path = os.path.join(temp_dir, "output.wav")
    convert_to_wav(temp_path, wav_path)
    return wav_path

def classify_audio(wav_path):
    waveform, sample_rate = torchaudio.load(wav_path)
    resampled = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
    input_array = np.squeeze(resampled.numpy())
    result = clf_pipeline(input_array)[0]
    return result["label"], result["score"]

# === Upload or Link ===
input_mode = st.radio("Choose input method:", ("YouTube link", "Local file"))

if input_mode == "YouTube link":
    youtube_url = st.text_input("Paste YouTube video URL (audio only)")
    if youtube_url:
        try:
            with st.spinner("Downloading and converting audio..."):
                audio_path = extract_audio_from_youtube(youtube_url)
                label, score = classify_audio(audio_path)
                st.success(f"✅ Detected Accent: **{label}** ({score*100:.2f}%)")
        except Exception as e:
            st.error(f"Something went wrong: {e}")

elif input_mode == "Local file":
    uploaded = st.file_uploader("Upload audio or video file", type=["mp3", "mp4", "wav", "m4a"])
    if uploaded:
        try:
            with st.spinner("Processing file..."):
                audio_path = prepare_local_audio(uploaded)
                label, score = classify_audio(audio_path)
                st.success(f"✅ Detected Accent: **{label}** ({score*100:.2f}%)")
        except Exception as e:
            st.error(f"Something went wrong: {e}")
