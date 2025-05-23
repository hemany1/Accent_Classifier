import os
import subprocess
import tempfile
import numpy as np
import torchaudio
import streamlit as st
from transformers import pipeline
from yt_dlp import YoutubeDL
import matplotlib.pyplot as plt

# === Helper Functions ===

def convert_to_wav(input_path, output_path):
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", input_path, "-ar", "16000", "-ac", "1", output_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    if not os.path.exists(output_path):
        raise RuntimeError(f"❌ ffmpeg failed — '{output_path}' not created.\n{result.stderr}")

def download_youtube_audio(youtube_url, output_path=None):
    if output_path is None:
        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, "downloaded.mp4")
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'quiet': True,
        'noplaylist': True
    }

    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    
    return output_path

def is_remote_url(path):
    return path.startswith("http://") or path.startswith("https://")

def get_audio_path(input_source):
    temp_dir = tempfile.mkdtemp()
    wav_path = os.path.join(temp_dir, "output.wav")

    if is_remote_url(input_source):
        downloaded_path = download_youtube_audio(input_source)
        convert_to_wav(downloaded_path, wav_path)
    else:
        convert_to_wav(input_source, wav_path)

    return wav_path

def classify_audio(audio_path, clf_pipeline):
    waveform, sample_rate = torchaudio.load(audio_path)
    resampled = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
    input_array = np.squeeze(resampled.numpy())
    result = clf_pipeline(input_array)
    return result

def plot_results(predictions):
    labels = [item["label"] for item in predictions]
    scores = [item["score"] * 100 for item in predictions]

    fig, ax = plt.subplots()
    bars = ax.barh(labels, scores, color="skyblue")
    ax.set_xlabel("Confidence (%)")
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    for bar in bars:
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f"{bar.get_width():.1f}%", va='center')
    st.pyplot(fig)

# === Streamlit UI ===

st.set_page_config(page_title="Accent Classifier", page_icon="🗣️")
st.title("🗣️ Accent Classifier")

st.markdown("""
This app classifies spoken accents from audio or video.

### 🔍 Available Models:
- **`Abdelrahman2865/Accent_Classifier`** — *Custom model by Abdelrahman*  
  Detects: 🇺🇸 American, 🇬🇧 British, 🇮🇳 Indian, 🇦🇺 Australian, 🇿🇦 South African, Other
- **`dima806/multiple_accent_classification`** — *Multilingual model*  
  Detects: 🇸🇦 Arabic, 🇬🇧 English, 🇪🇸 Spanish, 🇫🇷 French, 🇷🇺 Russian, 🇨🇳 Mandarin, Other
""")

# Model selection
model_choice = st.selectbox("Select a model:", [
    "Abdelrahman2865/Accent_Classifier",
    "dima806/multiple_accent_classification"
])
clf_pipeline = pipeline("audio-classification", model=model_choice)

# Threshold slider
threshold = st.slider("Confidence threshold", min_value=0.0, max_value=1.0, value=0.0, step=0.01)

# Input method
source = st.radio("Select input source:", ["📁 Upload file", "🌐 YouTube/public link"])

audio_path = None

if source == "📁 Upload file":
    uploaded_file = st.file_uploader("Upload audio/video file (.mp3, .mp4, .wav, etc.)", type=["mp3", "mp4", "wav", "m4a"])
    if uploaded_file:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as temp_input:
                temp_input.write(uploaded_file.read())
                audio_path = get_audio_path(temp_input.name)
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")

elif source == "🌐 YouTube/public link":
    youtube_url = st.text_input("Paste a YouTube video or Shorts URL:")
    if youtube_url:
        try:
            with st.spinner("Downloading and processing..."):
                audio_path = get_audio_path(youtube_url)
        except Exception as e:
            st.error(f"❌ Error: {e}")

# Run classifier
if audio_path:
    try:
        with st.spinner("Classifying accent..."):
            results = classify_audio(audio_path, clf_pipeline)
            filtered_results = [res for res in results if res['score'] >= threshold]
            top_result = filtered_results[0] if filtered_results else results[0]

        st.success("✅ Accent classified successfully!")
        st.markdown(f"**Top Prediction:** `{top_result['label']}`")
        st.markdown(f"**Confidence:** `{top_result['score'] * 100:.2f}%`")

        if filtered_results:
            st.subheader("🔢 Top Predictions")
            plot_results(filtered_results[:5])  # show up to top-5
        else:
            st.info("ℹ️ No predictions met the confidence threshold.")

    except Exception as e:
        st.error(f"❌ Error during classification: {e}")
