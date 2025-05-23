import os
import subprocess
import tempfile
import numpy as np
import torchaudio
import streamlit as st
from transformers import pipeline
from yt_dlp import YoutubeDL

# === Model Setup ===
clf_pipeline = pipeline("audio-classification", model="Abdelrahman2865/Accent_Classifier")

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

def classify_audio(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    resampled = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
    input_array = np.squeeze(resampled.numpy())
    result = clf_pipeline(input_array)[0]
    return result['label'], result['score']

# === Streamlit UI ===

st.set_page_config(page_title="Accent Classifier", page_icon="🗣️")
st.title("🗣️ English Accent Classifier")
st.markdown("""
This app classifies English accents in audio or video clips. It can detect **American**, **British**, **Indian**, **South African**, and other accents.  
Upload a local file or paste a YouTube video/shorts link.
""")

# Input method
source = st.radio("Select input source:", ["📁 Upload file", "🌐 YouTube/public link"])

audio_path = None

if source == "📁 Upload file":
    uploaded_file = st.file_uploader("Upload audio/video file (.mp3, .mp4, .wav, etc.)", type=["mp3", "mp4", "wav", "m4a"])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as temp_input:
            temp_input.write(uploaded_file.read())
            audio_path = get_audio_path(temp_input.name)

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
            label, confidence = classify_audio(audio_path)
        st.success("✅ Accent classified successfully!")
        st.markdown(f"**Detected Accent:** `{label}`")
        st.markdown(f"**Confidence:** `{confidence * 100:.2f}%`")
    except Exception as e:
        st.error(f"❌ Error during classification: {e}")
