# models2.py
import os
import io
import base64
import numpy as np
import tempfile
from typing import Dict, Any, List

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langdetect import detect, detect_langs
import torch
import scipy.io.wavfile as wavfile
from gtts import gTTS

# ---------- Load models ONCE (module-level singletons) ----------
# Whisper-small (CPU)
asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",
    device=-1,  # CPU
)

# M2M100 multilingual translation model (CPU)
M2M_MODEL_ID = "facebook/m2m100_418M"
m2m_tokenizer = AutoTokenizer.from_pretrained(M2M_MODEL_ID)
m2m_model = AutoModelForSeq2SeqLM.from_pretrained(M2M_MODEL_ID)

# Optional: map generic/variant codes to M2M100 codes (expand as needed)
LANG_CODE_ALIAS = {
    "zh-cn": "zh", "zh-tw": "zh", "pt-br": "pt", "pt-pt": "pt",
    "he": "he", "iw": "he",  # Hebrew
}

def _normalize_lang_code(code: str) -> str:
    code = (code or "").lower().strip()
    return LANG_CODE_ALIAS.get(code, code)

# ---------- Core functions ----------
def transcribe_and_translate_to_target(file_path: str, target_lang: str) -> Dict[str, Any]:
    """
    1) Transcribe speech to text (auto language via Whisper).
    2) Auto-detect src language from transcribed text.
    3) Translate to `target_lang` using M2M100.
    """
    # 1) Transcribe (Whisper auto-detects; we read text)
    asr_out = asr_pipeline(file_path)
    text = asr_out.get("text", "").strip()

    if not text:
        raise ValueError("Transcription produced empty text.")

    # 2) Detect source language from text (langdetect)
    #    We also return candidate probabilities for transparency.
    try:
        src_lang = detect(text)
        candidates = detect_langs(text)  # e.g., ["en:0.999", "fr:0.001"]
        detection_conf = [
            {"lang": str(c).split(":")[0], "score": float(str(c).split(":")[1])}
            for c in candidates
        ]
    except Exception:
        src_lang = "en"
        detection_conf = [{"lang": "en", "score": 0.0}]

    src_lang = _normalize_lang_code(src_lang)
    tgt_lang = _normalize_lang_code(target_lang)

    if not tgt_lang:
        raise ValueError("Please provide a valid target language code, e.g., 'es', 'fr', 'en'.")

    # 3) Translate with M2M100
    #    Tell tokenizer the source language; force BOS to target language.
    m2m_tokenizer.src_lang = src_lang
    inputs = m2m_tokenizer(text, return_tensors="pt")

    try:
        forced_bos_token_id = m2m_tokenizer.get_lang_id(tgt_lang)
    except KeyError:
        raise ValueError(f"Unsupported target language for M2M100: '{tgt_lang}'")

    generated_ids = m2m_model.generate(
        **inputs,
        forced_bos_token_id=forced_bos_token_id,
        max_new_tokens=256
    )
    translation = m2m_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    return {
        "transcription": text,
        "source_lang_detected": src_lang,
        "target_lang": tgt_lang,
        "translation": translation,
        "language_detection_candidates": detection_conf,
    }

def text_to_audio(text: str, speaker_embedding_id: int = 0) -> bytes:
    """
    Convert text to speech using Google Text-to-Speech (gTTS).
    Returns raw audio bytes that can be streamed directly.
    
    Args:
        text: Text to convert to speech
        speaker_embedding_id: Not used but kept for API compatibility
    """
    if not text.strip():
        raise ValueError("Text cannot be empty.")
    
    # Limit text length
    if len(text) > 1000:
        text = text[:1000] + "..."
    
    try:
        # Create gTTS object
        tts = gTTS(text=text, lang='en', slow=False)
        
        # Use BytesIO instead of temporary file for better reliability
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        
        # Get the audio bytes
        audio_bytes = audio_buffer.getvalue()
        
        # Verify we have audio data
        if len(audio_bytes) == 0:
            raise ValueError("Generated audio is empty")
            
        return audio_bytes
        
    except Exception as e:
        raise ValueError(f"Text-to-speech generation failed: {str(e)}")

def text_to_audio_base64(text: str, speaker_embedding_id: int = 0) -> Dict[str, Any]:
    """
    Convert text to speech and return as base64 (for JSON APIs).
    """
    try:
        audio_bytes = text_to_audio(text, speaker_embedding_id)
        
        # Verify audio bytes are valid before encoding
        if not audio_bytes or len(audio_bytes) == 0:
            raise ValueError("No audio data generated")
            
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        # Verify base64 encoding worked
        if not audio_base64:
            raise ValueError("Base64 encoding failed")
        
        return {
            "text": text,
            "audio_base64": audio_base64,
            "sample_rate": 24000,
            "format": "mp3",
            "model": "google_tts",
            "audio_size_bytes": len(audio_bytes),
            "base64_size": len(audio_base64)
        }
        
    except Exception as e:
        raise ValueError(f"Base64 audio generation failed: {str(e)}")
