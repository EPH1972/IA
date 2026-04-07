"""
Vowel Dataset Creator
---------------------
1. Generates TTS audio from Spanish text using gTTS
2. Detects vowel segments via energy thresholding + formant (F1/F2) analysis
3. Exports Edu.json in the same format as alex.json

Output format:
    [{"vocal": "A", "start": "14.55", "end": "14.65"}, ...]

Dependencies:
    pip install gtts pydub librosa praat-parselmouth static-ffmpeg
    (static-ffmpeg bundles the ffmpeg binary; no separate install needed)
"""

import os
import json
import numpy as np
import librosa
import parselmouth
import static_ffmpeg
static_ffmpeg.add_paths()  
from gtts import gTTS
from pydub import AudioSegment
WAV_PATH  = "beppo.wav"
JSON_PATH = "Edu.json"

MIN_DURATION   = 0.04  
MAX_DURATION   = 0.35   
ENERGY_THRESH  = 0.12  

VOWEL_REFS = {
    "A": (750, 1250),
    "E": (500, 1900),
    "I": (300, 2300),
    "O": (500,  900),
    "U": (300,  800),
}

def generate_audio(text: str, wav_path: str) -> str:
    mp3_path = wav_path.replace(".wav", ".mp3")
    print(f"  Synthesising speech -> {mp3_path}")
    tts = gTTS(text=text, lang="es", slow=False)
    tts.save(mp3_path)
    print(f"  Converting MP3 -> {wav_path}")
    AudioSegment.from_mp3(mp3_path).export(wav_path, format="wav")
    os.remove(mp3_path)
    return wav_path

def classify_vowel(f1: float, f2: float) -> str | None:
    """Return closest Spanish vowel label or None if out of range."""
    if not (200 < f1 < 1200 and 600 < f2 < 3500):
        return None
    best, best_d = None, float("inf")
    for label, (r1, r2) in VOWEL_REFS.items():
        d = np.sqrt(((f1 - r1) / 250) ** 2 + ((f2 - r2) / 500) ** 2)
        if d < best_d:
            best_d, best = d, label
    return best if best_d < 3.0 else None


def detect_vowels(wav_path: str) -> list[tuple[float, float, str]]:
    """
    Energy-based segmentation followed by Praat formant classification.
    Returns list of (start_sec, end_sec, vowel_label).
    """
    y, sr = librosa.load(wav_path, sr=None, mono=True)
    snd   = parselmouth.Sound(wav_path)

    hop = int(sr * 0.010)   # 10 ms
    win = int(sr * 0.025)   # 25 ms

    rms      = librosa.feature.rms(y=y, frame_length=win, hop_length=hop)[0]
    rms_norm = rms / (rms.max() + 1e-9)
    times    = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)

    # --- find voiced segments ---
    voiced = rms_norm > ENERGY_THRESH
    segs: list[tuple[float, float]] = []
    in_seg, t0 = False, 0.0
    for i, v in enumerate(voiced):
        if v and not in_seg:
            in_seg, t0 = True, times[i]
        elif not v and in_seg:
            in_seg = False
            dur = times[i] - t0
            if MIN_DURATION <= dur <= MAX_DURATION:
                segs.append((t0, times[i]))
    if in_seg:
        segs.append((t0, times[-1]))

    # --- classify with formants ---
    formant = snd.to_formant_burg(
        time_step=0.010,
        max_number_of_formants=5,
        maximum_formant=5500,
        window_length=0.025,
        pre_emphasis_from=50,
    )

    results: list[tuple[float, float, str]] = []
    for s, e in segs:
        mid = (s + e) / 2
        f1  = formant.get_value_at_time(1, mid)
        f2  = formant.get_value_at_time(2, mid)
        if np.isnan(f1) or np.isnan(f2):
            continue
        label = classify_vowel(f1, f2)
        if label:
            results.append((s, e, label))
    return results

if __name__ == "__main__":
    print("=== Step 1: Generate audio ===")
    if os.path.exists(WAV_PATH):
        print(f"  Found existing {WAV_PATH}, skipping synthesis.")
    else:
        generate_audio(TEXT, WAV_PATH)

    print("\n=== Step 2: Detect vowels ===")
    segments = detect_vowels(WAV_PATH)
    print(f"  Detected {len(segments)} vowel candidates")

    print("\n=== Step 3: Save Edu.json ===")
    save_json(segments, JSON_PATH)

    print("\nDone.")
