import os
import sys
import torch
import librosa
import soundfile as sf

# ---------------------------------------------------
# Path safety
# ---------------------------------------------------
ROOT = os.getcwd()
if ROOT not in sys.path:
    sys.path.append(ROOT)

# ---------------------------------------------------
# Audio loader
# ---------------------------------------------------
def load_audio(path):
    audio, _ = librosa.load(path, sr=16000, mono=True)
    return audio

# ---------------------------------------------------
# Inference
# ---------------------------------------------------
def run_infer(
    model_path,
    index_path,
    input_audio,
    output_audio,
    f0_method="rmvpe",
    use_f0=True,
    index_rate=0.0,
):
    # SAFE import (prevents circular import)
    from rvc.infer.pipeline import Pipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    is_half = device == "cuda"

    # Init pipeline (NO unsupported args)
    vc = Pipeline(device, is_half)

    # Load model (method name is stable)
    vc.load_model(model_path)

    # Load audio
    audio = load_audio(input_audio)

    # Disable index cleanly if missing
    if not index_path or not os.path.exists(index_path):
        index_rate = 0.0

    # -------- CORE CALL (compatible order) --------
    audio_out = vc.vc(
        0,                  # sid
        audio,              # audio
        0,                  # f0_up_key
        f0_method if use_f0 else "pm",
        index_rate,
        0.33,               # protect
    )

    # Save
    sf.write(output_audio, audio_out, vc.tgt_sr)
    print(f"[OK] Saved: {output_audio}")

# ---------------------------------------------------
# Entry
# ---------------------------------------------------
if __name__ == "__main__":

    MODEL_PATH = "model.pth"
    INDEX_PATH = "added.index"    # set None to disable
    INPUT_AUDIO = "input.wav"
    OUTPUT_AUDIO = "output.wav"

    run_infer(
        model_path=MODEL_PATH,
        index_path=INDEX_PATH,
        input_audio=INPUT_AUDIO,
        output_audio=OUTPUT_AUDIO,
        f0_method="rmvpe",
        use_f0=True,
        index_rate=0.0,
  )
