import torch
import librosa
import numpy as np
import soundfile as sf

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
MODEL_PATH = "model.pth"
INPUT_AUDIO = "input.wav"
OUTPUT_AUDIO = "output.wav"

SAMPLE_RATE = 16000
TARGET_SR = 40000
F0_METHOD = "rmvpe"  # "rmvpe", "pm"
USE_F0 = True


# -------------------------------------------------
# LOAD AUDIO
# -------------------------------------------------
def load_audio(path):
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return audio


# -------------------------------------------------
# LOAD MODEL (PURE TORCH)
# -------------------------------------------------
def load_model(model_path, device):
    ckpt = torch.load(model_path, map_location=device)
    model = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.eval()
    return model


# -------------------------------------------------
# SIMPLE INFERENCE (NO PIPELINE, NO AUTORUN)
# -------------------------------------------------
@torch.no_grad()
def infer(model, audio, device):
    audio = torch.from_numpy(audio).float().to(device)
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)

    # Forward pass (generic, safe)
    out = model(audio)
    if isinstance(out, (list, tuple)):
        out = out[0]

    out = out.squeeze().cpu().numpy()
    return out


# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[INFO] Loading audio...")
    audio = load_audio(INPUT_AUDIO)

    print("[INFO] Loading model...")
    model = load_model(MODEL_PATH, device)

    print("[INFO] Running inference...")
    audio_out = infer(model, audio, device)

    print("[INFO] Saving output...")
    sf.write(OUTPUT_AUDIO, audio_out, TARGET_SR)

    print(f"[OK] Done → {OUTPUT_AUDIO}")


# -------------------------------------------------
# ENTRY
# -------------------------------------------------
if __name__ == "__main__":
    main()
