import os
import sys
import time
import torch
import librosa
import logging
import traceback
import numpy as np
import soundfile as sf
import noisereduce as nr
from pedalboard import (
    Pedalboard,
    Chorus,
    Distortion,
    Reverb,
    PitchShift,
    Limiter,
    Gain,
    Bitcrush,
    Clipping,
    Compressor,
    Delay,
)

now_dir = os.getcwd()
# Keep current working directory on sys.path only if necessary for local imports
if now_dir not in sys.path:
    sys.path.append(now_dir)

from rvc.infer.pipeline import Pipeline as VC
from rvc.lib.utils import load_audio_infer, load_embedding
from rvc.lib.tools.split_audio import process_audio, merge_audio
from rvc.lib.algorithm.synthesizers import Synthesizer
from rvc.configs.config import Config

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("faiss").setLevel(logging.WARNING)
logging.getLogger("faiss.loader").setLevel(logging.WARNING)


class VoiceConverter:
    def __init__(self):
        self.config = Config()
        self.hubert_model = None
        self.last_embedder_model = None
        self.tgt_sr = None
        self.net_g = None
        self.vc = None
        self.cpt = None
        self.version = None
        self.n_spk = None
        self.use_f0 = None
        self.loaded_model = None

    def load_hubert(self, embedder_model: str, embedder_model_custom: str = None):
        self.hubert_model = load_embedding(embedder_model, embedder_model_custom)
        self.hubert_model = self.hubert_model.to(self.config.device).float()
        self.hubert_model.eval()

    @staticmethod
    def remove_audio_noise(data, sr, reduction_strength=0.7):
        try:
            return nr.reduce_noise(y=data, sr=sr, prop_decrease=reduction_strength)
        except Exception:
            return data

    @staticmethod
    def convert_audio_format(input_path, output_path, output_format):
        # Use pydub/ffmpeg to support formats like mp3. Fallback to soundfile WAV if conversion fails.
        base, _ = os.path.splitext(output_path)
        out_path = output_path
        fmt = (output_format or "").lower()

        # If target is WAV, just return the WAV path (ensure proper extension)
        if fmt == "wav":
            return out_path

        # Try pydub if available (uses ffmpeg/avlib underneath)
        try:
            from pydub import AudioSegment

            audio_seg = AudioSegment.from_file(input_path)
            audio_seg.export(out_path, format=fmt)
            return out_path
        except Exception:
            pass

        # Try ffmpeg CLI if available
        try:
            import subprocess

            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                input_path,
                out_path,
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return out_path
        except Exception:
            pass

        # Fallback: write WAV using soundfile and return WAV path
        try:
            audio, sr = librosa.load(input_path, sr=None)
            wav_path = f"{base}.wav"
            sf.write(wav_path, audio, sr, format="WAV")
            return wav_path
        except Exception:
            # As last resort, return the original input_path
            return input_path

    @staticmethod
    def post_process_audio(audio_input, sample_rate, **kwargs):
        board = Pedalboard()
        if kwargs.get("reverb", False):
            board.append(Reverb())
        if kwargs.get("pitch_shift", False):
            board.append(PitchShift(semitones=kwargs.get("pitch_shift_semitones", 0)))
        if kwargs.get("limiter", False):
            board.append(Limiter())
        if kwargs.get("gain", False):
            board.append(Gain(gain_db=kwargs.get("gain_db", 0)))
        return board(audio_input, sample_rate)

    def convert_audio(
        self,
        audio_input_path: str,
        audio_output_path: str,
        model_path: str,
        index_path: str,
        pitch: int = 0,
        f0_method: str = "rmvpe",
        index_rate: float = 0.0,
        volume_envelope: float = 1.0,
        protect: float = 0.5,
        hop_length: int = 128,
        split_audio: bool = False,
        f0_autotune: bool = False,
        f0_autotune_strength: float = 1,
        embedder_model: str = "contentvec",
        embedder_model_custom: str = None,
        clean_audio: bool = False,
        clean_strength: float = 0.5,
        export_format: str = "WAV",
        post_process: bool = False,
        resample_sr: int = 0,
        sid: int = 0,
        proposed_pitch: bool = False,
        proposed_pitch_threshold: float = 155.0,
        **kwargs,
    ):
        if not model_path:
            return

        self.get_vc(model_path, sid)

        try:
            start_time = time.time()

            # HuBERT input MUST be 16k (correct)
            audio = load_audio_infer(audio_input_path, 16000)
            audio_max = np.abs(audio).max() / 0.95
            if audio_max > 1:
                audio /= audio_max

            # reload hubert if embedder changed (include custom path in comparison)
            embedder_key = (embedder_model, embedder_model_custom)
            if not self.hubert_model or embedder_key != self.last_embedder_model:
                self.load_hubert(embedder_model, embedder_model_custom)
                self.last_embedder_model = embedder_key

            file_index = (index_path or "").strip().replace("trained", "added")

            # LOCK target SR to model native SR (40k)
            self.tgt_sr = self.cpt["config"][-1]

            if split_audio:
                chunks, intervals = process_audio(audio, 16000)
            else:
                chunks = [audio]

            converted_chunks = []
            for c in chunks:
                audio_opt = self.vc.pipeline(
                    model=self.hubert_model,
                    net_g=self.net_g,
                    sid=sid,
                    audio=c,
                    pitch=0,                    # HARD LOCK (no pitch forcing)
                    f0_method="rmvpe",           # LOCK to RMVPE
                    file_index=file_index,
                    index_rate=index_rate,       # SAFE default
                    pitch_guidance=False,        # CRITICAL FIX
                    volume_envelope=volume_envelope,
                    version=self.version,
                    protect=protect,
                    f0_autotune=False,           # DISABLED
                    f0_autotune_strength=0,
                    proposed_pitch=False,
                    proposed_pitch_threshold=proposed_pitch_threshold,
                )
                converted_chunks.append(audio_opt)

            if split_audio:
                audio_opt = merge_audio(
                    chunks, converted_chunks, intervals, 16000, self.tgt_sr
                )
            else:
                audio_opt = converted_chunks[0]

            if clean_audio:
                audio_opt = self.remove_audio_noise(
                    audio_opt, self.tgt_sr, clean_strength
                )

            if post_process:
                audio_opt = self.post_process_audio(
                    audio_input=audio_opt,
                    sample_rate=self.tgt_sr,
                    **kwargs,
                )

            # ensure proper extension and format when writing
            base_out, _ = os.path.splitext(audio_output_path)
            wav_out = f"{base_out}.wav"
            sf.write(wav_out, audio_opt, self.tgt_sr, format="WAV")

            final_output = wav_out
            if export_format and export_format.upper() != "WAV":
                # attempt to convert to requested format (e.g., mp3)
                target_out = f"{base_out}.{export_format.lower()}"
                converted = self.convert_audio_format(wav_out, target_out, export_format)
                final_output = converted or wav_out

            logging.info(
                f"Done in {time.time() - start_time:.2f}s → {final_output}"
            )
        except Exception as e:
            logging.exception("Error during conversion: %s", e)

    def get_vc(self, weight_root, sid):
        if not self.loaded_model or self.loaded_model != weight_root:
            self.load_model(weight_root)
            if self.cpt is not None:
                self.setup_network()
                self.setup_vc_instance()
                self.loaded_model = weight_root

    def load_model(self, weight_root):
        if not os.path.isfile(weight_root):
            self.cpt = None
            return
        # torch.load in newer PyTorch supports weights_only; fall back for older versions
        try:
            self.cpt = torch.load(weight_root, map_location="cpu", weights_only=True)
        except TypeError:
            self.cpt = torch.load(weight_root, map_location="cpu")

    def setup_network(self):
        self.tgt_sr = self.cpt["config"][-1]   # 40000
        self.cpt["config"][-3] = self.cpt["weight"]["emb_g.weight"].shape[0]
        self.use_f0 = self.cpt.get("f0", 1)
        self.version = self.cpt.get("version", "v2")

        text_enc_hidden_dim = 768 if self.version == "v2" else 256
        self.net_g = Synthesizer(
            *self.cpt["config"],
            use_f0=self.use_f0,
            text_enc_hidden_dim=text_enc_hidden_dim,
            vocoder=self.cpt.get("vocoder", "HiFi-GAN"),
        )
        # remove enc_q only if present
        if hasattr(self.net_g, "enc_q"):
            del self.net_g.enc_q
        self.net_g.load_state_dict(self.cpt["weight"], strict=False)
        self.net_g = self.net_g.to(self.config.device).float()
        self.net_g.eval()

    def setup_vc_instance(self):
        self.vc = VC(self.tgt_sr, self.config)
        self.n_spk = self.cpt["config"][-3]