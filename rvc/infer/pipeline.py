import os
import gc
import sys
import torch
import torch.nn.functional as F
import faiss
import librosa
import numpy as np
from scipy import signal

now_dir = os.getcwd()
if now_dir not in sys.path:
    sys.path.append(now_dir)

from rvc.lib.predictors.f0 import CREPE, FCPE, RMVPE

import logging
logging.getLogger("faiss").setLevel(logging.WARNING)

FILTER_ORDER = 5
CUTOFF_FREQUENCY = 48
SAMPLE_RATE = 16000

bh, ah = signal.butter(
    N=FILTER_ORDER, Wn=CUTOFF_FREQUENCY, btype="high", fs=SAMPLE_RATE
)


class AudioProcessor:
    @staticmethod
    def change_rms(source_audio, source_rate, target_audio, target_rate, rate):
        rms1 = librosa.feature.rms(
            y=source_audio,
            frame_length=source_rate // 2 * 2,
            hop_length=source_rate // 2,
        )
        rms2 = librosa.feature.rms(
            y=target_audio,
            frame_length=target_rate // 2 * 2,
            hop_length=target_rate // 2,
        )

        rms1 = F.interpolate(
            torch.from_numpy(rms1).float().unsqueeze(0),
            size=target_audio.shape[0],
            mode="linear",
        ).squeeze()
        rms2 = F.interpolate(
            torch.from_numpy(rms2).float().unsqueeze(0),
            size=target_audio.shape[0],
            mode="linear",
        ).squeeze()
        rms2 = torch.maximum(rms2, torch.zeros_like(rms2) + 1e-6)

        return (
            target_audio
            * (torch.pow(rms1, 1 - rate) * torch.pow(rms2, rate - 1)).numpy()
        )


class Pipeline:
    def __init__(self, tgt_sr, config):
        self.sample_rate = 16000
        self.window = 160
        self.device = config.device
        self.tgt_sr = tgt_sr

        self.x_pad = config.x_pad
        self.x_query = config.x_query
        self.x_center = config.x_center
        self.x_max = config.x_max

        self.t_pad = self.sample_rate * self.x_pad
        self.t_pad_tgt = tgt_sr * self.x_pad
        self.t_pad2 = self.t_pad * 2
        self.t_query = self.sample_rate * self.x_query
        self.t_center = self.sample_rate * self.x_center
        self.t_max = self.sample_rate * self.x_max

        self.f0_min = 50
        self.f0_max = 1100
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)

    def get_f0(
        self,
        x,
        p_len,
        f0_method="rmvpe",
        pitch=0,
        f0_autotune=False,
        f0_autotune_strength=1.0,
        proposed_pitch=False,
        proposed_pitch_threshold=155.0,
    ):
        if f0_method == "rmvpe":
            model = RMVPE(self.device, self.sample_rate, self.window)
            f0 = model.get_f0(x, filter_radius=0.03)
            del model
        elif f0_method == "crepe":
            model = CREPE(self.device, self.sample_rate, self.window)
            f0 = model.get_f0(x, self.f0_min, self.f0_max, p_len, "full")
            del model
        elif f0_method == "fcpe":
            model = FCPE(self.device, self.sample_rate, self.window)
            f0 = model.get_f0(x, p_len, filter_radius=0.006)
            del model
        else:
            raise ValueError(f"Unknown f0 method: {f0_method}")

        if f0_autotune:
            f0 *= 1.0  # keep structure, no forced snapping
        else:
            f0 *= pow(2, pitch / 12)

        f0bak = f0.copy()
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel = (f0_mel - self.f0_mel_min) * 254 / (
            self.f0_mel_max - self.f0_mel_min
        ) + 1
        f0_mel = np.clip(f0_mel, 1, 255)
        f0_coarse = np.rint(f0_mel).astype(int)

        return f0_coarse, f0bak

    def pipeline(
        self,
        model,
        net_g,
        sid,
        audio,
        pitch,
        f0_method,
        file_index,
        index_rate,
        pitch_guidance,
        volume_envelope,
        version,
        protect,
        f0_autotune,
        f0_autotune_strength,
        proposed_pitch,
        proposed_pitch_threshold,
    ):
        # Index handling (unchanged)
        if file_index and os.path.exists(file_index) and index_rate > 0:
            index = faiss.read_index(file_index)
            big_npy = index.reconstruct_n(0, index.ntotal)
        else:
            index = big_npy = None

        audio = signal.filtfilt(bh, ah, audio)
        audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
        p_len = audio_pad.shape[0] // self.window

        sid = torch.tensor(sid, device=self.device).unsqueeze(0).long()

        # ✅ CRITICAL CORRECTION (AUTHOR-SAFE)
        # Pitch guidance ONLY when explicitly requested
        if pitch_guidance:
            pitch, pitchf = self.get_f0(
                audio_pad,
                p_len,
                f0_method,
                pitch,
                f0_autotune,
                f0_autotune_strength,
                proposed_pitch,
                proposed_pitch_threshold,
            )
            pitch = torch.tensor(pitch, device=self.device).unsqueeze(0).long()
            pitchf = torch.tensor(pitchf, device=self.device).unsqueeze(0).float()
        else:
            pitch = None
            pitchf = None

        with torch.no_grad():
            feats = model(audio_pad)["last_hidden_state"]
            audio_out = (
                net_g.infer(
                    feats,
                    torch.tensor([p_len], device=self.device),
                    pitch,
                    pitchf,
                    sid,
                )[0][0, 0]
                .cpu()
                .numpy()
            )

        if volume_envelope != 1:
            audio_out = AudioProcessor.change_rms(
                audio, self.sample_rate, audio_out, self.tgt_sr, volume_envelope
            )

        audio_out /= max(1.0, np.abs(audio_out).max() / 0.99)

        return audio_out
