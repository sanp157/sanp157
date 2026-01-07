import os
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
        source_audio = np.asarray(source_audio)
        target_audio = np.asarray(target_audio)

        frame_length1 = min(len(source_audio), max(2, source_rate // 100))
        hop_length1 = max(1, frame_length1 // 2)
        frame_length2 = min(len(target_audio), max(2, target_rate // 100))
        hop_length2 = max(1, frame_length2 // 2)

        rms1 = librosa.feature.rms(
            y=source_audio,
            frame_length=frame_length1,
            hop_length=hop_length1,
        )
        rms2 = librosa.feature.rms(
            y=target_audio,
            frame_length=frame_length2,
            hop_length=hop_length2,
        )

        rms1 = F.interpolate(
            torch.from_numpy(rms1).float().unsqueeze(0).unsqueeze(0),
            size=target_audio.shape[0],
            mode="linear",
            align_corners=False,
        ).squeeze()
        rms2 = F.interpolate(
            torch.from_numpy(rms2).float().unsqueeze(0).unsqueeze(0),
            size=target_audio.shape[0],
            mode="linear",
            align_corners=False,
        ).squeeze()

        eps = 1e-6
        rms1 = torch.maximum(rms1, torch.zeros_like(rms1) + eps)
        rms2 = torch.maximum(rms2, torch.zeros_like(rms2) + eps)

        factor = (torch.pow(rms1, 1 - rate) * torch.pow(rms2, rate - 1)).cpu().numpy()
        return target_audio * factor


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

        self.t_pad = int(self.sample_rate * self.x_pad)
        self.t_pad_tgt = int(self.tgt_sr * self.x_pad)
        self.t_pad2 = int(self.t_pad * 2)
        self.t_query = int(self.sample_rate * self.x_query)
        self.t_center = int(self.sample_rate * self.x_center)
        self.t_max = int(self.sample_rate * self.x_max)

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

        # Pitch shift only (no autotune)
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
    ):
        if file_index and os.path.exists(file_index) and index_rate > 0:
            index = faiss.read_index(file_index)
            big_npy = index.reconstruct_n(0, index.ntotal)
        else:
            index = big_npy = None

        audio = signal.filtfilt(bh, ah, audio)
        audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
        p_len = audio_pad.shape[0] // self.window

        sid = torch.tensor(sid, device=self.device).unsqueeze(0).long()

        if pitch_guidance:
            pitch, pitchf = self.get_f0(
                audio_pad,
                p_len,
                f0_method,
                pitch,
            )
            pitch = torch.tensor(pitch, device=self.device).unsqueeze(0).long()
            pitchf = torch.tensor(pitchf, device=self.device).unsqueeze(0).float()
        else:
            pitch = pitchf = None

        # HuBERT must see unpadded audio (author-correct)
        audio_input = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            feats = model(audio_input)["last_hidden_state"]
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
