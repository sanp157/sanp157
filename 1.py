# rvc/realtime/pipeline.py

import numpy as np
import torch

from rvc.infer.pipeline import AudioProcessor
from rvc.realtime.core import AUDIO_SAMPLE_RATE


class RealtimePipeline:
    """
    Realtime audio processing pipeline.
    Autotune has been intentionally removed to avoid
    import errors and pitch conflicts.
    """

    def __init__(
        self,
        model,
        index=None,
        device="cuda",
        use_f0=True,
        f0_method="rmvpe",
        index_rate=0.0,
        protect=0.33,
        resample_sr=0,
    ):
        self.model = model
        self.index = index
        self.device = device

        self.audio_processor = AudioProcessor(
            model=model,
            index=index,
            device=device,
            use_f0=use_f0,
            f0_method=f0_method,
            index_rate=index_rate,
            protect=protect,
            resample_sr=resample_sr,
        )

    @torch.no_grad()
    def process(self, audio: np.ndarray) -> np.ndarray:
        """
        Process realtime audio frame.
        """
        if audio is None or len(audio) == 0:
            return audio

        audio = audio.astype(np.float32)

        output = self.audio_processor.process_audio(
            audio,
            sample_rate=AUDIO_SAMPLE_RATE,
        )

        return output
