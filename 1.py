import os
import sys
import torch
import numpy as np

now_dir = os.getcwd()
if now_dir not in sys.path:
    sys.path.append(now_dir)


def run_infer(
    model_path,
    index_path,
    input_audio,
    output_audio,
    f0_method="rmvpe",
    use_f0=True,
    index_rate=0.0,
):
    # IMPORT INSIDE FUNCTION (this fixes the error)
    from rvc.infer.pipeline import Pipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"

    vc = Pipeline(
        device=device,
        is_half=device == "cuda",
    )

    vc.load_model(model_path)
    vc.load_index(index_path)

    vc.convert(
        audio_path=input_audio,
        output_path=output_audio,
        f0_method=f0_method,
        use_f0=use_f0,
        index_rate=index_rate,
    )


if __name__ == "__main__":
    run_infer(
        model_path="model.pth",
        index_path="added.index",
        input_audio="input.wav",
        output_audio="output.wav",
        f0_method="rmvpe",
        use_f0=True,
        index_rate=0.0,
    )
