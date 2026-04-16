"""
Evaluate an unwrapped autoencoder model using PESQ or ViSQOL.

Usage:
    python evaluate_autoencoder.py \
        --model_config stable_audio_tools/configs/model_configs/autoencoders/stable_audio_2_0_vae.json \
        --ckpt_path exported_model.ckpt \
        --dataset_config stable_audio_tools/configs/dataset_configs/local_dac.json \
        --batch_size 2 \
        --metric pesq          # or visqol (requires building visqol from source)

Requires:
    pip install pesq argbind
    For visqol: see install.sh
"""

import argbind
import json
import os
import sys

import numpy as np
import scipy.stats
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm

from stable_audio_tools.data.dataset import create_dataloader_from_config
from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.models.utils import copy_state_dict, load_ckpt_state_dict


# ---------------------------------------------------------------------------
# Metric backends
# ---------------------------------------------------------------------------

class PESQBackend:
    SAMPLE_RATE = 16000

    def __init__(self):
        from pesq import pesq, NoUtterancesError
        self._pesq = pesq
        self._no_utt = NoUtterancesError

    def __call__(self, ref: np.ndarray, deg: np.ndarray) -> float:
        try:
            return self._pesq(self.SAMPLE_RATE, ref, deg, "wb")
        except self._no_utt:
            return float("nan")

    @property
    def name(self):
        return "PESQ"

    @property
    def score_label(self):
        return "Mean PESQ"

    @property
    def sample_rate(self):
        return self.SAMPLE_RATE


class ViSQOLBackend:
    SAMPLE_RATE = 48000

    def __init__(self, speech_mode: bool = False):
        from visqol import visqol_lib_py
        from visqol.pb2 import visqol_config_pb2

        config = visqol_config_pb2.VisqolConfig()
        config.audio.sample_rate = self.SAMPLE_RATE
        config.options.use_speech_scoring = speech_mode
        svr_model_name = ("lattice_tcditool_in_speech.tflite" if speech_mode
                          else "libsvm_nu_svr_model.txt")
        config.options.svr_model_path = os.path.join(
            os.path.dirname(visqol_lib_py.__file__), "model", svr_model_name)
        self._api = visqol_lib_py.VisqolApi()
        self._api.Create(config)

    def __call__(self, ref: np.ndarray, deg: np.ndarray) -> float:
        return self._api.Measure(ref, deg).moslqo

    @property
    def name(self):
        return "ViSQOL"

    @property
    def score_label(self):
        return "Mean MOS-LQO"

    @property
    def sample_rate(self):
        return self.SAMPLE_RATE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def gain_db(reference: np.ndarray, degraded: np.ndarray) -> float:
    return 10.0 * np.log10(np.mean(degraded ** 2) / (np.mean(reference ** 2) + 1e-12))


def mean_confidence_interval(data, confidence=0.95):
    a = np.array(data, dtype=np.float64)
    n = len(a)
    mean = np.mean(a)
    if n < 2:
        return mean, 0.0
    return mean, scipy.stats.sem(a) * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@argbind.bind(without_prefix=True)
def evaluate(
    model_config: str = "",
    ckpt_path: str = "",
    dataset_config: str = "",
    batch_size: int = 16,
    num_workers: int = 6,
    max_samples: int = 0,
    device: str = "cuda",
    metric: str = "pesq",
    speech_mode: bool = False,
    out_path: str = "",
    gain: float = 2.0,
):
    assert model_config, "--model_config is required"
    assert ckpt_path, "--ckpt_path is required"
    assert dataset_config, "--dataset_config is required"

    if metric == "pesq":
        backend = PESQBackend()
    elif metric == "visqol":
        backend = ViSQOLBackend(speech_mode=speech_mode)
    else:
        raise ValueError(f"Unknown metric: {metric}. Use 'pesq' or 'visqol'.")

    print(f"Using metric: {backend.name}")

    save_audio = bool(out_path)
    if save_audio:
        os.makedirs(os.path.join(out_path, "original"), exist_ok=True)
        os.makedirs(os.path.join(out_path, "reconstructed"), exist_ok=True)
        os.makedirs(os.path.join(out_path, "swapped"), exist_ok=True)

    with open(model_config) as f:
        model_cfg = json.load(f)

    with open(dataset_config) as f:
        dataset_cfg = json.load(f)
    dataset_cfg["random_crop"] = True
    dataset_cfg["drop_last"] = False

    print("Creating model from config ...")
    model = create_model_from_config(model_cfg)
    print(f"Loading checkpoint from {ckpt_path} ...")
    copy_state_dict(model, load_ckpt_state_dict(ckpt_path))
    model.to(device).eval().requires_grad_(False)
    print("Model ready.")

    data_loader = create_dataloader_from_config(
        dataset_cfg,
        batch_size=batch_size,
        num_workers=num_workers,
        sample_rate=model_cfg["sample_rate"],
        sample_size=model_cfg["sample_rate"] * 10,
        audio_channels=model_cfg.get("audio_channels", 1),
        shuffle=False,
    )

    metric_sr = backend.sample_rate
    resampler = (T.Resample(model_cfg["sample_rate"], metric_sr)
                 if model_cfg["sample_rate"] != metric_sr else None)

    scores = []
    swapped_scores = []
    recon_gain_dbs = []
    swapped_gain_dbs = []
    total_processed = 0
    effective_max = max_samples if max_samples > 0 else float("inf")

    pbar = tqdm(data_loader, desc="Evaluating", unit="batch")
    for batch in pbar:
        audio = batch[0].to(device)

        with torch.no_grad():
            latent = model.encode(audio)
            reconstructed = model.decode(latent)

            latent_gained = model.encode(audio * gain)
            latent_gained[:, 0:1, :] = latent[:, 0:1, :]
            swapped = model.decode(latent_gained)

        min_len = min(audio.shape[-1], reconstructed.shape[-1], swapped.shape[-1])
        audio_cpu = audio[..., :min_len].cpu()
        reconstructed_cpu = reconstructed[..., :min_len].cpu()
        swapped_cpu = swapped[..., :min_len].cpu()

        for i in range(audio_cpu.shape[0]):
            if total_processed >= effective_max:
                break

            ref = audio_cpu[i].mean(dim=0)
            deg = reconstructed_cpu[i].mean(dim=0)
            deg_sw = swapped_cpu[i].mean(dim=0)

            if resampler is not None:
                ref = resampler(ref.unsqueeze(0)).squeeze(0)
                deg = resampler(deg.unsqueeze(0)).squeeze(0)
                deg_sw = resampler(deg_sw.unsqueeze(0)).squeeze(0)

            ref_np = ref.numpy().astype(np.float64)
            deg_np = deg.numpy().astype(np.float64)
            deg_sw_np = deg_sw.numpy().astype(np.float64)

            try:
                score = backend(ref_np, deg_np)
                if not np.isnan(score):
                    scores.append(score)
            except Exception as e:
                print(f"\n[WARNING] {backend.name} failed on sample {total_processed} (reconstructed): {e}", file=sys.stderr)

            try:
                sw_score = backend(ref_np, deg_sw_np)
                if not np.isnan(sw_score):
                    swapped_scores.append(sw_score)
            except Exception as e:
                print(f"\n[WARNING] {backend.name} failed on sample {total_processed} (swapped): {e}", file=sys.stderr)

            recon_gain_dbs.append(gain_db(ref_np, deg_np))
            swapped_gain_dbs.append(gain_db(ref_np, deg_sw_np))

            total_processed += 1

            if save_audio:
                sr = model_cfg["sample_rate"]
                torchaudio.save(os.path.join(out_path, "original", f"{total_processed:05d}.wav"), audio_cpu[i], sr)
                torchaudio.save(os.path.join(out_path, "reconstructed", f"{total_processed:05d}.wav"), reconstructed_cpu[i], sr)
                torchaudio.save(os.path.join(out_path, "swapped", f"{total_processed:05d}.wav"), swapped_cpu[i], sr)

            if scores:
                pbar.set_postfix(
                    score=f"{np.mean(scores):.4f}",
                    swapped=f"{np.mean(swapped_scores):.4f}" if swapped_scores else "n/a",
                    r_dB=f"{np.mean(recon_gain_dbs):+.2f}",
                    sw_dB=f"{np.mean(swapped_gain_dbs):+.2f}",
                    n=len(scores),
                )

        if total_processed >= effective_max:
            break

    if not scores and not swapped_scores:
        print("\nNo scores were computed. Check your dataset / model.")
        return

    label = backend.score_label

    print("\n" + "=" * 60)

    if scores:
        mean, ci = mean_confidence_interval(scores, confidence=0.95)
        print(f"  {backend.name} Evaluation - Reconstructed  ({len(scores)} samples)")
        print("=" * 60)
        print(f"  {label:14s}:  {mean:.4f} +/- {ci:.4f}")
        print(f"  Std Dev      :  {np.std(scores):.4f}")
        print(f"  Min / Max    :  {np.min(scores):.4f} / {np.max(scores):.4f}")
        print("=" * 60)

    if recon_gain_dbs:
        gdb_mean, gdb_ci = mean_confidence_interval(recon_gain_dbs, confidence=0.95)
        print(f"  Gain dB vs Reference - Reconstructed  ({len(recon_gain_dbs)} samples)")
        print("=" * 60)
        print(f"  Mean Gain dB :  {gdb_mean:+.4f} +/- {gdb_ci:.4f}")
        print(f"  Std Dev      :  {np.std(recon_gain_dbs):.4f}")
        print(f"  Min / Max    :  {np.min(recon_gain_dbs):+.4f} / {np.max(recon_gain_dbs):+.4f}")
        print("=" * 60)

    if swapped_scores:
        sw_mean, sw_ci = mean_confidence_interval(swapped_scores, confidence=0.95)
        print(f"  {backend.name} Evaluation - Swapped Latent (gain={gain})  ({len(swapped_scores)} samples)")
        print("=" * 60)
        print(f"  {label:14s}:  {sw_mean:.4f} +/- {sw_ci:.4f}")
        print(f"  Std Dev      :  {np.std(swapped_scores):.4f}")
        print(f"  Min / Max    :  {np.min(swapped_scores):.4f} / {np.max(swapped_scores):.4f}")
        print("=" * 60)

    if swapped_gain_dbs:
        sw_gdb_mean, sw_gdb_ci = mean_confidence_interval(swapped_gain_dbs, confidence=0.95)
        print(f"  Gain dB vs Reference - Swapped Latent (gain={gain})  ({len(swapped_gain_dbs)} samples)")
        print("=" * 60)
        print(f"  Mean Gain dB :  {sw_gdb_mean:+.4f} +/- {sw_gdb_ci:.4f}")
        print(f"  Std Dev      :  {np.std(swapped_gain_dbs):.4f}")
        print(f"  Min / Max    :  {np.min(swapped_gain_dbs):+.4f} / {np.max(swapped_gain_dbs):+.4f}")
        print("=" * 60)


if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        evaluate()
