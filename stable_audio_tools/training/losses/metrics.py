import torch
import torchaudio
from pesq import pesq, NoUtterancesError

from torch.nn import functional as F
from torch import nn


class PESQMetric(nn.Module):
    def __init__(self, sample_rate: int):
        super().__init__()
        self.resampler = (
            torchaudio.transforms.Resample(sample_rate, 16000)
            if sample_rate != 16000 else None)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        inputs = inputs.detach().cpu().float()
        targets = targets.detach().cpu().float()

        if self.resampler is not None:
            inputs = self.resampler(inputs)
            targets = self.resampler(targets)

        inputs_np = inputs.numpy().astype("float64")
        targets_np = targets.numpy().astype("float64")
        batch_size = targets.shape[0]

        scores = []
        for i in range(batch_size):
            try:
                scores.append(pesq(16000, targets_np[i].reshape(-1), inputs_np[i].reshape(-1), "wb"))
            except NoUtterancesError:
                continue

        if not scores:
            return float("nan")
        return sum(scores) / len(scores)