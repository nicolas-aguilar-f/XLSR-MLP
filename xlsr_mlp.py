"""Shared building blocks of the XLSR-MLP Spanish anti-spoofing detector.

Feature extraction, training and inference all import from this module, so
every stage applies exactly the same audio preprocessing.

Labels:  0 = bona fide (real speech),  1 = spoof (synthetic speech)
"""
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

XLSR_MODEL = "facebook/wav2vec2-xls-r-300m"
SAMPLE_RATE = 16000
MAX_SECONDS = 15        # longer audio is truncated
MIN_SAMPLES = 400       # XLS-R needs 25 ms of audio to produce one frame
SNR_DB = 36.0           # SNR of the noise injected by neutralize()
LABELS = ("bonafide", "spoof")

# Checkpoint keys. They keep the (Spanish) names the released model was saved
# with, so checkpoints produced by train_mlp.py stay interchangeable with it.
CKPT_MODEL = "modelo"       # MLP state_dict
CKPT_MEAN = "media"         # train-set mean of the embeddings
CKPT_STD = "desv"           # train-set standard deviation of the embeddings
CKPT_THRESHOLD = "umbral"   # decision threshold (EER point on validation)
CKPT_VAL_EER = "eer_val"    # validation EER, in %


def get_device(name="auto"):
    """Resolve "auto" to CUDA when it is available, otherwise to CPU."""
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def load_audio(path):
    """Return the waveform as a 1-D mono tensor at 16 kHz, cut to 15 s."""
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    wave = torch.from_numpy(np.ascontiguousarray(data.T))     # (channels, samples)
    if sr != SAMPLE_RATE:
        wave = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(wave)
    if wave.shape[0] > 1:
        wave = wave.mean(dim=0, keepdim=True)
    return wave.squeeze(0)[:SAMPLE_RATE * MAX_SECONDS]


def neutralize(wave, generator=None):
    """Add white noise at a fixed SNR to mask the original noise floor.

    This keeps the classifier from learning recording conditions (silence,
    microphone noise) instead of synthesis artifacts.
    """
    rms = wave.pow(2).mean().sqrt() + 1e-12
    noise = torch.randn(wave.shape, generator=generator, dtype=wave.dtype)
    noise = noise / (noise.pow(2).mean().sqrt() + 1e-12)
    return wave + noise * rms * 10.0 ** (-SNR_DB / 20.0)


class XLSREmbedder:
    """Map waveforms to 2048-D vectors: [mean, std] of the last XLS-R layer."""

    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(XLSR_MODEL)
        self.model = Wav2Vec2Model.from_pretrained(
            XLSR_MODEL, use_safetensors=True).to(self.device).eval()

    @torch.no_grad()
    def __call__(self, waves):
        """List of 1-D waveforms -> (B, 2048) float32 NumPy array."""
        inputs = self.processor([w.numpy() for w in waves],
                                sampling_rate=SAMPLE_RATE, return_tensors="pt",
                                padding=True, return_attention_mask=True)
        values = inputs.input_values.to(self.device)
        mask = inputs.attention_mask.to(self.device)

        frames = self.model(values, attention_mask=mask).last_hidden_state
        # Frame-level mask: without it the statistics of the short clips of a
        # padded batch would include the padding and get diluted.
        m = self.model._get_feature_vector_attention_mask(
            frames.shape[1], mask).unsqueeze(-1).float()

        n = m.sum(dim=1).clamp(min=1.0)
        mean = (frames * m).sum(dim=1) / n
        std = (((frames - mean.unsqueeze(1)) ** 2 * m).sum(dim=1) / n
               ).clamp(min=1e-8).sqrt()
        return torch.cat([mean, std], dim=1).cpu().numpy()


class MLPClassifier(nn.Module):
    """2048 -> 256 -> 2 classifier (525,570 parameters)."""

    def __init__(self, input_dim=2048, hidden_dim=256, num_classes=2):
        super().__init__()
        # "red" (Spanish for "network") matches the keys of the checkpoint.
        self.red = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.red(x)


class SpoofDetector:
    """End-to-end detector: audio file -> bona fide / spoof decision.

    The checkpoint bundles the MLP weights, the train-set statistics used to
    standardize the embeddings and the decision threshold.
    """

    def __init__(self, checkpoint, device="auto"):
        self.device = get_device(device)
        # weights_only=False because the checkpoint stores NumPy arrays:
        # only load checkpoints you trust.
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.mlp = MLPClassifier()
        self.mlp.load_state_dict(ckpt[CKPT_MODEL])
        self.mlp.to(self.device).eval()
        self.mean = torch.as_tensor(ckpt[CKPT_MEAN], dtype=torch.float32,
                                    device=self.device)
        self.std = torch.as_tensor(ckpt[CKPT_STD], dtype=torch.float32,
                                   device=self.device)
        self.threshold = float(ckpt[CKPT_THRESHOLD])
        self.embedder = XLSREmbedder(self.device)

    @torch.no_grad()
    def score(self, path, seed=1234):
        """Return the spoof score (softmax of the synthetic class) of a file."""
        wave = load_audio(path)
        if wave.numel() < MIN_SAMPLES:
            raise ValueError("audio is too short (minimum 25 ms)")
        # A fresh generator per file: the same audio always gets the same
        # noise, and therefore the same score.
        wave = neutralize(wave, torch.Generator().manual_seed(seed))
        x = torch.from_numpy(self.embedder([wave])).to(self.device)
        logits = self.mlp((x - self.mean) / self.std)
        return torch.softmax(logits, dim=1)[0, 1].item()

    def predict(self, path, seed=1234):
        """Classify one audio file and return the result as a dict."""
        score = self.score(path, seed)
        return {"file": str(path),
                "label": LABELS[int(score >= self.threshold)],
                "spoof_score": score,
                "threshold": self.threshold}
