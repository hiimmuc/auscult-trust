"""Audio validation, preprocessing, inference and serialisation for the serving bundle (`api.bundle`).

Preprocessing follows training: mono, torchaudio resampling to 16 kHz, 1/16 s linear fade on the whole recording, every
segment cut or repeat-padded to 8 s, SC (if the pipeline has it) on the 8 s waveform, Kaldi fbank, frozen model, softmax.

Segments: annotated cycles when given (the unit the model was trained on), else sliding 8 s windows. The recording-level
screening view always uses the registered windows: max(1, floor(duration / 8)) windows evenly spread.

SC target spectrum: an ICBHI device name uses the coefficient saved with the unit (estimated from that device's unlabelled
official-test cycles, train cycles for LittC2SE); otherwise it is the mean spectrum of this recording's own segments.
"""

import io
import threading

import numpy as np
import soundfile as sf
import torch
from torchaudio import transforms as T

from src.evaluation.conformal import predict_sets
from src.evaluation.kauh_infer import images_of, load_frozen, predict_images
from src.evaluation.lodo_ttsc import load_args
from src.processing.correction import apply_spectrum_correction, mean_spectrum, spectrum_coefficients
from src.training.patchmix_cl.cycles import FADE_RATIO, cut_pad

SR, SEG_SEC = 16000, 8
MIN_SEC, MAX_SEC = 2.0, 90.0
MIN_SR, MAX_SR = 4000, 48000


class AudioError(ValueError):
    """Input audio fails validation; `code` is machine-readable."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def read_wav(data):
    """Validate WAV bytes. Returns (mono float32 at the native rate, info dict)."""
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise AudioError("not_wav", "File is not a RIFF/WAVE file.")
    try:
        info = sf.info(io.BytesIO(data))
        x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception as e:  # soundfile raises its own error types
        raise AudioError("unreadable", "WAV could not be decoded: {}".format(e)) from e
    meta = {"sample_rate": sr, "channels": x.shape[1], "duration_sec": round(x.shape[0] / sr, 3), "subtype": info.subtype}
    if x.shape[1] > 2:
        raise AudioError("channels", "Expected mono or stereo, got {} channels.".format(x.shape[1]))
    if not MIN_SR <= sr <= MAX_SR:
        raise AudioError("sample_rate", "Sample rate {} Hz is outside {}-{} Hz.".format(sr, MIN_SR, MAX_SR))
    if not MIN_SEC <= meta["duration_sec"] <= MAX_SEC:
        raise AudioError("duration", "Duration {:.1f} s is outside {:.0f}-{:.0f} s.".format(meta["duration_sec"], MIN_SEC, MAX_SEC))
    x = x.mean(1)
    if not np.isfinite(x).all():
        raise AudioError("non_finite", "Audio contains NaN or infinite samples.")
    peak = float(np.abs(x).max())
    if peak < 1e-4:
        raise AudioError("silent", "Audio is silent (peak below -80 dBFS).")
    meta.update(peak_dbfs=round(20 * np.log10(peak), 1), clipped_share=round(float((np.abs(x) > 0.999).mean()), 5))
    return x, meta

if not torch.cuda.is_available():
    # ponytail: the training code moves models and tensors with .cuda(); on a CPU host that becomes a no-op. Slower (about 20x), same maths.
    torch.nn.Module.cuda = lambda self, *a, **k: self
    torch.Tensor.cuda = lambda self, *a, **k: self


def to_model_rate(x, sr):
    """Resampled and faded (1, n) tensor at 16 kHz, as in training."""
    t = torch.from_numpy(np.ascontiguousarray(x))[None]
    if sr != SR:
        t = T.Resample(sr, SR)(t)
    fade = int(SR / FADE_RATIO)
    return T.Fade(fade_in_len=fade, fade_out_len=fade, fade_shape="linear")(t)


def windows(duration, win, hop):
    starts = np.arange(0, max(duration - win, 0) + 1e-9, hop)
    if duration > win and starts[-1] + win < duration - 0.5:  # cover the tail
        starts = np.append(starts, duration - win)
    return [(float(s), float(min(s + win, duration))) for s in starts]


def screen_windows(duration):
    n = max(1, int(duration // SEG_SEC))
    return [(float(s), float(min(s + SEG_SEC, duration))) for s in np.linspace(0, max(duration - SEG_SEC, 0), n)]


def cut(x, segs):
    """8 s repeat-padded waveforms (numpy) of the (start, end) segments of the (1, n) tensor `x`."""
    n = x.shape[1]
    return [cut_pad(x[:, min(int(s * SR), n) : min(int(e * SR), n)], SEG_SEC, SR, "repeat")[0].numpy() for s, e in segs]


def gcc_phat_lag(ref, rec, max_lead=0.25, max_lag=1.5, band=(100.0, 2000.0)):
    """Delay (s) of `rec` behind `ref` (both 16 kHz) by band-limited GCC-PHAT, and the normalised peak height."""
    n = 1 << int(np.ceil(np.log2(len(ref) + len(rec))))
    R, X = np.fft.rfft(ref, n), np.fft.rfft(rec, n)
    f = np.fft.rfftfreq(n, 1 / SR)
    g = X * np.conj(R)
    g = np.where((f >= band[0]) & (f <= band[1]), g / np.maximum(np.abs(g), 1e-12), 0)
    cc = np.fft.irfft(g, n)
    lags = np.r_[np.arange(0, int(max_lag * SR)), np.arange(-int(max_lead * SR), 0)]
    vals = cc[lags]
    k = int(np.argmax(vals))
    rest = vals[np.abs(lags - lags[k]) > int(0.005 * SR)]
    score = float((vals[k] - rest.mean()) / (rest.std() + 1e-12))  # peak height in SDs of the other lags
    return lags[k] / SR, score


def shift(x, lag):
    """Advance the (1, n) tensor by `lag` seconds (negative: delay), zero-filled, same length."""
    k = int(round(lag * SR))
    out = torch.zeros_like(x)
    if k >= 0:
        out[:, : x.shape[1] - k] = x[:, k:]
    else:
        out[:, -k:] = x[:, : x.shape[1] + k]
    return out


def align(ref16, take16):
    """Shift the (1, n) `take16` onto the (1, m) `ref16` and cut or zero-pad it to m samples. Returns (take, lag_sec, peak_score)."""
    lag, score = gcc_phat_lag(ref16[0].numpy(), take16[0].numpy())
    take16 = shift(take16, lag)[:, : ref16.shape[1]]
    if take16.shape[1] < ref16.shape[1]:
        take16 = torch.nn.functional.pad(take16, (0, ref16.shape[1] - take16.shape[1]))
    return take16, lag, score


class Pipeline:
    def __init__(self, name, entry, bundle_dir):
        self.name, self.entry = name, entry
        self.args = load_args(entry["unit"])
        self.model, self.classifier = load_frozen(self.args, entry["checkpoint"])
        self.thr = entry["thresholds"]
        self.sc = None
        if entry["spectrum_correction"]:
            st = np.load(bundle_dir / entry["sc"]["state"])
            self.sc = {k: st[k] for k in st.files}

    def coefficient(self, device, segs8, devices):
        """(SC coefficient, source label) for a device name or, when unknown, this recording's own segments."""
        if device in devices:
            d = devices.index(device)
            key = "test_coefficients_{}".format(d) if "test_coefficients_{}".format(d) in self.sc else "train_coefficients_{}".format(d)
            return self.sc[key], "saved:{}".format(key.split("_")[0])
        a = self.args
        coef = spectrum_coefficients({"x": mean_spectrum(segs8)}, reference=a.sc_reference, sc_mode=a.sc_mode,
                                     limit_freq_low=a.sc_limit_freq_low, limit_freq_high=a.sc_limit_freq_high,
                                     limit_freq_diff=a.sc_limit_freq_diff, reference_spectra={"ref": self.sc["reference"]})["x"]
        return coef, "estimated:recording"

    def run(self, segs8, screen8, device, devices):
        out = {"label": self.entry["label"], "sc": {"applied": False}}
        if self.sc is not None:
            coef, source = self.coefficient(device, segs8, devices)
            segs8 = [apply_spectrum_correction(w, coef).astype(np.float32) for w in segs8]
            screen8 = [apply_spectrum_correction(w, coef).astype(np.float32) for w in screen8]
            out["sc"] = {"applied": True, "mode": str(self.sc["sc_mode"]), "coefficient_source": source,
                         "gain_db": np.round(20 * np.log10(np.maximum(coef, 1e-8)), 2).tolist()}
        p = predict_images(images_of(segs8 + screen8, self.args), self.model, self.classifier).astype(np.float64)
        return out, p[: len(segs8)], p[len(segs8) :]


def serialise_segments(segs, p4, thr4, thr2, classes, classes2, truth=None):
    p2 = np.stack([p4[:, 0], 1 - p4[:, 0]], 1)
    s4, s2 = predict_sets(p4, thr4), predict_sets(p2, thr2)
    rows = []
    for i, (s, e) in enumerate(segs):
        r = {"index": i, "start": round(s, 3), "end": round(e, 3),
             "probs4": dict(zip(classes, np.round(p4[i], 4).tolist())), "pred4": classes[int(p4[i].argmax())],
             "set4": [c for c, m in zip(classes, s4[i]) if m],
             "probs2": dict(zip(classes2, np.round(p2[i], 4).tolist())), "pred2": classes2[int(p2[i].argmax())],
             "set2": [c for c, m in zip(classes2, s2[i]) if m]}
        if truth is not None:
            r["truth4"] = truth[i]
            r["truth2"] = "normal" if truth[i] == "normal" else "abnormal"
        rows.append(r)
    return rows


def serialise_screen(wins, p, thr):
    p_abn = 1 - p[:, 0]
    rec = np.array([[1 - p_abn.mean(), p_abn.mean()]])
    s = predict_sets(rec, thr)[0]
    triage = {(True, False): "no_referral", (False, True): "refer", (True, True): "uncertain", (False, False): "rerecord"}
    return {"p_abnormal": round(float(rec[0, 1]), 4), "set": [c for c, m in zip(("normal", "abnormal"), s) if m],
            "triage": triage[(bool(s[0]), bool(s[1]))], "aggregation": "mean",
            "windows": [{"start": round(a, 3), "end": round(b, 3), "p_abnormal": round(float(v), 4)} for (a, b), v in zip(wins, p_abn)]}


class Engine:
    """All pipelines of one bundle. Inference is serialised by one lock (one GPU, shared with training jobs)."""

    def __init__(self, manifest, bundle_dir):
        self.m = manifest
        self.pipelines = {n: Pipeline(n, e, bundle_dir) for n, e in manifest["pipelines"].items()}
        self.lock = threading.Lock()

    def infer(self, x16, duration, device=None, cycles=None, window_sec=8.0, hop_sec=4.0):
        """Run every pipeline on the (1, n) 16 kHz tensor. `cycles`: list of (start, end[, label]) in seconds."""
        if cycles:
            segs = [(float(c[0]), float(min(c[1], duration))) for c in cycles if c[0] < duration and c[1] > c[0]]
            truth = [c[2] for c in cycles if c[0] < duration and c[1] > c[0]] if all(len(c) > 2 for c in cycles) else None
            mode = "cycles"
        else:
            segs, truth, mode = windows(duration, window_sec, hop_sec), None, "windows"
        if not segs:
            raise AudioError("no_segments", "No segment falls inside the recording.")
        wins = screen_windows(duration)
        segs8, screen8 = cut(x16, segs), cut(x16, wins)
        res = {"segmentation": {"mode": mode, **({"window_sec": window_sec, "hop_sec": hop_sec} if mode == "windows" else {})},
               "pipelines": {}}
        with self.lock, torch.no_grad():
            for name, pl in self.pipelines.items():
                out, p4, ps = pl.run(segs8, screen8, device, self.m["devices"])
                out["segments"] = serialise_segments(segs, p4, pl.thr["cycle4"], pl.thr["cycle2"], self.m["classes"],
                                                     self.m["classes2"], truth)
                out["recording"] = serialise_screen(wins, ps, pl.thr["screen"])
                res["pipelines"][name] = out
        return res
