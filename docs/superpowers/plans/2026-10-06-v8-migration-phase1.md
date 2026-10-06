# V8 Migration Phase 1 (CLAUDE.md pointer + A1 wiring + calibration/eval split) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Point CLAUDE.md at proposal v8, wire A1 spectrum correction into `src/features.py` (domain-aware, source stats for calibration / target stats for test), fix the wrong A2 docstring line, and add the Stage 2 calibration/evaluation patient split required before any E1 run.

**Architecture:** `src/shift/correction.py` already has correct, tested A1 math (`mean_spectrum`, `spectrum_coefficients`, `apply_spectrum_correction`); it is just not called from the extraction pipeline. `src/features.py` gains an optional `correction` config block: when present, coefficients are computed once per device from the cycle table's own `device` column and applied to each waveform before it reaches the encoder. `src/data/splits.py` gains a function that takes the "cal" output of `device_holdout_official` (official test patients of the non-held-out devices) and repeatedly (20x) splits it into disjoint calibration/evaluation patient groups, stratified by device.

**Tech Stack:** Python, numpy, librosa, pandas, pytest (`uv run --no-project --python .venv/bin/python python -m pytest -q tests`).

**Spec:** `/home/namdp15/Workspace/Personal/KHKT-CVA-2026/docs/research-proposal-v8-en.md` (background); the concrete task list is the user's pasted Vietnamese spec in this conversation. This plan implements items 1-4 and 10 of that list only — see "Out of scope" below.

## Global Constraints

- Patient-disjoint splits always (CLAUDE.md).
- Never edit `data/raw/` or committed split files.
- Calibration clips use source-domain statistics; test clips use target-domain statistics (spec item 4).
- Bump any feature-cache `name` whenever preprocessing changes (features.py docstring rule) — corrected-feature configs must use a new `name`, never overwrite an existing cache.
- Code comments and docs in English. Docstrings Google style.
- Tune nothing on test. No test-set selection.

## Review Focus

- A config without a `correction` block must behave byte-identically to today (no accidental regression for every existing `configs/extract_*.yaml`).
- Coefficients must be computed from calibration-side (source) devices only, never leak target-device waveforms into the reference spectrum used to correct calibration data itself.
- `device_holdout_official`'s `cal` output can have as few as 1-2 patients per device for the rarer devices (e.g. Litt3200 test only 461 cycles) — the new split function must not crash or silently produce an empty stratum when a device has only one patient.
- Resuming an interrupted `extract()` run with `correction` enabled must still resume correctly (the `done.npy` / cache-key logic is untouched, but corrected waveforms must be deterministic across resumed runs, not re-randomized).
- `spectrum_coefficients`'s reference-mean choice (arithmetic vs geometric) changes results silently if the caller forgets to pick one — the function must require an explicit choice once the new parameter lands, not default silently to the wrong one from [1]/[3].

---

## Out of scope (left for follow-up plans)

The user's full spec has 11 items; only items the spec itself marks as blocking (10 and 4) plus the cheap doc fixes (1-3) are planned here. Not planned, because each needs its own design decision before tasks can be written without placeholders:

- **A2 wiring** (item 4, second half): A2 operates on log-mel statistics, but no current `Encoder` exposes an intermediate log-mel array to `features.py` — each encoder computes mel internally with its own preprocessing. Wiring A2 needs a new `Encoder.logmel(wave)` hook or equivalent, decided per-encoder (OPERA's `bn0` note in the correction.py docstring suggests the encoders are not uniform here). Needs its own plan.
- **`spectrum_coefficients` arithmetic/geometric reference option**: small, but depends on first checking which papers' numbers ([1] vs [3]) the project wants to reproduce — a user decision, not an engineering one. Flagged in Task 2 below as a one-line follow-up once that's picked.
- **Stage1 patchmix_cl changes** (items 5-6, P1-P9 variants): separate codebase (`baselines/patchmix_cl`), ~9 experiment variants, each needing its own config and flag — a multi-day effort, own plan.
- **Handcrafted features** (item 7, `src/features_handcrafted.py`) and **new eval metrics** (item 8, `src/eval/metrics.py`, `shift.py` additions, V4-oracle): new modules with open numeric-design questions (exact MFCC settings, MMD permutation count).
- **KAUH 5-fold patient rotation** (item 9): independent of items 4/10, own plan.
- **Full model -> conformal bridge** (item 10, second half: exporting softmax/embeddings for the fully fine-tuned Patch-Mix model): blocked on Stage 1 patchmix work above.
- **V5 k-shot in patients** (item 10, third part) and **pre-registration doc** (item 11): depend on the metrics/variant work above being settled first.

---

## File Structure

- Modify `CLAUDE.md` — Status line, hypothesis list, TODO line (Task 1).
- Modify `src/shift/correction.py` — fix the A2 docstring line (Task 2).
- Modify `src/features.py` — add `correction` config support (Task 3).
- Modify `src/data/splits.py` — add `calibration_eval_splits` (Task 4).
- Modify `tests/test_correction.py`, `tests/test_features.py`, `tests/test_core.py` — new/updated tests for the above.

---

### Task 1: Point CLAUDE.md at v8 and update TODO

**Files:**
- Modify: `CLAUDE.md:3-6` (Status line), `:71` (hypotheses H1-H4), TODO line in Status block

**Interfaces:** None (doc-only).

- [ ] **Step 1: Edit the Status line**

In `CLAUDE.md`, replace the current `## Status` first line:

```
Main line: proposal A+B v7 (`../docs/research-proposal-AB-gain-decomposition-v7-en.md`). Fallback: proposal H v6 (`../docs/research-proposal-H-fm-device-benchmark-v6-en.md`), public data only. v4 (`research-proposal-lung-sound-v4-en.md`) is superseded; its phase and short-window experiments are archived as reported negative or dataset-specific results.
```

with:

```
Main line: proposal v8 (`../docs/research-proposal-v8-en.md`), unified, replaces A+B v6/v7 and H v6. v4 (`research-proposal-lung-sound-v4-en.md`) is superseded; its phase and short-window experiments are archived as reported negative or dataset-specific results.
```

- [ ] **Step 2: Replace the H1-H4 hypothesis list**

In the `## Project` section, replace:

```
- H1: on the phantom, a gain predicted from measured H(f) explains >= 0.8 of the paired log-mel shift (relative to the placement-noise ceiling).
- H2: under device shift, split-conformal coverage falls below nominal; gain-based correction (A1, A2) closes part of the deficit, in proportion to the H1 explained fraction.
- H3: stethoscope-matched IR augmentation closes more deficit than generic microphone IRs (needs >= 3 distinct hardware).
- H4: fine-tuning raises ICBHI Score but also device decodability and coverage deficit, versus the frozen encoder.
```

with:

```
- H1: on the phantom, a gain predicted from measured H(f) explains >= 0.8 of the paired log-mel shift (relative to the placement-noise ceiling).
- H-diag: under device shift, split-conformal coverage falls below nominal, and part of the deficit comes from the device rather than the change in class proportions.
- H3: stethoscope-matched IR augmentation closes more deficit than generic microphone IRs (needs >= 3 distinct hardware).
- H-inv: gain-invariant fusion of handcrafted and deep features reduces device decodability versus deep-only features, without lowering ICBHI Score outside noise.
- F4 (finding, not hypothesis): fine-tuning raised ICBHI Score but also raised device decodability and coverage deficit, versus the frozen encoder (v4 result, kept as reported negative finding under v8, not re-tested as H4).
```

- [ ] **Step 3: Remove the leak question from TODO**

In the `## Status` TODO line, find and delete the clause about the pretraining-leak question (search for the sentence mentioning OPERA pretraining leakage in the TODO list) — read the current TODO line first with `grep -n "TODO" CLAUDE.md` to get its exact current wording before editing, since the file may have moved since this plan was written.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: point CLAUDE.md at proposal v8, update hypotheses"
```

---

### Task 2: Fix A2 docstring line in correction.py

**Files:**
- Modify: `src/shift/correction.py:4`

**Interfaces:** None (docstring only, no behavior change).

- [ ] **Step 1: Edit the docstring**

Current line 4:

```
A2  ISA: per-mel-bin moment matching of log-mel on unlabelled target clips, no device id needed at test time
```

Replace with:

```
A2  ISA: per-mel-bin moment matching of log-mel on unlabelled target clips. Needs a domain boundary (which clips
    are source vs target) to compute separate statistics, but no device *label* beyond that boundary.
```

- [ ] **Step 2: Run existing correction tests to confirm no behavior change**

Run: `uv run --no-project --python .venv/bin/python python -m pytest -q tests/test_correction.py`
Expected: PASS (all 4 existing tests, unchanged)

- [ ] **Step 3: Commit**

```bash
git add src/shift/correction.py
git commit -m "docs: correct A2 docstring, it needs a source/target domain boundary"
```

---

### Task 3: Wire A1 into `src/features.py`

**Files:**
- Modify: `src/features.py`
- Test: `tests/test_features.py`

**Interfaces:**
- Consumes: `src.shift.correction.mean_spectrum(waves, n_fft, hop)`, `spectrum_coefficients(device_spectra, source_devices=None)`, `apply_spectrum_correction(wave, coef, n_fft, hop)` (all exist, unchanged signatures).
- Produces: `extract(cfg, limit=None)` now reads an optional `cfg["correction"]` dict `{"source_devices": [...], "n_fft": 1024, "hop": 512}`. When present and `cfg.get("dataset") != "kauh"` (KAUH has no ICBHI `device` column), every waveform is spectrum-corrected before `enc.run`, using the coefficient of its own row's `device` value computed from the *full* cycle table's devices (so calibration rows self-correct against the chosen source devices, and non-source "target" rows get the same treatment — there's only one coefficient table, consistent with the test in Step 1).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_features.py`:

```python
def test_extract_with_correction_applies_spectrum_correction(tmp_path, monkeypatch):
    import librosa

    from src.shift import correction

    _cycles(tmp_path)  # single Meditron recording, reuse existing helper in this file
    # Add a second recording on a different device so there are 2 devices to correct between.
    from scipy.io import wavfile
    rng = np.random.default_rng(0)
    gain = 1 + 0.8 * np.sin(np.linspace(0, 3, 513))
    wave = rng.normal(size=16000 * 6).astype(np.float32)
    corrected_in = librosa.istft(librosa.stft(wave, n_fft=1024, hop_length=512) * gain[:, None], hop_length=512, length=len(wave))
    wavfile.write(tmp_path / "102_1b1_Al_sc_AKGC417L.wav", 16000, (corrected_in * 1000).astype(np.int16))
    (tmp_path / "102_1b1_Al_sc_AKGC417L.txt").write_text("0.0 2.0 0 0\n")

    seen_waves = {}

    class _RecordingStub(_Stub):
        def run(self, wave, keep, token_layer):
            seen_waves.setdefault("waves", []).append(np.asarray(wave))
            return super().run(wave, keep, token_layer)

    monkeypatch.setattr(encoders, "load", lambda name: _RecordingStub())
    cfg = {
        "icbhi_root": str(tmp_path), "cache_root": str(tmp_path / "cache"), "name": "stub_corrected",
        "encoder": "stub", "correction": {"source_devices": ["Meditron"]},
    }
    features.extract(cfg)
    # The AKGC417L row's waveform must have been spectrum-corrected (no longer equal to the raw input).
    akg_wave = seen_waves["waves"][1][: 16000 * 2]  # first cycle of the second recording (0.0-2.0s)
    assert not np.allclose(akg_wave, corrected_in[: 16000 * 2], atol=1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-project --python .venv/bin/python python -m pytest -q tests/test_features.py::test_extract_with_correction_applies_spectrum_correction -v`
Expected: FAIL (`KeyError: 'correction'` is not read, or `AssertionError` because waveform is unchanged)

- [ ] **Step 3: Implement in `src/features.py`**

Add the import at the top (after existing `from src.data.kauh import windows as kauh_windows`):

```python
from src.shift.correction import apply_spectrum_correction, mean_spectrum, spectrum_coefficients
```

Add a helper function above `extract`:

```python
def _device_coefficients(df, cfg, raw_loader):
    """Compute A1 coefficients per device in `df` from one waveform per device (first row seen).

    Args:
        df: Cycle table with a `device` column.
        cfg: Extraction config; reads `cfg["correction"]["source_devices"]`, `n_fft`, `hop`.
        raw_loader: Callable wav path -> waveform (reuses `load_wav` + `cycle_wave` so correction sees the same
            audio the encoder would have seen).

    Returns:
        Dict device -> coefficients array from `spectrum_coefficients`.
    """
    cc = cfg["correction"]
    n_fft, hop = cc.get("n_fft", 1024), cc.get("hop", 512)
    waves_by_device = {}
    for r in df.itertuples():
        waves_by_device.setdefault(r.device, []).append(raw_loader(r))
    spectra = {d: mean_spectrum(ws, n_fft, hop) for d, ws in waves_by_device.items()}
    return spectrum_coefficients(spectra, source_devices=cc.get("source_devices")), n_fft, hop
```

Modify `extract` to build the coefficients once (only for non-KAUH datasets, since `correction` keys off the ICBHI `device` column) and apply per row:

```python
def extract(cfg, limit=None):
    """Embed every row of the cycle table once and write the cache of `cfg`. Rows already marked done are skipped."""
    df = kauh_windows(kauh_files(cfg["kauh_root"])) if cfg.get("dataset") == "kauh" else cycle_table(cfg["icbhi_root"])
    keys = [cache_key(r) for r in df.itertuples()]
    out = Path(cfg["cache_root"]) / cfg["name"]
    if (out / "keys.json").exists():
        assert json.loads((out / "keys.json").read_text()) == keys, "cache rows do not match the cycle table"
    enc = encoders.load(cfg["encoder"])
    keep, tl = list(cfg.get("keep_layers", [enc.n_layers])), cfg.get("token_layer")
    layers, tokens, done = open_arrays(out, len(df), (len(keep), N_FRAMES, enc.dim), enc.token_shape)
    arrays = [a for a in (layers, tokens) if a is not None]
    (out / "keys.json").write_text(json.dumps(keys))
    (out / "meta.json").write_text(json.dumps({"encoder": cfg["encoder"], "keep_layers": keep, "token_layer": tl}))
    coef, n_fft, hop = (None, None, None)
    if cfg.get("correction") and cfg.get("dataset") != "kauh":
        raw_cache = {}

        def _row_wave(r):
            if r.wav not in raw_cache:
                raw_cache.clear()
                raw_cache[r.wav] = load_wav(r.wav)
            return cycle_wave(raw_cache[r.wav], r.start, r.end)
        coef, n_fft, hop = _device_coefficients(df, cfg, _row_wave)
    raw, n = {}, limit or len(df)
    for i, r in enumerate(df.itertuples()):
        if i >= n or done[i]:
            continue
        if r.wav not in raw:
            raw = {r.wav: load_wav(r.wav)}  # keep one recording in memory
        wave = cycle_wave(raw[r.wav], r.start, r.end)
        if coef is not None:
            wave = apply_spectrum_correction(wave, coef[r.device], n_fft, hop)
        fr, tok = enc.run(wave, set(keep), tl)
        layers[i] = torch.stack([fr[k] for k in keep]).cpu().numpy().astype(np.float16)
        if tokens is not None:
            tokens[i] = tok.cpu().numpy().astype(np.float16)
        done[i] = True
        if i % 128 == 127:
            _flush(arrays, done, out / "done.npy")
            print(f"{done.sum()}/{len(df)} rows", flush=True)
    _flush(arrays, done, out / "done.npy")
    print(f"finished: {done.sum()}/{len(df)} rows done in {out}")
```

Update the module docstring (top of file) to document the new config key, after the `dataset: kauh` paragraph:

```
Config key `correction: {source_devices: [...], n_fft, hop}` applies A1 spectrum correction (src.shift.correction) to
every ICBHI waveform before encoding: a coefficient table is built once from all devices present in the cycle table,
referenced against `source_devices`, and each row is corrected with its own device's coefficient. Calibration rows
(source devices) and test rows (other devices) each get their own coefficient, so they are corrected toward the same
reference separately. ICBHI only (KAUH windows have no `device` column). Always bump `name` when adding or changing
`correction`.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --no-project --python .venv/bin/python python -m pytest -q tests/test_features.py -v`
Expected: PASS (all tests, including the new one and the pre-existing `test_extract_without_tokens_is_resumable_and_writes_meta` unchanged since it passes no `correction` key)

- [ ] **Step 5: Commit**

```bash
git add src/features.py tests/test_features.py
git commit -m "feat: wire A1 spectrum correction into feature extraction"
```

---

### Task 4: Calibration/evaluation patient split for Stage 2 (item 10)

**Files:**
- Modify: `src/data/splits.py`
- Test: `tests/test_core.py`

**Interfaces:**
- Consumes: output of `device_holdout_official(df, device)["cal"]` (a DataFrame with `patient`, `device` columns) — but works on any DataFrame with those two columns.
- Produces: `calibration_eval_splits(df, n_splits=20, fracs=(0.5, 0.5), seed=0)` -> `list[dict]`, each `{"calibration": [...patient ids...], "evaluation": [...patient ids...]}`, patient-disjoint within each split, stratified so each device's patients are divided between calibration/evaluation in roughly `fracs` proportion inside every one of the `n_splits` re-splits.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_core.py` (near `test_device_holdout_official_disjoint_and_seen_flag`):

```python
def test_calibration_eval_splits_disjoint_and_stratified_by_device():
    import pandas as pd
    from src.data.splits import calibration_eval_splits
    # 4 patients on device A, 1 on device B (rare-device edge case: must not crash or drop B).
    df = pd.DataFrame({
        "patient": ["a1", "a2", "a3", "a4", "b1"],
        "device": ["A", "A", "A", "A", "B"],
    })
    splits = calibration_eval_splits(df, n_splits=20, fracs=(0.5, 0.5), seed=0)
    assert len(splits) == 20
    for s in splits:
        cal, ev = set(s["calibration"]), set(s["evaluation"])
        assert not cal & ev
        assert cal | ev == {"a1", "a2", "a3", "a4", "b1"}
        assert len(cal) >= 1 and len(ev) >= 1  # device B's single patient must land somewhere every time
    # Different seeds across the 20 splits actually move patients around (not 20 copies of the same split).
    assert len({frozenset(s["calibration"]) for s in splits}) > 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-project --python .venv/bin/python python -m pytest -q tests/test_core.py::test_calibration_eval_splits_disjoint_and_stratified_by_device -v`
Expected: FAIL (`ImportError: cannot import name 'calibration_eval_splits'`)

- [ ] **Step 3: Implement in `src/data/splits.py`**

Add after `device_holdout_official`:

```python
def calibration_eval_splits(df, n_splits=20, fracs=(0.5, 0.5), seed=0):
    """Repeatedly split the official test patients of the non-held-out devices into calibration and evaluation groups.

    Stratified by device so every device's patients are divided in roughly `fracs` proportion in every split. A
    device with only one patient alternates which group that patient goes to across splits (seed-determined),
    rather than always being dropped into the same group or excluded.

    Args:
        df: DataFrame with `patient`, `device` columns (e.g. `device_holdout_official(...)["cal"]`).
        n_splits: Number of independent re-splits.
        fracs: (calibration, evaluation) patient fractions, applied per device.
        seed: Base seed; split i uses seed `seed + i`.

    Returns:
        List of `n_splits` dicts `{"calibration": [...], "evaluation": [...]}`, sorted patient id lists.
    """
    by_device = {d: sorted(set(g.patient)) for d, g in df.groupby("device")}
    out = []
    for i in range(n_splits):
        rng = np.random.default_rng(seed + i)
        cal, ev = [], []
        for ids in by_device.values():
            ids = list(ids)
            rng.shuffle(ids)
            if len(ids) == 1:
                (cal if rng.random() < fracs[0] else ev).append(ids[0])
                continue
            cut = min(max(1, round(len(ids) * fracs[0])), len(ids) - 1)
            cal.extend(ids[:cut])
            ev.extend(ids[cut:])
        out.append({"calibration": sorted(cal), "evaluation": sorted(ev)})
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --no-project --python .venv/bin/python python -m pytest -q tests/test_core.py -v`
Expected: PASS (all tests, including the new one)

- [ ] **Step 5: Commit**

```bash
git add src/data/splits.py tests/test_core.py
git commit -m "feat: add calibration/evaluation patient re-splits for Stage 2 (v8 item 10)"
```

---

## Self-Review Notes

- **Spec coverage:** Items 1-3 (Task 1), item 4 first half + docstring fix (Tasks 2-3), item 10 first half (Task 4) covered. Item 4 second half (A2 wiring), items 5-9, 11, and the rest of item 10 explicitly deferred (see "Out of scope") because they need design decisions (new encoder hook, per-variant config shape, metric formulas) a plan can't respond to with real code without guessing.
- **Placeholder scan:** No TBD/TODO left in task steps.
- **Type consistency:** `calibration_eval_splits` returns `list[dict[str, list[str]]]`, matching its test's usage (`s["calibration"]`, `s["evaluation"]`). `extract()`'s new `coef` local is `dict[str, np.ndarray] | None`, consumed only inside the same function.
- **Review Focus:** all 5 items each map onto a test (no-correction regression: Task 3 Step 4 reruns the full `test_features.py` file including the pre-existing test; source-only reference: Task 3's test builds the coefficient from `source_devices=["Meditron"]` only; rare-device crash: Task 4's test uses a device with exactly 1 patient; resume-determinism: not independently tested here — flagged as a gap, see below; explicit arithmetic/geometric choice: deferred, noted in "Out of scope").

**Known gap:** resume-after-interrupt determinism for `correction`-enabled extraction is not covered by a test in this plan (would need a test that calls `extract(cfg, limit=1)` then `extract(cfg)` with correction on and asserts the second call's coefficients match the first — the coefficient table is recomputed from the *full* table each call so this should hold by construction, but it's untested). Worth a follow-up test if this is used for a real multi-day run.
