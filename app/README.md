# AuscultTrust demo: inference API and web app

Two parts that deploy independently:

- `../api/` (repo root): FastAPI inference service (`uvicorn api.main:app`). It serves the frozen models from a **bundle** (`api/bundle.py`), so a new model only needs a new bundle. With no bundle or checkpoints it still starts, reports `unavailable` and answers 503, never a made-up prediction.
- `app/web/`: static web app (Vite + TypeScript, no UI framework). **Analyse a recording** (WAV upload, optionally with its ICBHI `.txt` to show the true labels, or live microphone recording; light, dark or system theme) and **Speaker → microphone demo**.

```
browser ─► web app on Cloudflare Pages (static, https://auscult-trust.pages.dev)
   │  fetch, Authorization: Bearer <token>, CORS allowlist
   ├─► primary API: this machine (GPU), public through Tailscale Funnel (https://auscult.tail09c58b.ts.net → 127.0.0.1:8000)
   └─► fallback API: Hugging Face Space (Docker, CPU), tried when the primary does not answer or has no model
```

The page tries the API address from its connection dialog (or `VITE_API_BASE`), then `VITE_API_FALLBACK`. The status chip says "Fallback API ready" when the fallback is in use. Both APIs must accept the same token.

## Install

Needs: the repo's training environment `.venv-train` (Python 3.10, see `docs/SETUP.md`), the two trained units' checkpoints under `checkpoints/train/`, an NVIDIA GPU (about 1.1 GB), Node 20 or newer.

```bash
uv pip install --python .venv-train/bin/python -r api/requirements.txt     # fastapi, uvicorn, python-multipart, httpx, pytest
PYTHONPATH=. .venv-train/bin/python -m api.bundle                          # writes app/bundle/ (22 MB: manifest, SC state, 12 sample WAVs)
(cd app/web && npm ci)
```

## Run locally

```bash
PYTHONPATH=. .venv-train/bin/uvicorn api.main:app --host 127.0.0.1 --port 8000   # API
cd app/web && npm run dev                                                          # web, http://127.0.0.1:5173 (default API: http://127.0.0.1:8000)
```

The connection button in the top bar sets the API address and the access token; the browser keeps them in local storage.

## Deploy

Order: configuration, API host, Tailscale Funnel, Cloudflare Pages, then (optional) the Hugging Face fallback.

### 1. Configuration

| Variable | Where | Meaning |
| --- | --- | --- |
| `AT_APP_TOKEN` | API | access token every route except `/v1/health` needs. Generate: `python -c "import secrets; print(secrets.token_urlsafe(24))"`. Empty means no auth: never leave it empty on a public API |
| `AT_APP_ORIGINS` | API | comma-separated CORS origins: the Pages origin, exact, no trailing slash, plus `http://localhost:5173,http://127.0.0.1:5173` for local development |
| `AT_APP_BUNDLE` | API | bundle directory (default `app/bundle`) |
| `VITE_API_BASE` | web build | the primary API's URL, read at build time (the saved connection in the browser wins) |
| `VITE_API_FALLBACK` | web build | the fallback API's URL; leave unset when there is none |
| `AT_WEIGHTS_REPO`, `HF_TOKEN` | Hugging Face Space | private weights repo and a read token for it (section 5) |

```bash
cp app/deploy/env.example app/deploy/.env      # git-ignored; set AT_APP_TOKEN and AT_APP_ORIGINS
```

The token is typed into the page's connection dialog; it is never part of the web build.

### 2. API host (this machine)

```bash
bash app/deploy/serve.sh start        # uvicorn on 127.0.0.1:8000, detached; reads app/deploy/.env. Also `status`, `stop`. Log: logs/app-api.log
curl -s localhost:8000/v1/health      # {"status":"ok","pipelines":["auscult","baseline"],"gpu":true,...}
```

Restart it after changing `.env`. The API binds to localhost only; the tunnel in the next step is what makes it public.

### 3. Tailscale Funnel (public HTTPS URL for the API)

Funnel gives a fixed `https://<host>.<tailnet>.ts.net` URL with a valid certificate, needs no domain and no open inbound port.

1. Install and log in (once). Create a free account at tailscale.com, then:
   ```bash
   curl -fsSL https://tailscale.com/install.sh | sh      # installs and starts the tailscaled service
   sudo tailscale up                                     # prints a login URL: open it and approve this machine
   ```
2. Name the machine (this becomes the URL's first part; a name already used by a stale device gets a `-1` suffix, so delete stale devices in the admin console, Machines):
   ```bash
   sudo tailscale set --operator=$USER                   # once: lets your user change settings without sudo
   tailscale set --hostname=auscult
   tailscale status --self --json | python3 -c "import sys,json; print(json.load(sys.stdin)['Self']['DNSName'])"
   ```
   The `tail09c58b` part is the tailnet name. Change it only in the admin console (DNS, tailnet name).
3. Enable Funnel for the tailnet (once): run `tailscale funnel --bg 8000`. If it prints `Funnel is not enabled on your tailnet` with a login.tailscale.com link, open the link and approve. In the admin console (DNS) HTTPS certificates must be on.
4. Start it (it persists across reboots) and check:
   ```bash
   tailscale funnel --bg 8000            # https://auscult.tail09c58b.ts.net → http://127.0.0.1:8000
   tailscale funnel status
   curl -s https://auscult.tail09c58b.ts.net/v1/health       # from a device NOT on the tailnet, for example a phone on mobile data
   ```
   Stop it with `tailscale funnel --https=443 off`, or `tailscale funnel reset` to clear everything.

Notes:
- Do not run `tailscale cert` in the repo: it writes a certificate and a private key into the current directory. Funnel needs no manual certificate.
- A device on your tailnet, including this machine, resolves the name to a tailnet address. From there the public page's request counts as local-network access and Chrome blocks it (`ERR_BLOCKED_BY_LOCAL_NETWORK_ACCESS_CHECKS`), and the host cannot reach its own public Funnel address. Test from an outside device.
- Funnel listens on ports 443, 8443 and 10000 only, and Tailscale applies non-configurable bandwidth limits ([docs](https://tailscale.com/docs/features/tailscale-funnel)).

### 4. Static web app on Cloudflare Pages

Pages serves from the site root, from direct upload, and is free. (GitHub Pages serves a project site under `/<repo>/`, which needs Vite's `base` changed, and a free account needs a public repo.)

```bash
(cd app/web && VITE_API_BASE=https://auscult.tail09c58b.ts.net npm run build)      # add VITE_API_FALLBACK=<space url> once there is one
npx wrangler@4.60.0 login                                                          # once, opens a browser
npx wrangler@4.60.0 pages project create auscult-trust --production-branch main    # once
npx wrangler@4.60.0 pages deploy app/web/dist --project-name auscult-trust --branch main
```

Node 20 needs wrangler 4.60 (newer wranglers need Node 22). The site is `https://auscult-trust.pages.dev`; if Cloudflare gives another name, put that origin into `AT_APP_ORIGINS` and restart the API. Rebuild and redeploy whenever the API URL changes. Open the site, click the status chip, paste the token. [Fact] A headless Chrome session on the Pages site connected to the API through Funnel, uploaded a recording and got results (the check disabled Chrome's local-network blocking because the test ran on the API host).

### 5. Fallback API: Hugging Face Space (CPU)

The Space runs the same `api.main:app` on CPU (`api/engine.py` turns the training code's `.cuda()` calls into no-ops when there is no GPU). Checked by building the image and running it locally with `--cpus 2 --memory 16g`: argmax agrees with the GPU on all 14 windows (max |Δp| 0.0008), memory 1.7 GiB, a 20 s recording takes 30 to 50 s (GPU: 0.4 s), so recordings near the 90 s limit can hit the 120 s timeout there. The Space has not been created on Hugging Face.

```bash
PYTHONPATH=. .venv-train/bin/python scripts/stage_space.py      # writes build/space (public, 13 MB) and build/weights (private, 1.1 GB); build/ is git-ignored
```

1. Create a private **model** repo (for example `<user>/auscult-trust-weights`) and upload `build/weights/` to it: the two checkpoints (345 MB each), the BEATs pretrained file and the SC state. The weights stay out of the public Space repo.
2. Create a **Docker Space** and push `build/space/` to it.
3. In the Space settings: secrets `AT_APP_TOKEN` and `HF_TOKEN` (read access to the weights repo); variables `AT_WEIGHTS_REPO=<user>/auscult-trust-weights` and `AT_APP_ORIGINS=https://auscult-trust.pages.dev`.
4. `curl https://<user>-auscult-trust-api.hf.space/v1/health` says `"status":"ok"` once the weights are downloaded, `unavailable` until then. Rebuild the web app with `VITE_API_FALLBACK` set to that URL and redeploy (section 4).

Caveats: [Assumption] free CPU Basic Spaces (2 vCPU, 16 GB) per the Hub docs, but a community thread reports Docker Spaces being restricted for unpaid accounts, so confirm your plan allows one. Free Spaces sleep when idle and need a minute or more to wake and download the weights. `stage_space.py` leaves the ICBHI demo WAVs out (dataset-terms TODO below), so Demo mode has no samples on the Space.

### Choosing the tunnel (measured 22/10/2026)

From this host, API without a model so only the tunnel is timed; the client was the host itself, so each request goes out to the edge and back. Health: 30 sequential GETs; upload: 4 POSTs of 8 MB.

| Route | Health median / p95 (ms) | 8 MB upload median (ms) | Stable URL | Needs |
| --- | --- | --- | --- | --- |
| direct, no tunnel | 1 / 1 | 12 | n/a | n/a |
| Cloudflare quick tunnel, QUIC | 186 / 283 | 794 | no | nothing |
| Cloudflare quick tunnel, `--protocol http2` | 177 / 230 | 758 | no | nothing |
| ngrok free | 221 / 243 | 2243 | `*.ngrok-free.dev` | ngrok account |
| Tailscale Funnel | not timed | not timed | yes | Tailscale account |

[Fact] Funnel works end to end (section 3). Not timed: this host resolves its own Funnel name to the tailnet address, so a timing here would be loopback. Choice: Funnel as primary (stable URL, no domain, no interstitial). ngrok free is the weakest here: slowest upload, 1 GB/month and 20,000 requests/month ([docs](https://ngrok.com/docs/pricing-limits/free-plan-limits)), a browser interstitial page unless requests carry `ngrok-skip-browser-warning` (which the API's CORS `allow_headers` does not list, so the preflight answers 400). A spare when Funnel is down: `cloudflared tunnel --protocol http2 --url http://127.0.0.1:8000` prints a temporary `https://*.trycloudflare.com` URL (set it in the page's connection dialog; it changes at every start). Cloudflare's named tunnels need a domain and are not used.

### Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| Page: `Cannot reach the API` | wrong address, API down, or `AT_APP_ORIGINS` does not equal the page's origin exactly (scheme and host). From the API host the Funnel URL may fail for the reasons in section 3 |
| Page: `Token needed` | token missing or wrong in the connection dialog |
| Page: `No model loaded` | the API is up but its bundle or checkpoints failed to load; read `/v1/health` `reason` and `logs/app-api.log` |
| `tailscale funnel`: `Access denied: serve config denied` | run `sudo tailscale set --operator=$USER` once |
| `tailscale funnel`: `Funnel is not enabled on your tailnet` | open the link it prints and approve |
| Microphone button errors | the page is not on HTTPS |
| `wrangler` hangs or needs Node 22 | use `wrangler@4.60.0` on Node 20 |

## API

All responses are JSON and carry `X-Request-ID`. Errors: `{"error": {"code", "message"}, "request_id"}`. With `AT_APP_TOKEN` set, every route except `/v1/health` needs `Authorization: Bearer <token>`.

| Method and path | Purpose |
| --- | --- |
| `GET /v1/health` | always 200 and open. `status`: `ok` when the models are loaded, else `unavailable` with a `reason` (no bundle, load failed; never a path). Also pipelines, GPU, pending requests |
| `GET /v1/model` | classes, alpha, preprocessing, per pipeline: label, checkpoint SHA-256, epoch, thresholds, calibration counts, SC mode and s_ref hash; upload limits; SC frequency axis |
| `GET /v1/samples` | demo samples: id, class, patient, device, chest location, annotated cycles, SHA-256 |
| `GET /v1/samples/{id}/audio` | the sample WAV |
| `GET /v1/samples/{id}/result` | inference on the sample with its device and annotated cycles (cached) |
| `POST /v1/infer` | multipart: `file` (WAV, required); `device` (ICBHI name or empty); `cycles` (JSON `[[start, end], ...]`) or `annotation` (ICBHI `.txt`); `reference_sample` (demo: align to that sample and use its cycles); `window_sec`, `hop_sec` |

Inference response (abridged):

```json
{
  "request_id": "a7920525864f4701",
  "source": {"kind": "upload"},
  "audio": {"filename": "x.wav", "sample_rate": 44100, "channels": 1, "duration_sec": 20.0, "subtype": "PCM_24",
            "peak_dbfs": -0.9, "clipped_share": 0.0, "sha256": "…"},
  "device": {"declared": null},
  "segmentation": {"mode": "windows", "window_sec": 8.0, "hop_sec": 4.0},
  "alignment": {"reference_sample": "…", "lag_sec": 1.0001, "peak_score": 650.89, "reliable": true, "trimmed_to_sec": 20.0},
  "pipelines": {
    "auscult": {
      "label": "AuscultTrust (SC)",
      "sc": {"applied": true, "mode": "static", "coefficient_source": "estimated:recording", "gain_db": [0.0, …513 values]},
      "segments": [{"index": 0, "start": 0.0, "end": 8.0,
                    "probs4": {"normal": 0.4871, "crackle": 0.0334, "wheeze": 0.4335, "both": 0.046}, "pred4": "normal",
                    "set4": ["normal", "crackle", "wheeze", "both"],
                    "probs2": {"normal": 0.4871, "abnormal": 0.5129}, "pred2": "abnormal", "set2": ["normal", "abnormal"],
                    "…": "truth4, truth2 when cycles carry labels"}],
      "recording": {"p_abnormal": 0.258, "set": ["normal", "abnormal"], "triage": "uncertain", "aggregation": "mean",
                    "windows": [{"start": 0.0, "end": 8.0, "p_abnormal": 0.5129}, {"start": 12.0, "end": 20.0, "p_abnormal": 0.0037}]}
    },
    "baseline": {"…": "same shape, sc.applied = false"}
  },
  "timing_ms": 1630
}
```

`cached` is true when the answer came from the API's result cache (same file and settings); `alignment` appears only with `reference_sample`; `truth4`/`truth2` only when the cycles carry labels (demo samples, annotation upload). Times are seconds on the input recording (after alignment for microphone takes).

Validation (server; the web app checks the same limits before upload): RIFF/WAVE header, decodable by libsndfile, 1-2 channels, 4-48 kHz, 2-90 s, at most 40 MB, finite samples, not silent (peak above -80 dBFS). Every route except health answers 503 `model_unavailable` while the status is `unavailable`; no prediction is ever produced without the model. Error codes: `model_unavailable` (503), `internal` (500, inference crashed; details only in the server log), `not_wav`, `unreadable`, `channels`, `sample_rate`, `duration`, `non_finite`, `silent`, `too_large`, `bad_device`, `bad_cycles`, `bad_annotation`, `bad_window`, `unknown_sample`, `unauthorized`, `busy` (more than 4 requests waiting, 429), `timeout` (120 s, 504).

Inference runs in a worker thread behind one lock (one GPU, shared with training jobs), so the event loop keeps answering. Results are cached by audio SHA-256 and parameters (last 64).

What is not exposed: checkpoint paths, training arguments, unit directories (`/v1/model` returns hashes and thresholds only). The token is typed by the user in the browser; it is not built into the web bundle.

## Demo mode: how playback and recording line up

1. The browser opens the microphone with echo cancellation, noise suppression and automatic gain **off** (with echo cancellation on, the browser would subtract the very sound the page plays).
2. An AudioWorklet posts raw microphone blocks with their audio-context frame index. Playback of the reference is scheduled on the same `AudioContext` 0.4 s later, so every captured sample is placed on the playback clock: sample 0 of the take is the moment playback starts.
3. Recording continues 1.2 s after the reference ends (speaker and microphone latency, room decay). The take is encoded as 16-bit PCM WAV at the context rate and posted with `reference_sample`.
4. The API estimates the remaining delay (output + input latency, acoustic path) with band-limited (100-2000 Hz) GCC-PHAT, search window -0.25 to +1.5 s. It shifts the take, cuts it to the reference length and scores it on the reference's annotated cycles. A weak correlation peak (less than 12 SDs above the other lags) is reported as unreliable. The cut comes from synthetic takes only (clean 651, noisy 18, white noise 4.3, a different ICBHI recording 8.1) and needs checking on real takes.
5. The web app shifts its displayed take by the same lag, so both panels share one time axis.

The page never routes the microphone to the speaker, so there is no feedback loop. Headphones would defeat the demo (the microphone must hear the speaker). The room tips in the page say this.

## What the model gives, and how the app shows it

The checks below come from the code (`src/evaluation/*`, `src/processing/correction.py`), not from assumptions.

| Output | Source | Shown as |
| --- | --- | --- |
| 4-class softmax (normal, crackle, wheeze, both) per segment | BEATs (fine-tuned) + Patch-Mix CL, 8 s input | coloured boxes over the waveform, packed into rows when segments overlap |
| 2-class view, p(abnormal) = 1 - p(normal), same softmax | same | lane under the waveform |
| Conformal sets, LAC, alpha 0.1 | `src/evaluation/conformal.py` | box label "set k", lane pattern (solid = one class, hatched = both, outline = empty), chips in the tables |
| Recording-level screening (8 s windows, mean of 1 - p(normal), 2-class LAC) | `src/evaluation/screening.py` | triage card with the p(abnormal) bar and the zones the threshold creates |
| SC = spectrum correction: per-bin STFT gain towards the training-device reference | `src/processing/correction.py` | gain curve (dB vs Hz) and the side-by-side pipeline without SC |

The two proposed methods carry a **Proposed** tag in the page: spectrum correction (the SC pipeline) and conformal prediction. The *Conformal prediction* switch in the toolbar (kept in local storage, shared by both modes) hides or shows the conformal layer: set labels on the boxes, set chips in the tables, the set pattern in the 2-class lane, the set stats in the demo comparison and the triage card. Off, the page shows top classes only and the screening card shows a plain Normal / Abnormal at p(abnormal) 0.5.

Pipelines in the default bundle (seed 0, official split, report epoch 50; the two units differ only by SC):

| Name | Unit | Epoch | Test-time SC | Thresholds cycle4 / cycle2 / screen |
| --- | --- | --- | --- | --- |
| `auscult` = AuscultTrust (SC) | `outputs/train/lodo-v1/beats+ft+patchmix_cl+sc+sc_static/seed0` | 50 | yes, static mode | 0.9992 / 0.9976 / 0.9911 |
| `baseline` = P0 (no SC) | `outputs/train/grid-v1/beats+ft+patchmix_cl/seed0` | 50 | no | 0.9993 / 0.9981 / 0.9664 |

The AST units these replaced (`screen-ast`) lost their checkpoints; to serve other units, edit `PIPELINES` in `api/bundle.py` and rebuild the bundle. The screening exports come from `python -m src.evaluation.screening infer` on the same checkpoints (`outputs/screening/beats_{auscult,p0}/seed0.npz`).

Thresholds are calibrated on split 0 of `calibration_eval_splits` over the official-test patients without the 2 patients on both sides: 24 patients, 1,238 cycles, 176 recordings. The cycle softmax comes from the unit's saved `test_preds.npz`; the screening softmax comes from `outputs/screening/<cond>/seed0.npz`, and its checkpoint hash is checked against the weights. `api/test_api.py` and the engine check reproduce the saved test softmax within float16 rounding (max |Δp| ≤ 0.004, argmax 100% agree on the 3 recordings checked).

Segments: an uploaded ICBHI annotation `.txt` (or `cycles` JSON) gives breathing cycles, which is the unit the model was trained on. Without one, the API scores sliding windows (default 8 s, step 4 s). The screening decision always uses the registered windows (max(1, floor(duration / 8)) windows evenly spread).

SC target spectrum: when the stethoscope is a known ICBHI device, the API uses the coefficient saved with the unit (from that device's unlabelled official-test cycles). When the device is unknown (any upload without a device, every microphone take), the API estimates it from the mean spectrum of the recording's own segments.

## Tests

```bash
PYTHONPATH=. .venv-train/bin/python -m pytest -q api/test_contract.py   # no checkpoints or GPU: start-up without a model (503, no fabricated output), validation, auth, CORS, error shape
PYTHONPATH=. .venv-train/bin/python -m pytest -q api/test_api.py        # API end to end on the real bundle (GPU; skipped without the bundle or a GPU)

# browser end to end, both modes, Chrome with a fake microphone (API and `npm run dev` running;
# E2E_API=<url> E2E_TOKEN=<token> point the page at another or token-protected API)
.venv/bin/python - <<'EOF'
import numpy as np, soundfile as sf
from scipy.signal import butter, lfilter
x, sr = sf.read("app/bundle/samples/178_1b2_Pl_mc_AKGC417L.wav", dtype="float32")
b, a = butter(2, [200 / (sr / 2), 1800 / (sr / 2)], "band")     # a crude speaker + microphone band
y = np.concatenate([np.zeros(sr), 0.5 * lfilter(b, a, x) / np.abs(x).max(), np.zeros(2 * sr)])
sf.write("/tmp/fake_mic.wav", y.astype(np.float32), sr, subtype="PCM_16")
EOF
cd app/web && npm run e2e -- /tmp/e2e-shots /tmp/fake_mic.wav
E2E_API=http://127.0.0.1:8000 node e2e/record.mjs /tmp/e2e-shots /tmp/fake_mic.wav http://127.0.0.1:5173/   # live recording + preprocess
E2E_API=http://127.0.0.1:8000 node e2e/labels.mjs /tmp/e2e-shots <ICBHI>/187_1b1_Ll_sc_Meditron.txt http://127.0.0.1:5173/   # wav + txt: true labels, stethoscope from the name, theme switch
E2E_API=<url of an API without a model> node e2e/unavailable.mjs /tmp/e2e-shots http://127.0.0.1:5173/   # unavailable-model state, retry, no overflow
# fallback: build the page with VITE_API_FALLBACK=<working API>, serve it, set E2E_API to a dead address
E2E_API=http://127.0.0.1:9 node e2e/fallback.mjs /tmp/e2e-shots <url of that build>
node e2e/preprocess.check.mjs                                                                            # noise reduction and trimming on synthetic audio
```

The API test covers: health and metadata (no paths leaked), sample inference with saved SC coefficients, upload = sample result for the same file, demo alignment of a delayed (0.37 s), band-limited, quieter 48 kHz take (recovered within 2 ms), and the error paths. The browser run covers upload → analysis → overlays and tooltips, then demo class and sample choice → microphone permission → playback and recording → alignment → comparison, plus dark theme at tablet width. `record.mjs` covers live recording (record, stop, discard), both preprocess switches and an analysis of the recorded take.

Preprocessing (Real mode, both off by default) runs in the browser, on uploads and recordings alike, and the waveform shows exactly what is uploaded: **Reduce noise** is spectral subtraction (per-frequency noise floor = 15th percentile of the take, over-subtraction 1.5, gain floor 0.2), **Trim silence** cuts leading and trailing silence only (20 ms frames, 0.3 s margin, never below the API minimum). The model was trained on unprocessed ICBHI audio, so processed input is outside what the conformal guarantee was calibrated on; a sound that fills the whole take is treated as noise.

## Known limitations

- **Demo samples are not same-patient device triplets.** No ICBHI patient was recorded on three devices. The four patients recorded on two (112, 158, 218, 226) are on the training side or on both official sides, so they are excluded. Each class instead has 3 recordings from evaluation-half patients (never trained on, never in the conformal calibration), spread over the three official-test devices (Meditron, Litt3200, AKGC417L; LittC2SE has no official-test recording). For "crackle + wheeze", no evaluation-half recording from another device has it as the majority label, so all three come from AKGC417L, and one has it on 67% of its cycles. The sample list in the page shows each recording's annotated share.
- **Windows are not cycles.** Without annotations the model scores 8 s windows. It was trained on breathing cycles repeat-padded to 8 s, so window predictions are less reliable than cycle predictions.
- **Coverage is guaranteed only for data exchangeable with the calibration set.** A microphone take or another stethoscope is a device shift. Whether SC keeps coverage under such a shift is the open research question (H2), not something the demo can assume. Both served models are very confident at epoch 50, so their 4-class thresholds are about 0.999, which means many 3- and 4-class sets; the app shows them as they are.
- **SC from one recording.** For an unknown device the correction is estimated from that single recording, so it also flattens the sound's own spectrum. The registered pipeline estimates it from many unlabelled recordings per device.
- **One global lag.** The demo corrects a constant delay. Clock drift between separate speaker and microphone hardware (parts per million) and room reverberation are not corrected.
- **One seed per arm**, chosen for serving; the reported results average 5-10 seeds.
- **Real hardware not tested yet.** Testing so far used Chrome's fake microphone. A run with a real speaker and microphone, and in Firefox and Safari, is still to do.
- **Dataset terms.** The ICBHI sample WAVs are served from the API host, not built into the web bundle. TODO: confirm the ICBHI 2017 database terms allow serving these files on a public URL before deploying publicly.
