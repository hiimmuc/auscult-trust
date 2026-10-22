# AuscultTrust demo: inference API and web app

Two parts that run and deploy independently:

- `app/api/`: FastAPI inference service. It serves the frozen Stage-1 models from a **bundle** (`app/api/bundle.py`), so a new model only needs a new bundle.
- `app/web/`: static web app (Vite + TypeScript, no UI framework). It has two modes: **Analyse a recording** (WAV upload, optionally with its ICBHI `.txt` to show the true labels, or live microphone recording with a Siri-style level view; light, dark or system theme) and **Speaker → microphone demo**.

```
browser ── https://auscult.ausculttrust.vn ──────┐
   │  HTTPS, Authorization: Bearer <token>,       │
   ▼  CORS allowlist                              ▼
Cloudflare edge ── Tunnel ──► cloudflared on the GPU host ──► uvicorn 127.0.0.1:8000 (app.api.server, https://auscult-api.ausculttrust.vn)
                                                         └──► static web build 127.0.0.1:8090 (app/web/dist)
                                                                 │
                                                                 ├─ Engine: validation → 16 kHz → segments → SC → fbank → BEATs → softmax
                                                                 └─ bundle: manifest.json, sc_*.npz, samples/*.wav (+ checkpoints by path, SHA-256 checked)
```

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

The AST units these replaced (`screen-ast`) lost their checkpoints; to serve other units, edit `PIPELINES` in `app/api/bundle.py` and rebuild the bundle. The screening exports come from `python -m src.evaluation.screening infer` on the same checkpoints (`outputs/screening/beats_{auscult,p0}/seed0.npz`).

Thresholds are calibrated on split 0 of `calibration_eval_splits` over the official-test patients without the 2 patients on both sides: 24 patients, 1,238 cycles, 176 recordings. The cycle softmax comes from the unit's saved `test_preds.npz`; the screening softmax comes from `outputs/screening/<cond>/seed0.npz`, and its checkpoint hash is checked against the weights. `app/api/test_api.py` and the engine check reproduce the saved test softmax within float16 rounding (max |Δp| ≤ 0.004, argmax 100% agree on the 3 recordings checked).

Segments: an uploaded ICBHI annotation `.txt` (or `cycles` JSON) gives breathing cycles, which is the unit the model was trained on. Without one, the API scores sliding windows (default 8 s, step 4 s). The screening decision always uses the registered windows (max(1, floor(duration / 8)) windows evenly spread).

SC target spectrum: when the stethoscope is a known ICBHI device, the API uses the coefficient saved with the unit (from that device's unlabelled official-test cycles). When the device is unknown (any upload without a device, every microphone take), the API estimates it from the mean spectrum of the recording's own segments.

## Setup

Python: the training env `.venv-train` plus the API packages (`app/api/requirements.txt`):

```bash
uv pip install --python .venv-train/bin/python -r app/api/requirements.txt
PYTHONPATH=. .venv-train/bin/python -m app.api.bundle          # writes app/bundle/ (22 MB: manifest, SC state, 12 sample WAVs)
```

Web (Node 20 or newer):

```bash
cd app/web && npm ci
```

## Run locally

```bash
# API (needs the two units' checkpoints under checkpoints/train/ and a GPU; about 1.1 GB of GPU memory)
PYTHONPATH=. .venv-train/bin/uvicorn app.api.server:app --host 127.0.0.1 --port 8000

# web app, http://127.0.0.1:5173 (talks to http://127.0.0.1:8000 by default)
cd app/web && npm run dev
```

The connection button in the top bar sets the API address and the access token; they are kept in the browser's local storage.

## API

All responses are JSON and carry `X-Request-ID`. Errors: `{"error": {"code", "message"}, "request_id"}`. With `AT_APP_TOKEN` set, every route except `/v1/health` needs `Authorization: Bearer <token>`.

| Method and path | Purpose |
| --- | --- |
| `GET /v1/health` | `status` (`ok` once the models are loaded), pipelines, GPU, pending requests |
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

`alignment` appears only with `reference_sample`; `truth4`/`truth2` only when the cycles carry labels (demo samples, annotation upload). Times are seconds on the input recording (after alignment for microphone takes).

Validation (server; the web app checks the same limits before upload): RIFF/WAVE header, decodable by libsndfile, 1-2 channels, 4-48 kHz, 2-90 s, at most 40 MB, finite samples, not silent (peak above -80 dBFS). Error codes: `not_wav`, `unreadable`, `channels`, `sample_rate`, `duration`, `non_finite`, `silent`, `too_large`, `bad_device`, `bad_cycles`, `bad_annotation`, `bad_window`, `unknown_sample`, `unauthorized`, `busy` (more than 4 requests waiting, 429), `timeout` (120 s, 504).

Inference runs in a worker thread behind one lock (one GPU, shared with training jobs), so the event loop keeps answering. Results are cached by audio SHA-256 and parameters (last 64).

## Demo mode: how playback and recording line up

1. The browser opens the microphone with echo cancellation, noise suppression and automatic gain **off** (with echo cancellation on, the browser would subtract the very sound the page plays).
2. An AudioWorklet posts raw microphone blocks with their audio-context frame index. Playback of the reference is scheduled on the same `AudioContext` 0.4 s later, so every captured sample is placed on the playback clock: sample 0 of the take is the moment playback starts.
3. Recording continues 1.2 s after the reference ends (speaker and microphone latency, room decay). The take is encoded as 16-bit PCM WAV at the context rate and posted with `reference_sample`.
4. The API estimates the remaining delay (output + input latency, acoustic path) with band-limited (100-2000 Hz) GCC-PHAT, search window -0.25 to +1.5 s. It shifts the take, cuts it to the reference length and scores it on the reference's annotated cycles. A weak correlation peak (less than 12 SDs above the other lags) is reported as unreliable. The cut comes from synthetic takes only (clean 651, noisy 18, white noise 4.3, a different ICBHI recording 8.1) and needs checking on real takes.
5. The web app shifts its displayed take by the same lag, so both panels share one time axis.

The page never routes the microphone to the speaker, so there is no feedback loop. Headphones would defeat the demo (the microphone must hear the speaker). The room tips in the page say this.

## Deploy across networks with Cloudflare

1. **API host** (the GPU machine): set the environment (`app/deploy/env.example`): a long random `AT_APP_TOKEN` and `AT_APP_ORIGINS` = the web app's origin. Run uvicorn on `127.0.0.1` only.
2. **Tunnel**: `cloudflared` creates an outbound-only tunnel, so the GPU host needs no open inbound port. Use `app/deploy/cloudflared.example.yml` for a named tunnel on your domain (`https://api.example.org`). For a quick test without a domain: `cloudflared tunnel --url http://127.0.0.1:8000` gives a temporary `https://*.trycloudflare.com` address. `cloudflared` 2026.10.1 is installed on the GPU host.
3. **Web app**: `VITE_API_BASE=https://api.example.org npm run build` and publish `app/web/dist/` on any HTTPS static host (Cloudflare Pages: build command `npm run build`, output `dist`, root `app/web`). HTTPS is required: browsers allow microphone access only on secure origins (or localhost).
4. **This host** (tunnel `auscult-trust-tunnel`, `app/deploy/cloudflared.yml`): `https://auscult-api.ausculttrust.vn` → API on 127.0.0.1:8000, `https://auscult.ausculttrust.vn` → static web build on 127.0.0.1:8090 (DNS CNAMEs created with `cloudflared tunnel route dns`). Build and start:

   ```bash
   (cd app/web && VITE_API_BASE=https://auscult-api.ausculttrust.vn npm run build)
   bash app/deploy/serve.sh start      # API + web + tunnel, detached; `status`, `stop`
   grep AT_APP_TOKEN app/deploy/.env   # the token to type into the page's connection dialog
   ```

   `app/deploy/.env` (mode 600, not in git) holds the token and `AT_APP_ORIGINS=https://auscult.ausculttrust.vn`. cloudflared needs outbound TCP or UDP port 7844 to the Cloudflare edge (`region1.v2.argotunnel.com`, `region2.v2.argotunnel.com`); the network of the GPU hosts blocks it as of 20/10/2026, so the tunnel keeps retrying until the port is opened (`logs/app-tunnel.log`, line `Registered tunnel connection`).
5. **Cloudflare controls** (dashboard): a rate-limiting rule on `/v1/infer` (for example 20 requests per minute per IP), "Always use HTTPS", and a request-body size limit at or above 40 MB. Cloudflare Access with an identity login is possible, but cross-origin calls from the web app then need the Access cookie on the API hostname. The bearer token is the simpler control for a demo.

### Cloudflare setup, step by step

Run each step in order; steps 1 and 2 are one-off for the domain, steps 3 to 7 repeat for every GPU host. Host A is the first machine, host B is `10.148.182.97` (RTX 4060, 8 GB; same layout under `~/Workspace/Personal/KHKT-CVA-2026/auscult-trust`).

1. **Put the domain on Cloudflare** (once). Dashboard → *Add a domain* → `ausculttrust.vn` → Free plan. Cloudflare shows two nameservers. At the `.vn` registrar, replace the domain's nameservers with those two. Wait until the dashboard says *Active* (minutes to hours). Check: `dig +short NS ausculttrust.vn` returns the Cloudflare names. Until then `getent hosts auscult-api.ausculttrust.vn` fails (NXDOMAIN).
2. **Tunnel and DNS records** (once, already done on host A). `cloudflared tunnel login` (opens a browser, writes `~/.cloudflared/cert.pem`), `cloudflared tunnel create auscult-trust-tunnel`, then `cloudflared tunnel route dns auscult-trust-tunnel auscult-api.ausculttrust.vn` and the same for `auscult.ausculttrust.vn`. This writes the credentials file `~/.cloudflared/<tunnel-id>.json` (the tunnel id is `b2884e4d-d836-4ba0-96e9-92f5ea32a9c3`). The DNS records live in the account, so another host needs only the credentials file, not `cert.pem`.
3. **Install cloudflared on the host.** Download the binary for the CPU (`linux-amd64`) from the cloudflared GitHub releases, or copy `/usr/bin/cloudflared` from a host that has it, to `~/.local/bin/cloudflared` and `chmod +x` it (copy the real file: `rsync -aL`, not the symlink). Check: `cloudflared --version`.
4. **Credentials.** Copy only the tunnel credentials file, never `cert.pem` (that file can create and delete tunnels for the whole account):

   ```bash
   mkdir -p -m 700 ~/.cloudflared
   rsync -a hostA:~/.cloudflared/b2884e4d-d836-4ba0-96e9-92f5ea32a9c3.json ~/.cloudflared/   # mode 400
   ```

   `app/deploy/cloudflared.yml` already points at `/home/namdp15/.cloudflared/<id>.json`; the username must match or the path in the file must change.
5. **Open the network** (the blocker so far). The host needs outbound **TCP and UDP 7844** to `region1.v2.argotunnel.com` and `region2.v2.argotunnel.com` (IPv4 `198.41.192.0/24` and `198.41.200.0/24`), and DNS to resolve them. Nothing inbound is needed. Ask the network admin for that rule. Test: `cloudflared tunnel --loglevel info run` and read the connectivity table at the start; both rows must say `PASS`:

   ```
   UDP Connectivity  region1.v2.argotunnel.com  PASS
   TCP Connectivity  region1.v2.argotunnel.com  PASS
   ```

   If only TCP is allowed, add `protocol: http2` at the top of `cloudflared.yml` (QUIC needs UDP). As of 22/10/2026 hosts A and B both fail this test (`Connection refused` on TCP, QUIC timeout). Without admin help, run the tunnel from a network that allows 7844 (for example a phone hotspot) while keeping the API on the GPU host reachable from there.
6. **Start everything.** On the host:

   ```bash
   cd ~/Workspace/Personal/KHKT-CVA-2026/auscult-trust
   (cd app/web && VITE_API_BASE=https://auscult-api.ausculttrust.vn npm run build)   # once, or copy dist/
   bash app/deploy/serve.sh start
   grep -c "Registered tunnel connection" logs/app-tunnel.log                      # 4 when connected
   ```

   Keep the tunnel alive across reboots with a user service (run `loginctl enable-linger $USER` once):

   ```ini
   # ~/.config/systemd/user/cloudflared-auscult.service
   [Unit]
   Description=Cloudflare tunnel for AuscultTrust
   After=network-online.target
   [Service]
   ExecStart=%h/.local/bin/cloudflared --no-autoupdate --config %h/Workspace/Personal/KHKT-CVA-2026/auscult-trust/app/deploy/cloudflared.yml tunnel run
   Restart=always
   RestartSec=5
   [Install]
   WantedBy=default.target
   ```

   `systemctl --user daemon-reload && systemctl --user enable --now cloudflared-auscult`. Then start only the API and web from `serve.sh` (or write two more units the same way) and stop the `cloudflared` line there, so the tunnel is not started twice.
7. **Verify from any other network** (phone on mobile data):

   ```bash
   curl -s https://auscult-api.ausculttrust.vn/v1/health          # {"status":"ok","pipelines":["auscult","baseline"],"gpu":true,...}
   curl -sI https://auscult.ausculttrust.vn | head -1             # HTTP/2 200
   ```

   Open `https://auscult.ausculttrust.vn`, click the status chip, paste the token from `grep AT_APP_TOKEN app/deploy/.env`, and run a sample.

Two hosts on one tunnel: both register as connectors and Cloudflare spreads requests over them. Both must serve the same bundle and the same token, or requests alternate between different answers. To switch hosts, stop cloudflared on the one you retire. For separate public names per host, create a second tunnel and hostname instead.

| Symptom | Cause and fix |
| --- | --- |
| `Failed to dial a quic connection ... timeout` | UDP 7844 blocked; allow it, or set `protocol: http2` |
| `HTTP/2 connection is blocked or unreachable` | TCP 7844 blocked too; step 5 |
| `Network is unreachable` in a manual `/dev/tcp` test | the name resolved to IPv6 first; test the IPv4 address or use `cloudflared` itself |
| `Unable to reach the origin service ... connection refused` in the tunnel log | API or web not running on 8000 / 8090; `bash app/deploy/serve.sh status` |
| Browser says `Token needed` or `Cannot reach the API` | wrong token, or `AT_APP_ORIGINS` does not equal the page's origin exactly (scheme and host) |
| Microphone button errors on the public page | the page is not HTTPS; use the Cloudflare hostname, not an IP address |

What is not exposed: checkpoint paths, training arguments, unit directories (`/v1/model` returns hashes and thresholds only). The token is typed by the user in the browser; it is not built into the web bundle.

## Tests

```bash
PYTHONPATH=. .venv-train/bin/python -m pytest -q app/api/test_api.py     # API end to end on the real bundle (GPU)

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
E2E_API=http://127.0.0.1:8000 node e2e/record.mjs /tmp/e2e-shots /tmp/fake_mic.wav http://127.0.0.1:8090/   # live recording + preprocess
E2E_API=http://127.0.0.1:8000 node e2e/labels.mjs /tmp/e2e-shots <ICBHI>/187_1b1_Ll_sc_Meditron.txt http://127.0.0.1:8090/   # wav + txt: true labels, stethoscope from the name, theme switch
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
