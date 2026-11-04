# Dialed In, live

Turn the dial and watch Muse Glimmer-30B respond in real time. A small server runs only the
first layers of the model, adds the sine probe at all seven periods in one batch, and measures
how strongly each attention head in the read layer carries each period. The page draws the
gain curve and one wave per head, and sends the dial setting back to the server.

This is a live readout, not a sealed result. It uses one text window and one read layer, and it
is measured on the spot. The sealed numbers stay in `results/derived` and on `demo/index.html`.

## Files

| File | What it does |
| --- | --- |
| `server.py` | Loads the first layers, injects the probe, runs the lock-in, serves `ws://127.0.0.1:8765` |
| `index.html` | The live page: dial, curve against the dial-at-0 baseline, one wave per head |
| `dial_shim.py` | Where your trained adapter plugs in (fill in from `run_c9.py`) |

## Run it

1. Check the pipeline with no weights (any laptop, CPU is fine):

   ```
   pip install torch transformers websockets
   python3 live/server.py --self-test --device cpu --dtype fp32 --tokens 4096
   ```

   Open `live/index.html` in a browser. The badge turns green, and turning the dial flattens a toy
   curve. It updates every 2 to 3 seconds on a laptop CPU; on the GPU it is much faster.

2. On the GPU pod, with the repo checked out:

   ```
   pip install websockets
   PYTHONPATH=src:. python3 live/server.py \
       --model <Glimmer path or Hugging Face id> \
       --text <one of the five books as .txt> \
       --adapter module:live.dial_shim:load \
       --probe <the probe direction your runs use, saved with torch.save> \
       --inject-after 0 --read-layer 1 --read-label "layer 2"
   ```

   It prints `ready: … ms per update` once the dial-at-0 baseline is measured.

3. On your Mac, forward the port and open the page:

   ```
   ssh -N -L 8765:127.0.0.1:8765 <your RunPod ssh target>
   open live/index.html
   ```

## Match these to your sealed runs before filming

- `--inject-after` and `--read-layer` are 0-based layer indexes. The defaults assume "layer 2" in
  the write-up means index 1 and the probe enters after index 0. Check `run_c9.py`.
- `--probe`: use the same probe direction as your runs. Without it the server draws a random one.
- `--eps`: probe size as a fraction of the hidden-state RMS (default 0.05). Use your runs' value.
- The probe is a cosine. A sine sampled once per token is zero at every token when N = 2, so it would
  inject nothing there. If your runs use a sine with a phase offset, match it.
- `--tokens`: window length, a multiple of 4,096 so every period fits whole cycles (default 8,192).

## Speed

Only layers up to the read layer are loaded (a few GB instead of about 60), and the model stops
right before the read layer's output projection. Each update is one batch of 8 windows: the seven
periods plus one without the probe. The server skips stale dial positions, so the page always shows
the newest setting. The server prints its time per update at startup. It has not been timed on a GPU yet; if it is
slower than about half a second, lower `--tokens` to 4096 so the dial still feels live on camera.

## How the gain is measured

For each period N, the server takes each head's output just before the heads are mixed, subtracts
the same window run without the probe, and multiplies by a sine and cosine at 1/N cycles per token
(a lock-in amplifier). The amplitude it finds, divided by the probe's amplitude, is that head's gain
at N. The curve is the mean over heads, shown as a percentage of the slowest wave. Tilt is the
slope of log gain on log period, the same definition the demo page uses.
