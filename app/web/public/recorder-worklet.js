// Raw microphone capture on the audio thread: after "start", every 128-frame block is posted with its context frame index,
// so the main thread can place it on the same clock as the playback.
class Recorder extends AudioWorkletProcessor {
  constructor() {
    super();
    this.on = false;
    this.port.onmessage = (e) => { this.on = e.data === "start"; };
  }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (this.on && ch) this.port.postMessage({ frame: currentFrame, data: ch.slice(0) });
    return true;
  }
}
registerProcessor("recorder", Recorder);
