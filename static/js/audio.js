// Microphone capture + energy-based voice activity detection.
// Keeps a rolling buffer of 16 kHz audio; when speech starts, waits for the word to
// finish and hands ~1.5 s (with pre-roll) to onUtterance. The server then picks the
// loudest 1-second window, matching the Speech Commands clip format.

const TARGET_SR = 16000;
const BUFFER_SECONDS = 2.5;
const CLIP_SECONDS = 1.5;
const POST_ONSET_MS = 850;   // how long to keep recording after the word starts
const FRAME = 320;           // 20 ms frames at 16 kHz

const WORKLET = `
class Tap extends AudioWorkletProcessor {
  process(inputs) { const ch = inputs[0][0]; if (ch) this.port.postMessage(ch.slice(0)); return true; }
}
registerProcessor('tap', Tap);`;

export class MicListener {
  constructor({ onUtterance, onLevel, onState }) {
    Object.assign(this, { onUtterance, onLevel, onState });
    this.ring = new Float32Array(TARGET_SR * BUFFER_SECONDS);
    this.writePos = 0;
    this.pending = new Float32Array(0);
    this.noiseFloor = 0.005;
    this.state = 'off';
    this.sensitivity = 4;      // speech must be this many times louder than the noise floor
    this.busy = false;
  }

  async start() {
    this.stream = await navigator.mediaDevices.getUserMedia({
      // Noise suppression / echo cancellation reshape the spectrum, so the MFCCs stop looking
      // like the raw Speech Commands clips the SNN was trained on. Gain control only changes
      // loudness (MFCCs are min-max normalised per clip anyway) and keeps the VAD reliable.
      audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: true },
    });
    try { this.ctx = new AudioContext({ sampleRate: TARGET_SR }); }
    catch { this.ctx = new AudioContext(); } // some browsers refuse a custom rate; we resample below
    this.ratio = this.ctx.sampleRate / TARGET_SR;
    const url = URL.createObjectURL(new Blob([WORKLET], { type: 'application/javascript' }));
    await this.ctx.audioWorklet.addModule(url);
    this.src = this.ctx.createMediaStreamSource(this.stream);
    this.node = new AudioWorkletNode(this.ctx, 'tap');
    this.node.port.onmessage = (e) => this.push(e.data);
    this.src.connect(this.node);
    this.setState('listening');
  }

  stop() {
    this.stream?.getTracks().forEach((t) => t.stop());
    this.ctx?.close();
    this.ctx = null;
    clearTimeout(this.captureTimer);
    this.setState('off');
  }

  setState(s) { this.state = s; this.onState?.(s); }

  push(chunk) {
    const samples = this.ratio === 1 ? chunk : this.resample(chunk);
    // append to ring buffer
    for (let i = 0; i < samples.length; i++) {
      this.ring[this.writePos] = samples[i];
      this.writePos = (this.writePos + 1) % this.ring.length;
    }
    // frame-level RMS for the VAD
    const buf = new Float32Array(this.pending.length + samples.length);
    buf.set(this.pending); buf.set(samples, this.pending.length);
    let off = 0;
    while (off + FRAME <= buf.length) {
      let sum = 0;
      for (let i = off; i < off + FRAME; i++) sum += buf[i] * buf[i];
      this.onFrame(Math.sqrt(sum / FRAME), buf.subarray(off, off + FRAME));
      off += FRAME;
    }
    this.pending = buf.slice(off);
  }

  resample(chunk) {
    const outLen = Math.floor(chunk.length / this.ratio);
    const out = new Float32Array(outLen);
    for (let i = 0; i < outLen; i++) {
      const x = i * this.ratio, i0 = Math.floor(x), f = x - i0;
      out[i] = chunk[i0] * (1 - f) + (chunk[Math.min(i0 + 1, chunk.length - 1)] * f);
    }
    return out;
  }

  onFrame(rms, frame) {
    const threshold = Math.max(0.012, this.noiseFloor * this.sensitivity);
    this.onLevel?.(rms, threshold, frame);
    if (this.state === 'listening') {
      if (rms > threshold && !this.busy) {
        this.setState('capturing');
        this.captureTimer = setTimeout(() => this.emit(), POST_ONSET_MS);
      } else {
        // track background noise slowly (only while nobody is speaking)
        this.noiseFloor = this.noiseFloor * 0.995 + Math.min(rms, 0.05) * 0.005;
      }
    }
  }

  snapshot(seconds) {
    const n = Math.floor(TARGET_SR * seconds);
    const out = new Float32Array(n);
    let start = (this.writePos - n + this.ring.length) % this.ring.length;
    for (let i = 0; i < n; i++) out[i] = this.ring[(start + i) % this.ring.length];
    return out;
  }

  async emit() {
    if (!this.ctx) return;
    this.setState('processing');
    this.busy = true;
    try { await this.onUtterance(this.snapshot(CLIP_SECONDS)); }
    finally {
      this.busy = false;
      if (this.ctx) this.setState('listening');
    }
  }
}
