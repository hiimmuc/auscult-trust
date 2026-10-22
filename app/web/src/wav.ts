// WAV header reading (client-side validation before upload) and 16-bit PCM encoding of browser recordings.

export interface WavInfo {
  sampleRate: number;
  channels: number;
  bitsPerSample: number;
  format: string;
  durationSec: number;
}

const FORMATS: Record<number, string> = { 1: "PCM", 3: "IEEE float", 0xfffe: "extensible" };

export function readWavHeader(buf: ArrayBuffer): WavInfo {
  const v = new DataView(buf);
  const tag = (o: number) => String.fromCharCode(v.getUint8(o), v.getUint8(o + 1), v.getUint8(o + 2), v.getUint8(o + 3));
  if (buf.byteLength < 44 || tag(0) !== "RIFF" || tag(8) !== "WAVE") throw new Error("This is not a WAV file (no RIFF/WAVE header).");
  let fmt: WavInfo | null = null;
  let o = 12;
  while (o + 8 <= buf.byteLength) {
    const id = tag(o), size = v.getUint32(o + 4, true);
    if (id === "fmt ") {
      const code = v.getUint16(o + 8, true);
      fmt = { format: FORMATS[code] || `code ${code}`, channels: v.getUint16(o + 10, true), sampleRate: v.getUint32(o + 12, true),
              bitsPerSample: v.getUint16(o + 22, true), durationSec: 0 };
    } else if (id === "data" && fmt) {
      const bytes = Math.min(size, buf.byteLength - o - 8);
      fmt.durationSec = bytes / (fmt.sampleRate * fmt.channels * (fmt.bitsPerSample / 8));
      return fmt;
    }
    o += 8 + size + (size & 1);
  }
  throw new Error("WAV file has no readable format or data chunk.");
}

export function encodeWav(samples: Float32Array, sampleRate: number): Blob {
  const out = new DataView(new ArrayBuffer(44 + samples.length * 2));
  const str = (o: number, s: string) => [...s].forEach((c, i) => out.setUint8(o + i, c.charCodeAt(0)));
  str(0, "RIFF"); out.setUint32(4, 36 + samples.length * 2, true); str(8, "WAVE");
  str(12, "fmt "); out.setUint32(16, 16, true); out.setUint16(20, 1, true); out.setUint16(22, 1, true);
  out.setUint32(24, sampleRate, true); out.setUint32(28, sampleRate * 2, true); out.setUint16(32, 2, true); out.setUint16(34, 16, true);
  str(36, "data"); out.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) out.setInt16(44 + 2 * i, Math.max(-1, Math.min(1, samples[i])) * 0x7fff, true);
  return new Blob([out.buffer], { type: "audio/wav" });
}
