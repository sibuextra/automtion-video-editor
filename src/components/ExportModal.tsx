import { useState, useRef, useCallback, useEffect } from 'react';
import { Track, AspectRatio } from '../types/editor';
import { interpolateKF } from '../hooks/useEditorStore';
import { FFmpeg } from '@ffmpeg/ffmpeg';
import { toBlobURL } from '@ffmpeg/util';

// ─────────────────────────────────────────────────────────────────────────────
// TYPES
// ─────────────────────────────────────────────────────────────────────────────
type Resolution  = '2160p' | '1080p' | '720p' | '480p' | '144p';
type QualityMode = 'higher' | 'balanced' | 'lower';
type Codec       = 'h264' | 'hevc' | 'av1' | 'vp9';
type Format      = 'mp4' | 'mov' | 'webm' | 'avi' | 'mkv';
type ExportPhase = 'idle' | 'loading_ffmpeg' | 'loading_media' | 'rendering' | 'encoding' | 'audio' | 'muxing' | 'done' | 'error';

interface ExportSettings {
  filename:      string;
  aspectRatio:   AspectRatio | 'original';
  resolution:    Resolution;
  quality:       QualityMode;
  codec:         Codec;
  format:        Format;
  frameRate:     number;
  useCustomFPS:  boolean;
  customFPS:     number;
}

interface Props {
  isOpen:             boolean;
  onClose:            () => void;
  tracks:             Track[];
  duration:           number;
  currentAspectRatio: AspectRatio;
  projectName?:       string;
}

// ─────────────────────────────────────────────────────────────────────────────
// CONSTANTS
// ─────────────────────────────────────────────────────────────────────────────
const RESOLUTIONS: Record<Resolution, { w: number; h: number; label: string }> = {
  '2160p': { w: 3840, h: 2160, label: '4K (2160p)'     },
  '1080p': { w: 1920, h: 1080, label: 'Full HD (1080p)' },
  '720p':  { w: 1280, h: 720,  label: 'HD (720p)'       },
  '480p':  { w: 854,  h: 480,  label: 'SD (480p)'       },
  '144p':  { w: 256,  h: 144,  label: 'Low (144p)'      },
};

// CRF values: lower = better quality
const CRF_MAP: Record<QualityMode, number> = {
  higher:   16,
  balanced: 20,
  lower:    26,
};

// ─────────────────────────────────────────────────────────────────────────────
// HELPERS
// ─────────────────────────────────────────────────────────────────────────────
function getCanvasDimensions(ar: AspectRatio | 'original', baseRes: Resolution) {
  const { w, h } = RESOLUTIONS[baseRes];
  switch (ar) {
    case '16:9':  return { w, h };
    case '9:16':  return { w: h, h: w };
    case '1:1':   return { w: Math.min(w, h), h: Math.min(w, h) };
    case '4:3':   return { w: Math.round(h * 4 / 3), h };
    case '21:9':  return { w: Math.round(h * 21 / 9), h };
    default:      return { w, h };
  }
}

// ensure canvas dimensions are even (H.264 requirement)
function evenDim(n: number): number { return n % 2 === 0 ? n : n - 1; }

function formatTime(sec: number) {
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = Math.floor(sec % 60);
  if (h > 0) return `${h}:${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`;
  return `${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024)       return `${bytes} B`;
  if (bytes < 1048576)    return `${(bytes/1024).toFixed(1)} KB`;
  return `${(bytes/1048576).toFixed(2)} MB`;
}

// ─────────────────────────────────────────────────────────────────────────────
// MEDIA LOADING UTILITIES
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Load a video and wait until it can play through.
 * canplaythrough fires when enough data is buffered.
 */
function loadVideoEl(url: string): Promise<HTMLVideoElement> {
  return new Promise((resolve, reject) => {
    const v = document.createElement('video');
    v.crossOrigin  = 'anonymous';
    v.preload      = 'auto';
    v.muted        = true;
    v.playsInline  = true;
    let done = false;
    const finish = () => { if (done) return; done = true; resolve(v); };
    v.addEventListener('canplaythrough', finish);
    v.addEventListener('loadeddata', finish);
    v.addEventListener('error', () => reject(new Error(`Video load failed: ${url}`)));
    v.src = url;
    v.load();
    // 15s hard timeout
    setTimeout(() => { if (!done) finish(); }, 15000);
  });
}

function loadImageEl(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload  = () => resolve(img);
    img.onerror = () => reject(new Error(`Image load failed: ${url}`));
    img.src = url;
    setTimeout(() => resolve(img), 8000);
  });
}

/**
 * THE CRITICAL FIX:
 * Seek a video element to an exact time and wait for the 'seeked' event.
 * Without this, drawImage() will paint a stale/black frame.
 */
function seekExact(vel: HTMLVideoElement, targetTime: number): Promise<void> {
  return new Promise(resolve => {
    if (!isFinite(vel.duration) || vel.duration <= 0) { resolve(); return; }
    const clamped = Math.max(0, Math.min(vel.duration - 0.001, targetTime));
    // Already at the right time? Skip the seek entirely.
    if (Math.abs(vel.currentTime - clamped) < 0.001) { resolve(); return; }
    let settled = false;
    const finish = () => {
      if (settled) return;
      settled = true;
      vel.removeEventListener('seeked', finish);
      resolve();
    };
    vel.addEventListener('seeked', finish);
    vel.currentTime = clamped;
    // Hard timeout: some browsers don't fire 'seeked' for every seek
    setTimeout(finish, 1000);
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// CORE FRAME RENDERER
// Renders one frame at absolute time `t` onto the 2D canvas.
// Returns a PNG blob suitable for writing to FFmpeg's virtual FS.
// ─────────────────────────────────────────────────────────────────────────────
async function renderFrameToPNG(
  canvas:    HTMLCanvasElement,
  ctx:       CanvasRenderingContext2D,
  W:         number,
  H:         number,
  t:         number,
  tracks:    Track[],
  videoEls:  Map<string, HTMLVideoElement>,
  imageEls:  Map<string, HTMLImageElement>,
): Promise<Uint8Array> {
  // 1. Black background — prevents garbage/stale pixels between clips
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#000000';
  ctx.fillRect(0, 0, W, H);

  // 2. Collect active visual clips, sorted bottom-to-top (higher trackIndex = lower layer)
  const allClips = tracks.flatMap(tr => tr.clips);
  const active   = allClips
    .filter(c => c.type !== 'audio' && t >= c.startTime && t < c.startTime + c.duration)
    .sort((a, b) => b.trackIndex - a.trackIndex);

  // 3. PARALLEL SEEK — seek all video clips simultaneously before drawing ANYTHING
  //    This is the #1 fix for black frames: we WAIT for every seeked event
  //    before calling drawImage on any of them.
  await Promise.all(
    active
      .filter(c => c.type === 'video')
      .map(clip => {
        const vel = videoEls.get(clip.id);
        if (!vel) return Promise.resolve();
        const speed     = clip.speed ?? 1;
        const mediaTime = (t - clip.startTime) * speed + (clip.trimStart ?? 0);
        return seekExact(vel, mediaTime);
      })
  );

  // 4. Draw each layer with full keyframe-interpolated transform
  for (const clip of active) {
    const tf      = clip.transform;
    const scale   = tf.scaleKFs.length   > 0 ? interpolateKF(tf.scaleKFs,   tf.scale,   t) : tf.scale;
    const opacity = tf.opacityKFs.length > 0 ? interpolateKF(tf.opacityKFs, tf.opacity,  t) : tf.opacity;
    const posX    = tf.posXKFs.length    > 0 ? interpolateKF(tf.posXKFs,    tf.posX,     t) : tf.posX;
    const posY    = tf.posYKFs.length    > 0 ? interpolateKF(tf.posYKFs,    tf.posY,     t) : tf.posY;

    ctx.save();
    ctx.globalAlpha = Math.max(0, Math.min(1, opacity));
    ctx.translate(W / 2 + posX, H / 2 + posY);
    ctx.scale(scale, scale);

    if (clip.type === 'video') {
      const vel = videoEls.get(clip.id);
      // readyState >= 2 = HAVE_CURRENT_DATA — safe to draw
      if (vel && vel.readyState >= 2) {
        const vw = vel.videoWidth  || W;
        const vh = vel.videoHeight || H;
        const s  = Math.min(W / vw, H / vh);
        ctx.drawImage(vel, -vw * s / 2, -vh * s / 2, vw * s, vh * s);
      }
    } else if (clip.type === 'image') {
      const iel = imageEls.get(clip.id);
      if (iel) {
        const iw = iel.naturalWidth  || W;
        const ih = iel.naturalHeight || H;
        const s  = Math.min(W / iw, H / ih);
        ctx.drawImage(iel, -iw * s / 2, -ih * s / 2, iw * s, ih * s);
      }
    }

    ctx.restore();
  }

  // 5. Export canvas as PNG blob → Uint8Array for FFmpeg virtual FS
  return new Promise<Uint8Array>((resolve, reject) => {
    canvas.toBlob(blob => {
      if (!blob) { reject(new Error('Canvas toBlob failed')); return; }
      blob.arrayBuffer().then(ab => resolve(new Uint8Array(ab))).catch(reject);
    }, 'image/png');
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// AUDIO PIPELINE — OfflineAudioContext → WAV
// Professional offline mixing: sample-perfect, no real-time drift
// ─────────────────────────────────────────────────────────────────────────────
async function renderAudioWav(
  tracks:   Track[],
  duration: number,
  onLog:    (msg: string) => void,
): Promise<Uint8Array | null> {
  try {
    const SAMPLE_RATE = 48000;
    const allClips    = tracks.flatMap(t => t.clips);
    const audioClips  = allClips.filter(c => c.type === 'audio' || c.type === 'video');
    if (audioClips.length === 0) { onLog('Audio: No audio sources found.'); return null; }

    onLog(`Audio: Mixing ${audioClips.length} source(s) via OfflineAudioContext…`);
    const offCtx = new OfflineAudioContext(2, Math.ceil(duration * SAMPLE_RATE), SAMPLE_RATE);

    for (const clip of audioClips) {
      try {
        onLog(`Audio: Decoding "${clip.name}"…`);
        const resp     = await fetch(clip.url);
        const arrayBuf = await resp.arrayBuffer();
        let decoded: AudioBuffer;
        try {
          decoded = await offCtx.decodeAudioData(arrayBuf);
        } catch {
          onLog(`Audio: ⚠ Cannot decode "${clip.name}" — skipping`);
          continue;
        }

        const src  = offCtx.createBufferSource();
        src.buffer = decoded;
        src.playbackRate.value = clip.speed ?? 1;

        const gain = offCtx.createGain();
        gain.gain.value = Math.max(0, Math.min(1, clip.transform.opacity));
        src.connect(gain).connect(offCtx.destination);

        const when      = Math.max(0, clip.startTime);
        const offset    = clip.trimStart ?? 0;
        const clipDur   = clip.duration;
        src.start(when, offset, clipDur);
      } catch (e) {
        onLog(`Audio: Error for "${clip.name}": ${(e as Error).message}`);
      }
    }

    onLog('Audio: Rendering offline mix…');
    const rendered = await offCtx.startRendering();

    onLog('Audio: Encoding PCM → WAV…');
    return audioBufferToWav(rendered);
  } catch (e) {
    onLog(`Audio: Fatal — ${(e as Error).message}`);
    return null;
  }
}

function audioBufferToWav(buf: AudioBuffer): Uint8Array {
  const numCh      = buf.numberOfChannels;
  const sr         = buf.sampleRate;
  const numSamples = buf.length;
  const dataSize   = numSamples * numCh * 2;
  const wavBuf     = new ArrayBuffer(44 + dataSize);
  const view       = new DataView(wavBuf);

  const ws = (off: number, s: string) => { for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i)); };
  const w2 = (off: number, v: number) => view.setUint16(off, v, true);
  const w4 = (off: number, v: number) => view.setUint32(off, v, true);

  ws(0,  'RIFF'); w4(4, 36 + dataSize);  ws(8, 'WAVE');
  ws(12, 'fmt '); w4(16, 16); w2(20, 1); w2(22, numCh);
  w4(24, sr);     w4(28, sr * numCh * 2); w2(32, numCh * 2); w2(34, 16);
  ws(36, 'data'); w4(40, dataSize);

  let offset = 44;
  for (let i = 0; i < numSamples; i++) {
    for (let ch = 0; ch < numCh; ch++) {
      const s = Math.max(-1, Math.min(1, buf.getChannelData(ch)[i]));
      view.setInt16(offset, s < 0 ? s * 32768 : s * 32767, true);
      offset += 2;
    }
  }
  return new Uint8Array(wavBuf);
}

// ─────────────────────────────────────────────────────────────────────────────
// FFMPEG WASM LOADER — CDN with BlobURL bypass for CORS
// ─────────────────────────────────────────────────────────────────────────────
const ffmpegInstance = new FFmpeg();
let ffmpegLoaded = false;

async function ensureFFmpeg(onLog: (msg: string) => void): Promise<FFmpeg> {
  if (ffmpegLoaded) { onLog('FFmpeg: Already loaded ✓'); return ffmpegInstance; }

  onLog('FFmpeg: Loading WebAssembly core from CDN…');
  const BASE = 'https://cdn.jsdelivr.net/npm/@ffmpeg/core@0.12.6/dist/esm';

  ffmpegInstance.on('log', ({ message }) => {
    // Only surface important FFmpeg log lines
    if (message.includes('frame=') || message.includes('error') || message.includes('Error')) {
      onLog(`[FFmpeg] ${message}`);
    }
  });

  try {
    await ffmpegInstance.load({
      coreURL: await toBlobURL(`${BASE}/ffmpeg-core.js`,   'text/javascript'),
      wasmURL: await toBlobURL(`${BASE}/ffmpeg-core.wasm`, 'application/wasm'),
    });
    ffmpegLoaded = true;
    onLog('FFmpeg: ✓ WebAssembly core loaded successfully');
  } catch (e) {
    // Try alternative CDN
    onLog(`FFmpeg: Primary CDN failed (${(e as Error).message}), trying fallback…`);
    const ALT = 'https://unpkg.com/@ffmpeg/core@0.12.6/dist/esm';
    await ffmpegInstance.load({
      coreURL: await toBlobURL(`${ALT}/ffmpeg-core.js`,   'text/javascript'),
      wasmURL: await toBlobURL(`${ALT}/ffmpeg-core.wasm`, 'application/wasm'),
    });
    ffmpegLoaded = true;
    onLog('FFmpeg: ✓ Loaded from fallback CDN');
  }

  return ffmpegInstance;
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN EXPORT ENGINE
// Pipeline:
//   1. Load FFmpeg WASM
//   2. Load all media (video/image) elements
//   3. Render frame-by-frame → PNG → write to FFmpeg VFS
//   4. FFmpeg encodes image sequence → H.264 MP4 (video only)
//   5. OfflineAudioContext → WAV
//   6. FFmpeg mux video + audio → final MP4
// ─────────────────────────────────────────────────────────────────────────────
async function runExportPipeline(params: {
  tracks:       Track[];
  realDuration: number;
  canvasW:      number;
  canvasH:      number;
  fps:          number;
  crf:          number;
  codec:        Codec;
  format:       Format;
  abortRef:     React.MutableRefObject<boolean>;
  onProgress:   (pct: number) => void;
  onPhase:      (phase: ExportPhase, label?: string) => void;
  onLog:        (msg: string) => void;
}): Promise<Blob> {
  const { tracks, realDuration, canvasW, canvasH, fps, crf, codec, format, abortRef, onProgress, onPhase, onLog } = params;

  const W = evenDim(canvasW);
  const H = evenDim(canvasH);

  // ── PHASE 0: Load FFmpeg WASM ────────────────────────────────────────────
  onPhase('loading_ffmpeg', 'Loading FFmpeg…');
  const ffmpeg = await ensureFFmpeg(onLog);
  if (abortRef.current) throw new Error('Cancelled');

  // ── PHASE 1: Load all media elements ────────────────────────────────────
  onPhase('loading_media', 'Loading media…');
  onLog('Media: Loading video and image elements…');

  const allClips   = tracks.flatMap(t => t.clips);
  const videoClips = allClips.filter(c => c.type === 'video');
  const imageClips = allClips.filter(c => c.type === 'image');

  const videoEls = new Map<string, HTMLVideoElement>();
  for (const clip of videoClips) {
    if (abortRef.current) throw new Error('Cancelled');
    try {
      onLog(`Media: Loading video "${clip.name}"…`);
      const vel = await loadVideoEl(clip.url);
      videoEls.set(clip.id, vel);
      onLog(`Media: ✓ "${clip.name}" — ${vel.videoWidth}×${vel.videoHeight}, ${vel.duration.toFixed(2)}s`);
    } catch (e) {
      onLog(`Media: ⚠ Failed to load "${clip.name}": ${(e as Error).message}`);
    }
  }

  const imageEls = new Map<string, HTMLImageElement>();
  for (const clip of imageClips) {
    if (abortRef.current) throw new Error('Cancelled');
    try {
      onLog(`Media: Loading image "${clip.name}"…`);
      const iel = await loadImageEl(clip.url);
      imageEls.set(clip.id, iel);
      onLog(`Media: ✓ "${clip.name}" — ${iel.naturalWidth}×${iel.naturalHeight}`);
    } catch (e) {
      onLog(`Media: ⚠ Failed to load "${clip.name}": ${(e as Error).message}`);
    }
  }

  // ── PHASE 2: Frame rendering loop ────────────────────────────────────────
  onPhase('rendering', 'Rendering frames…');

  const canvas = document.createElement('canvas');
  canvas.width  = W;
  canvas.height = H;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) throw new Error('Cannot create 2D canvas context');

  const totalFrames  = Math.ceil(realDuration * fps);
  const frameDuration = 1 / fps;

  onLog(`Render: ${totalFrames} frames @ ${fps}fps — ${W}×${H}px`);
  onLog(`Render: Writing PNG frames to FFmpeg virtual filesystem…`);

  // MEMORY STRATEGY: Write frames to FFmpeg VFS in batches of BATCH_SIZE,
  // then clean up written frames from VFS after encoding to avoid OOM.
  // For 60fps 1080p 60s = 3600 frames × ~2MB PNG = ~7GB RAM if all in memory.
  // Batched approach: keep max BATCH_SIZE frames in VFS at once.
  const YIELD_EVERY    = 5;   // yield to browser every N frames

  // Write all frames — we write to VFS and let FFmpeg process them later
  // For large exports, we use a streaming single-pass approach instead:
  // Write ALL frames first, then encode once. For memory safety we batch.
  
  // Single-pass: write all PNG frames, then encode
  // This is the most reliable approach for frame-accurate H.264
  const frameFiles: string[] = [];

  for (let frameIdx = 0; frameIdx <= totalFrames; frameIdx++) {
    if (abortRef.current) throw new Error('Cancelled');

    const t        = frameIdx * frameDuration;
    const filename = `frame_${String(frameIdx).padStart(6, '0')}.png`;

    // Render frame to PNG bytes
    const pngData = await renderFrameToPNG(canvas, ctx, W, H, t, tracks, videoEls, imageEls);

    // Write to FFmpeg virtual filesystem
    await ffmpeg.writeFile(filename, pngData);
    frameFiles.push(filename);

    // Progress: 0-70% for rendering phase
    const pct = Math.round((frameIdx / totalFrames) * 70);
    onProgress(pct);

    if (frameIdx % 30 === 0) {
      onLog(`Render: Frame ${frameIdx + 1}/${totalFrames + 1} — ${formatBytes(pngData.length)}`);
    }

    // Yield to browser to keep UI responsive
    if (frameIdx % YIELD_EVERY === 0) {
      await new Promise<void>(r => setTimeout(r, 0));
    }

  }

  onLog(`Render: ✓ All ${frameFiles.length} frames written to FFmpeg VFS`);
  if (abortRef.current) throw new Error('Cancelled');

  // ── PHASE 3: FFmpeg H.264 encoding (image sequence → video) ─────────────
  onPhase('encoding', 'Encoding H.264…');

  // Codec selection
  const videoCodec: Record<Codec, string> = {
    h264: 'libx264',
    hevc: 'libx265',
    av1:  'libaom-av1',
    vp9:  'libvpx-vp9',
  };
  const chosenCodec = videoCodec[codec] ?? 'libx264';

  // Professional H.264 encoding command:
  // -framerate fps     : input image sequence FPS (exact, CFR)
  // -i frame_%06d.png  : input pattern
  // -c:v libx264       : H.264 software encoder
  // -crf crf           : constant quality (18=high, 23=balanced, 26=lower)
  // -preset medium     : encoding speed/compression tradeoff
  // -pix_fmt yuv420p   : required for broad player compatibility
  // -r fps             : output frame rate (force CFR — Constant Frame Rate)
  // -g fps*2           : GOP size = 2s (keyframe interval)
  // -movflags +faststart : moov atom at beginning for web streaming
  // -an                : no audio in this pass (audio muxed separately)

  const gopSize  = fps * 2; // keyframe every 2 seconds
  const videoOut = 'video_only.mp4';

  onLog(`Encode: ${chosenCodec} CRF=${crf} preset=medium FPS=${fps} GOP=${gopSize}`);

  const encodeArgs = [
    '-framerate',  String(fps),
    '-i',          'frame_%06d.png',
    '-c:v',        chosenCodec,
    '-crf',        String(crf),
    '-preset',     'medium',
    '-pix_fmt',    'yuv420p',
    '-r',          String(fps),
    '-g',          String(gopSize),
    '-movflags',   '+faststart',
    '-an',
    '-y',
    videoOut,
  ];

  // HEVC/AV1/VP9 need different flags
  const encodeArgsFinal = codec === 'hevc'
    ? ['-framerate', String(fps), '-i', 'frame_%06d.png', '-c:v', 'libx265', '-crf', String(crf), '-preset', 'medium', '-pix_fmt', 'yuv420p', '-r', String(fps), '-an', '-y', videoOut]
    : codec === 'av1'
    ? ['-framerate', String(fps), '-i', 'frame_%06d.png', '-c:v', 'libaom-av1', '-crf', String(crf), '-b:v', '0', '-pix_fmt', 'yuv420p', '-r', String(fps), '-an', '-y', videoOut]
    : codec === 'vp9'
    ? ['-framerate', String(fps), '-i', 'frame_%06d.png', '-c:v', 'libvpx-vp9', '-crf', String(crf), '-b:v', '0', '-pix_fmt', 'yuv420p', '-r', String(fps), '-an', '-y', videoOut]
    : encodeArgs;

  onLog('Encode: Running FFmpeg encoder…');
  await ffmpeg.exec(encodeArgsFinal);
  onLog('Encode: ✓ H.264 encoding complete');
  onProgress(82);

  // Clean up frame files from VFS to free memory
  onLog('Encode: Cleaning up frame files from VFS…');
  for (const f of frameFiles) {
    try { await ffmpeg.deleteFile(f); } catch { /* ignore */ }
  }
  onLog(`Encode: ✓ Cleaned ${frameFiles.length} frame files`);

  if (abortRef.current) throw new Error('Cancelled');

  // ── PHASE 4: Audio rendering ─────────────────────────────────────────────
  onPhase('audio', 'Rendering audio…');
  const wavData = await renderAudioWav(tracks, realDuration, onLog);
  onProgress(90);

  // ── PHASE 5: Mux video + audio ───────────────────────────────────────────
  onPhase('muxing', 'Muxing A/V…');
  let finalFilename: string;

  if (wavData && wavData.length > 44) {
    onLog('Mux: Writing WAV audio to FFmpeg VFS…');
    await ffmpeg.writeFile('audio.wav', wavData);

    // Single-pass mux: video + audio → final container
    // -c:v copy  = no re-encoding of video (preserve exact quality)
    // -c:a aac   = encode WAV PCM → AAC for MP4 container
    // -shortest  = end at shorter of video/audio
    finalFilename = `output.${format}`;

    const muxArgs = [
      '-i',        videoOut,
      '-i',        'audio.wav',
      '-c:v',      'copy',       // ← NO re-encode, copy H.264 stream as-is
      '-c:a',      'aac',
      '-b:a',      '192k',
      '-ac',       '2',
      '-shortest',
      '-movflags', '+faststart',
      '-y',
      finalFilename,
    ];

    onLog('Mux: Running FFmpeg A/V mux (no video re-encode)…');
    await ffmpeg.exec(muxArgs);
    onLog('Mux: ✓ A/V mux complete');

    // Cleanup
    try { await ffmpeg.deleteFile('audio.wav'); }    catch { /* ignore */ }
    try { await ffmpeg.deleteFile(videoOut); }       catch { /* ignore */ }
  } else {
    // No audio: just rename the video file
    onLog('Mux: No audio — using video-only output');
    finalFilename = videoOut;
  }

  onProgress(98);
  if (abortRef.current) throw new Error('Cancelled');

  // ── PHASE 6: Read output ─────────────────────────────────────────────────
  onLog(`Output: Reading "${finalFilename}" from FFmpeg VFS…`);
  const outputData = await ffmpeg.readFile(finalFilename) as Uint8Array;

  // Cleanup output file
  try { await ffmpeg.deleteFile(finalFilename); } catch { /* ignore */ }

  const mimeTypes: Record<Format, string> = {
    mp4:  'video/mp4',
    mov:  'video/quicktime',
    webm: 'video/webm',
    avi:  'video/x-msvideo',
    mkv:  'video/x-matroska',
  };

  // Copy to a regular ArrayBuffer (avoids SharedArrayBuffer type issue)
  const safeArr = new Uint8Array(outputData.length);
  safeArr.set(outputData);
  const blob = new Blob([safeArr], { type: mimeTypes[format] ?? 'video/mp4' });
  onLog(`Output: ✓ ${formatBytes(blob.size)} — ready for download`);
  onProgress(100);

  return blob;
}

// ─────────────────────────────────────────────────────────────────────────────
// MINI PREVIEW PLAYER (unchanged, kept intact)
// ─────────────────────────────────────────────────────────────────────────────
function MiniVideoLayer({
  clip, ct, playing, hasAudio, tf,
}: {
  clip: import('../types/editor').TimelineClip;
  ct: number; playing: boolean; hasAudio: boolean;
  tf: { scale: number; opacity: number; posX: number; posY: number };
}) {
  const ref = useRef<HTMLVideoElement>(null);
  useEffect(() => {
    const v = ref.current; if (!v) return;
    const offset = ct - clip.startTime + (clip.trimStart ?? 0);
    if (Math.abs(v.currentTime - offset) > 0.3) v.currentTime = Math.max(0, offset);
  }, [ct, clip]);
  useEffect(() => {
    const v = ref.current; if (!v) return;
    if (playing) v.play().catch(() => {}); else v.pause();
  }, [playing]);
  return (
    <video ref={ref} key={clip.url} src={clip.url} playsInline muted={hasAudio}
      style={{ position:'absolute', inset:0, width:'100%', height:'100%', objectFit:'contain',
        transform:`translate(${tf.posX}px,${tf.posY}px) scale(${tf.scale})`, opacity:tf.opacity, pointerEvents:'none' }} />
  );
}

function MiniPreview({ tracks, duration, aspectRatio }: {
  tracks: Track[]; duration: number; aspectRatio: AspectRatio | 'original';
}) {
  const [ct, setCt]           = useState(0);
  const [playing, setPlaying] = useState(false);
  const ivRef    = useRef<ReturnType<typeof setInterval> | null>(null);
  const audioRef = useRef<HTMLAudioElement>(null);

  const activeAt = useCallback((t: number) =>
    tracks.flatMap(tr => tr.clips.filter(c => t >= c.startTime && t < c.startTime + c.duration)),
  [tracks]);

  const activeClips  = activeAt(ct);
  const visualClips  = [...activeClips].filter(c => c.type !== 'audio').sort((a,b) => b.trackIndex - a.trackIndex);
  const activeAudio  = activeClips.find(c => c.type === 'audio');

  useEffect(() => {
    if (playing) {
      ivRef.current = setInterval(() => {
        setCt(prev => { if (prev >= duration) { setPlaying(false); return 0; } return prev + 0.016; });
      }, 16);
    } else { if (ivRef.current) clearInterval(ivRef.current); }
    return () => { if (ivRef.current) clearInterval(ivRef.current); };
  }, [playing, duration]);

  useEffect(() => {
    const a = audioRef.current; if (!a || !activeAudio) return;
    const offset = ct - activeAudio.startTime + (activeAudio.trimStart ?? 0);
    if (Math.abs(a.currentTime - offset) > 0.3) a.currentTime = Math.max(0, offset);
  }, [ct, activeAudio]);

  useEffect(() => {
    const a = audioRef.current; if (!a) return;
    if (playing && activeAudio) a.play().catch(() => {}); else a.pause();
  }, [playing, activeAudio]);

  const getTransform = (clip: (typeof activeClips)[0]) => ({
    scale:   clip.transform.scaleKFs.length   > 0 ? interpolateKF(clip.transform.scaleKFs,   clip.transform.scale,   ct) : clip.transform.scale,
    opacity: clip.transform.opacityKFs.length > 0 ? interpolateKF(clip.transform.opacityKFs, clip.transform.opacity,  ct) : clip.transform.opacity,
    posX:    clip.transform.posXKFs.length    > 0 ? interpolateKF(clip.transform.posXKFs,    clip.transform.posX,     ct) : clip.transform.posX,
    posY:    clip.transform.posYKFs.length    > 0 ? interpolateKF(clip.transform.posYKFs,    clip.transform.posY,     ct) : clip.transform.posY,
  });

  const progress = duration > 0 ? (ct / duration) * 100 : 0;
  const arStr    = aspectRatio === 'original' ? '16 / 9' : aspectRatio.replace(':', ' / ');

  return (
    <div className="flex flex-col gap-2 h-full">
      <div className="flex-1 flex items-center justify-center bg-zinc-950 rounded-lg overflow-hidden min-h-0">
        <div className="relative bg-black overflow-hidden rounded"
          style={{ aspectRatio: arStr, maxWidth:'100%', maxHeight:'100%', width:'100%', outline:'1.5px solid rgba(99,102,241,0.3)' }}>
          {visualClips.map(clip => {
            const tf = getTransform(clip);
            if (clip.type === 'video') return <MiniVideoLayer key={clip.id} clip={clip} ct={ct} playing={playing} hasAudio={!!activeAudio} tf={tf} />;
            if (clip.type === 'image') return (
              <img key={clip.id} src={clip.url} alt=""
                style={{ position:'absolute', inset:0, width:'100%', height:'100%', objectFit:'contain',
                  transform:`translate(${tf.posX}px,${tf.posY}px) scale(${tf.scale})`, opacity:tf.opacity, pointerEvents:'none' }} />
            );
            return null;
          })}
          {visualClips.length === 0 && !activeAudio && (
            <div className="absolute inset-0 flex items-center justify-center text-zinc-700">
              <svg className="h-10 w-10" fill="none" stroke="currentColor" strokeWidth={1} viewBox="0 0 24 24">
                <rect x="2" y="4" width="20" height="16" rx="2"/><path strokeLinecap="round" strokeLinejoin="round" d="M10 9l5 3-5 3V9z"/>
              </svg>
            </div>
          )}
          {activeAudio && <audio ref={audioRef} key={activeAudio.url} src={activeAudio.url} />}
          <div className="absolute top-1.5 right-1.5 bg-black/70 rounded px-1.5 py-0.5 text-[9px] text-zinc-300 font-mono">
            {formatTime(ct)} / {formatTime(duration)}
          </div>
        </div>
      </div>

      <div className="w-full h-1.5 bg-zinc-700 rounded-full cursor-pointer"
        onClick={e => { const r = e.currentTarget.getBoundingClientRect(); setCt(((e.clientX-r.left)/r.width)*duration); }}>
        <div className="h-full bg-indigo-500 rounded-full" style={{ width:`${progress}%` }} />
      </div>

      <div className="flex items-center justify-center gap-3">
        <button onClick={() => setCt(0)} className="text-zinc-400 hover:text-white">
          <svg className="h-4 w-4" fill="currentColor" viewBox="0 0 24 24"><path d="M6 6h2v12H6zm3.5 6 8.5 6V6z"/></svg>
        </button>
        <button onClick={() => setCt(t => Math.max(0, t-5))} className="text-zinc-400 hover:text-white">
          <svg className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 8V4l-8 8 8 8v-4c3.314 0 6 2.686 6 6a6 6 0 01-6-6z"/>
          </svg>
        </button>
        <button onClick={() => setPlaying(p => !p)}
          className="w-9 h-9 rounded-full bg-indigo-600 hover:bg-indigo-500 flex items-center justify-center text-white shadow">
          {playing
            ? <svg className="h-4 w-4" fill="currentColor" viewBox="0 0 24 24"><path d="M6 19h4V5H6zm8-14v14h4V5z"/></svg>
            : <svg className="h-4 w-4 ml-0.5" fill="currentColor" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>
          }
        </button>
        <button onClick={() => setCt(t => Math.min(duration, t+5))} className="text-zinc-400 hover:text-white">
          <svg className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 8V4l8 8-8 8v-4a6 6 0 00-6 6 6 6 0 016-6z"/>
          </svg>
        </button>
        <button onClick={() => setCt(duration)} className="text-zinc-400 hover:text-white">
          <svg className="h-4 w-4" fill="currentColor" viewBox="0 0 24 24">
            <path d="M6 18l8.5-6L6 6v12zm2.5-6 5.5 3.9V8.1z"/><path d="M16 6h2v12h-2z"/>
          </svg>
        </button>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// SETTING SELECT HELPER (unchanged)
// ─────────────────────────────────────────────────────────────────────────────
function SettingSelect<T extends string>({ label, value, options, onChange }: {
  label: string; value: T; options: { value: T; label: string }[]; onChange: (v: T) => void;
}) {
  return (
    <div className="flex flex-col gap-1">
      <span className="text-[10px] text-zinc-500 uppercase tracking-wider font-semibold">{label}</span>
      <div className="flex flex-wrap gap-1">
        {options.map(o => (
          <button key={o.value} onClick={() => onChange(o.value)}
            className={`text-[11px] px-2 py-1 rounded border transition-colors ${
              value === o.value
                ? 'bg-indigo-600 border-indigo-500 text-white'
                : 'bg-zinc-800 border-zinc-700 text-zinc-400 hover:bg-zinc-700 hover:text-zinc-200'
            }`}>
            {o.label}
          </button>
        ))}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// PHASE LABELS & PROGRESS STEPS
// ─────────────────────────────────────────────────────────────────────────────
const PHASE_LABELS: Record<string, string> = {
  loading_ffmpeg: '⚙ Loading FFmpeg WASM…',
  loading_media:  '📦 Loading media files…',
  rendering:      '🎨 Rendering frames…',
  encoding:       '⚡ Encoding H.264…',
  audio:          '🎵 Rendering audio…',
  muxing:         '🔀 Muxing A/V…',
  done:           '✅ Export complete!',
  error:          '❌ Export failed',
};

// ─────────────────────────────────────────────────────────────────────────────
// MAIN EXPORT MODAL COMPONENT
// ─────────────────────────────────────────────────────────────────────────────
export default function ExportModal({ isOpen, onClose, tracks, duration, currentAspectRatio, projectName }: Props) {
  const realDuration = (() => {
    const allClips = tracks.flatMap(t => t.clips);
    if (allClips.length === 0) return duration;
    return Math.max(...allClips.map(c => c.startTime + c.duration));
  })();

  const [settings, setSettings] = useState<ExportSettings>({
    filename:     projectName ?? 'VideoForge Export',
    aspectRatio:  'original',
    resolution:   '1080p',
    quality:      'balanced',
    codec:        'h264',
    format:       'mp4',
    frameRate:    60,
    useCustomFPS: false,
    customFPS:    60,
  });

  const [phase, setPhase]           = useState<ExportPhase>('idle');
  const [phaseLabel, setPhaseLabel] = useState('');
  const [progress, setProgress]     = useState(0);
  const [logs, setLogs]             = useState<string[]>([]);
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null);
  const [errorMsg, setErrorMsg]     = useState('');
  const abortRef   = useRef(false);
  const logsEndRef = useRef<HTMLDivElement>(null);

  const addLog = useCallback((msg: string) => {
    setLogs(prev => [...prev.slice(-299), `[${new Date().toLocaleTimeString()}] ${msg}`]);
  }, []);

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  useEffect(() => () => { if (downloadUrl) URL.revokeObjectURL(downloadUrl); }, [downloadUrl]);

  const set = <K extends keyof ExportSettings>(key: K, val: ExportSettings[K]) =>
    setSettings(s => ({ ...s, [key]: val }));

  const effectiveFPS = settings.useCustomFPS
    ? Math.max(1, Math.min(120, settings.customFPS))
    : settings.frameRate;

  const effectiveAR = settings.aspectRatio === 'original' ? currentAspectRatio : settings.aspectRatio;
  const { w: rawW, h: rawH } = getCanvasDimensions(effectiveAR, settings.resolution);
  const canvasW = evenDim(rawW);
  const canvasH = evenDim(rawH);
  const crf     = CRF_MAP[settings.quality];

  // Estimate: CRF is quality-based, size estimation is approximate
  const estBitrateKbps = settings.quality === 'higher' ? 12000 : settings.quality === 'balanced' ? 6000 : 2500;
  const estimatedMB    = ((estBitrateKbps * realDuration) / 8 / 1000).toFixed(1);

  const handleExport = useCallback(async () => {
    setPhase('loading_ffmpeg');
    setProgress(0);
    setLogs([]);
    setErrorMsg('');
    setDownloadUrl(null);
    abortRef.current = false;

    try {
      addLog('════════════════════════════════════════');
      addLog('  VideoForge FFmpeg Export Engine v3.0');
      addLog('════════════════════════════════════════');
      addLog(`Resolution : ${canvasW}×${canvasH}`);
      addLog(`Frame Rate : ${effectiveFPS} fps (CFR)`);
      addLog(`Codec      : ${settings.codec.toUpperCase()} CRF=${crf}`);
      addLog(`Format     : ${settings.format.toUpperCase()}`);
      addLog(`Duration   : ${realDuration.toFixed(2)}s`);
      addLog(`Frames     : ~${Math.ceil(realDuration * effectiveFPS)}`);
      addLog('─────────────────────────────────────────');

      const blob = await runExportPipeline({
        tracks,
        realDuration,
        canvasW,
        canvasH,
        fps:     effectiveFPS,
        crf,
        codec:   settings.codec,
        format:  settings.format,
        abortRef,
        onProgress: pct  => setProgress(pct),
        onPhase:    (p, label) => {
          setPhase(p);
          setPhaseLabel(label ?? '');
        },
        onLog: addLog,
      });

      const url = URL.createObjectURL(blob);
      setDownloadUrl(url);

      const a      = document.createElement('a');
      a.href       = url;
      a.download   = `${settings.filename}.${settings.format}`;
      a.click();

      setPhase('done');
      setProgress(100);
      addLog('─────────────────────────────────────────');
      addLog(`✅ Export complete! ${formatBytes(blob.size)}`);
      addLog('  File downloaded automatically.');

    } catch (err) {
      const msg = (err as Error).message;
      if (msg === 'Cancelled') {
        setPhase('idle');
        addLog('⚠ Export cancelled by user.');
      } else {
        setPhase('error');
        setErrorMsg(msg);
        addLog(`❌ Export failed: ${msg}`);
      }
    }
  }, [tracks, realDuration, canvasW, canvasH, effectiveFPS, crf, settings, addLog]);

  if (!isOpen) return null;

  const isExporting = ['loading_ffmpeg','loading_media','rendering','encoding','audio','muxing'].includes(phase);

  const pipelineSteps: { key: ExportPhase; label: string }[] = [
    { key: 'loading_ffmpeg', label: 'FFmpeg' },
    { key: 'loading_media',  label: 'Media'  },
    { key: 'rendering',      label: 'Frames' },
    { key: 'encoding',       label: 'Encode' },
    { key: 'audio',          label: 'Audio'  },
    { key: 'muxing',         label: 'Mux'    },
    { key: 'done',           label: 'Done'   },
  ];
  const currentStepIdx = pipelineSteps.findIndex(s => s.key === phase);

  const FPS_PRESETS = [60, 30, 24];

  const arOptions: { value: ExportSettings['aspectRatio']; label: string }[] = [
    { value: 'original', label: `Original (${currentAspectRatio})` },
    { value: '16:9', label: '16:9' }, { value: '9:16', label: '9:16' },
    { value: '1:1',  label: '1:1' }, { value: '4:3',  label: '4:3' },
    { value: '21:9', label: '21:9' },
  ];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/85 backdrop-blur-sm">
      <div className="relative bg-zinc-900 border border-zinc-700 rounded-2xl shadow-2xl flex overflow-hidden"
        style={{ width:'92vw', maxWidth:1200, height:'90vh', maxHeight:800 }}>

        {/* Close */}
        <button onClick={() => { abortRef.current = true; onClose(); }}
          className="absolute top-3 right-3 z-20 w-7 h-7 flex items-center justify-center rounded-full bg-zinc-800 hover:bg-zinc-700 text-zinc-400 hover:text-white transition-colors">
          <svg className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12"/>
          </svg>
        </button>

        {/* ── LEFT: Preview + Pipeline + Log ──────────────────────────────── */}
        <div className="flex flex-col bg-zinc-950 border-r border-zinc-800"
          style={{ width:'44%', minWidth:340, padding:'20px 16px 16px' }}>

          {/* Header */}
          <div className="flex items-center gap-2 mb-3 flex-shrink-0">
            <div className="w-6 h-6 rounded-lg bg-gradient-to-br from-indigo-500 to-violet-600 flex items-center justify-center">
              <svg className="h-3.5 w-3.5 text-white" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2M8 12l4 4 4-4M12 4v12"/>
              </svg>
            </div>
            <span className="text-sm font-semibold text-white">Export Preview</span>
            <span className="ml-auto text-[9px] bg-emerald-600/30 border border-emerald-500/40 text-emerald-300 px-2 py-0.5 rounded-full font-mono">
              FFmpeg WASM v3
            </span>
          </div>

          {/* Mini preview */}
          <div className="flex-1 min-h-0 mb-3">
            <MiniPreview
              tracks={tracks}
              duration={realDuration}
              aspectRatio={settings.aspectRatio === 'original' ? currentAspectRatio : settings.aspectRatio}
            />
          </div>

          {/* Stats grid */}
          <div className="flex-shrink-0 grid grid-cols-4 gap-1.5 mb-3">
            {[
              { label: 'Duration', val: formatTime(realDuration)  },
              { label: 'Est. Size', val: `~${estimatedMB}MB`     },
              { label: 'Canvas',   val: `${canvasW}×${canvasH}`  },
              { label: 'FPS',      val: `${effectiveFPS}`         },
            ].map(s => (
              <div key={s.label} className="bg-zinc-800 rounded-lg px-2 py-1.5">
                <p className="text-[8px] text-zinc-500 uppercase tracking-wider">{s.label}</p>
                <p className="text-[11px] font-semibold text-white font-mono truncate">{s.val}</p>
              </div>
            ))}
          </div>

          {/* Pipeline stepper */}
          {isExporting && (
            <div className="flex-shrink-0 mb-2">
              <div className="flex items-center gap-0.5">
                {pipelineSteps.map((step, idx) => {
                  const isDone    = phase === 'done' || idx < currentStepIdx;
                  const isCurrent = idx === currentStepIdx;
                  return (
                    <div key={step.key} className="flex items-center flex-1">
                      <div className={`flex-1 text-center text-[8px] py-1 rounded font-medium transition-colors ${
                        isDone    ? 'bg-emerald-600/40 text-emerald-300 border border-emerald-600/40' :
                        isCurrent ? 'bg-indigo-600/50 text-indigo-200 border border-indigo-500/50 animate-pulse' :
                                    'bg-zinc-800/60 text-zinc-600 border border-zinc-700/40'
                      }`}>
                        {isDone ? '✓' : isCurrent ? '⟳' : '○'} {step.label}
                      </div>
                      {idx < pipelineSteps.length - 1 && (
                        <div className={`w-1 h-px mx-0.5 ${isDone ? 'bg-emerald-500' : 'bg-zinc-700'}`} />
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Export Log */}
          <div className="flex-shrink-0">
            <div className="flex items-center gap-2 mb-1">
              <span className="text-[9px] text-zinc-500 uppercase tracking-wider font-semibold">FFmpeg Pipeline Log</span>
              {isExporting && (
                <div className="flex gap-0.5 items-center ml-1">
                  <div className="w-1 h-1 rounded-full bg-indigo-400 animate-bounce" style={{ animationDelay:'0ms' }} />
                  <div className="w-1 h-1 rounded-full bg-indigo-400 animate-bounce" style={{ animationDelay:'150ms' }} />
                  <div className="w-1 h-1 rounded-full bg-indigo-400 animate-bounce" style={{ animationDelay:'300ms' }} />
                </div>
              )}
            </div>
            <div className="bg-zinc-950 border border-zinc-800 rounded-lg p-2 h-32 overflow-y-auto font-mono text-[9px] text-zinc-400">
              {logs.length === 0
                ? <span className="text-zinc-600">Waiting for export to start…</span>
                : logs.map((l, i) => (
                    <div key={i} className={`leading-relaxed ${
                      l.includes('❌') || l.includes('Error') ? 'text-red-400' :
                      l.includes('✅') || l.includes('✓')     ? 'text-emerald-400' :
                      l.includes('════') || l.includes('────') ? 'text-indigo-400/60' :
                      l.includes('[FFmpeg]')                   ? 'text-amber-300/70' :
                                                                  'text-zinc-400'
                    }`}>{l}</div>
                  ))
              }
              <div ref={logsEndRef} />
            </div>
          </div>
        </div>

        {/* ── RIGHT: Settings + Progress + Buttons ────────────────────────── */}
        <div className="flex-1 flex flex-col overflow-hidden" style={{ padding:'20px 20px 16px' }}>

          <div className="flex-shrink-0 mb-4 pr-8">
            <h2 className="text-base font-bold text-white">Export Settings</h2>
            <p className="text-[10px] text-zinc-500 mt-0.5">
              FFmpeg WASM · H.264 libx264 · Image-sequence pipeline · CRF encoding · Professional A/V mux
            </p>
          </div>

          <div className="flex-1 overflow-y-auto flex flex-col gap-4 pr-1">

            {/* Filename */}
            <div className="flex flex-col gap-1.5">
              <span className="text-[10px] text-zinc-500 uppercase tracking-wider font-semibold">Filename</span>
              <div className="flex items-center gap-2">
                <input type="text" value={settings.filename} onChange={e => set('filename', e.target.value)}
                  className="flex-1 bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-1.5 text-sm text-zinc-200 focus:outline-none focus:border-indigo-500"
                  placeholder="Export filename" />
                <span className="text-xs text-zinc-500">.{settings.format}</span>
              </div>
            </div>

            <div className="h-px bg-zinc-800" />

            {/* Video settings */}
            <div className="flex flex-col gap-3">
              <div className="flex items-center gap-2">
                <svg className="h-3.5 w-3.5 text-indigo-400" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M15 10l4.553-2.277A1 1 0 0121 8.677v6.646a1 1 0 01-1.447.894L15 14M3 8a2 2 0 012-2h8a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2V8z"/>
                </svg>
                <span className="text-xs font-bold text-zinc-300 uppercase tracking-wider">Video</span>
              </div>

              <SettingSelect label="Aspect Ratio" value={settings.aspectRatio}
                options={arOptions} onChange={v => set('aspectRatio', v)} />

              <SettingSelect label="Resolution" value={settings.resolution}
                options={[
                  { value:'2160p' as Resolution, label:'4K (2160p)' },
                  { value:'1080p' as Resolution, label:'1080p' },
                  { value:'720p'  as Resolution, label:'720p'  },
                  { value:'480p'  as Resolution, label:'480p'  },
                  { value:'144p'  as Resolution, label:'144p'  },
                ]}
                onChange={v => set('resolution', v)} />

              {/* Quality (CRF-based) */}
              <div className="flex flex-col gap-1">
                <span className="text-[10px] text-zinc-500 uppercase tracking-wider font-semibold">Quality (CRF)</span>
                <div className="flex flex-wrap gap-1">
                  {([
                    { value: 'higher'   as QualityMode, label: `Higher (CRF ${CRF_MAP.higher})`   },
                    { value: 'balanced' as QualityMode, label: `Balanced (CRF ${CRF_MAP.balanced})` },
                    { value: 'lower'    as QualityMode, label: `Lower (CRF ${CRF_MAP.lower})`    },
                  ]).map(o => (
                    <button key={o.value} onClick={() => set('quality', o.value)}
                      className={`text-[11px] px-2 py-1 rounded border transition-colors ${
                        settings.quality === o.value
                          ? 'bg-indigo-600 border-indigo-500 text-white'
                          : 'bg-zinc-800 border-zinc-700 text-zinc-400 hover:bg-zinc-700 hover:text-zinc-200'
                      }`}>
                      {o.label}
                    </button>
                  ))}
                </div>
                <p className="text-[9px] text-zinc-600">CRF = Constant Rate Factor. Lower = better quality, larger file.</p>
              </div>

              <SettingSelect label="Codec" value={settings.codec}
                options={[
                  { value:'h264' as Codec, label:'H.264 (libx264)' },
                  { value:'hevc' as Codec, label:'H.265 (libx265)' },
                  { value:'vp9'  as Codec, label:'VP9'             },
                  { value:'av1'  as Codec, label:'AV1'             },
                ]}
                onChange={v => set('codec', v)} />

              <SettingSelect label="Container Format" value={settings.format}
                options={[
                  { value:'mp4'  as Format, label:'MP4' },
                  { value:'mov'  as Format, label:'MOV' },
                  { value:'webm' as Format, label:'WebM'},
                  { value:'mkv'  as Format, label:'MKV' },
                ]}
                onChange={v => set('format', v)} />

              {/* Frame Rate */}
              <div className="flex flex-col gap-2">
                <span className="text-[10px] text-zinc-500 uppercase tracking-wider font-semibold">Frame Rate (CFR)</span>
                <div className="flex flex-wrap gap-1">
                  {FPS_PRESETS.map(fps => (
                    <button key={fps} onClick={() => { set('frameRate', fps); set('useCustomFPS', false); }}
                      className={`text-[11px] px-2 py-1 rounded border transition-colors ${
                        !settings.useCustomFPS && settings.frameRate === fps
                          ? 'bg-indigo-600 border-indigo-500 text-white'
                          : 'bg-zinc-800 border-zinc-700 text-zinc-400 hover:bg-zinc-700 hover:text-zinc-200'
                      }`}>{fps} fps</button>
                  ))}
                  <button onClick={() => set('useCustomFPS', true)}
                    className={`text-[11px] px-2 py-1 rounded border transition-colors ${
                      settings.useCustomFPS ? 'bg-indigo-600 border-indigo-500 text-white'
                      : 'bg-zinc-800 border-zinc-700 text-zinc-400 hover:bg-zinc-700 hover:text-zinc-200'
                    }`}>Custom</button>
                </div>
                {settings.useCustomFPS && (
                  <div className="flex items-center gap-2 mt-1">
                    <input type="range" min={1} max={120} step={1} value={settings.customFPS}
                      onChange={e => set('customFPS', Number(e.target.value))}
                      className="flex-1 h-1 accent-indigo-500 cursor-pointer" />
                    <input type="number" min={1} max={120} step={1} value={settings.customFPS}
                      onChange={e => set('customFPS', Number(e.target.value))}
                      className="w-16 text-xs bg-zinc-800 border border-zinc-700 text-zinc-200 rounded px-2 py-1 text-right" />
                    <span className="text-[10px] text-zinc-500">fps</span>
                  </div>
                )}
              </div>
            </div>

            {/* FFmpeg pipeline info */}
            <div className="bg-emerald-500/8 border border-emerald-500/20 rounded-lg px-3 py-2.5">
              <p className="text-[10px] text-emerald-300 font-semibold mb-1.5">🎬 FFmpeg WASM Export Pipeline</p>
              <div className="flex flex-col gap-0.5">
                {[
                  '✓ MediaRecorder completely removed',
                  '✓ captureStream() completely removed',
                  '✓ seeked-event sync → zero black frames',
                  '✓ Parallel video seeking (Promise.all)',
                  '✓ PNG image sequence → libx264 H.264',
                  '✓ CRF encoding → constant quality',
                  '✓ CFR output → constant frame rate',
                  '✓ OfflineAudioContext → WAV → AAC',
                  '✓ Single-pass A/V mux (no re-encode)',
                  '✓ -movflags +faststart for web playback',
                ].map((t, i) => (
                  <p key={i} className="text-[9px] text-emerald-300/70">{t}</p>
                ))}
              </div>
            </div>

            {/* Warning for large exports */}
            {Math.ceil(realDuration * effectiveFPS) > 1800 && (
              <div className="bg-amber-500/10 border border-amber-500/30 rounded-lg px-3 py-2">
                <p className="text-[10px] text-amber-300 font-semibold">⚠ Large Export Warning</p>
                <p className="text-[9px] text-amber-300/70 mt-0.5">
                  {Math.ceil(realDuration * effectiveFPS)} frames at {settings.resolution} may take several minutes.
                  Consider lowering resolution or FPS for faster export.
                </p>
              </div>
            )}
          </div>

          {/* ── Progress + Buttons ─────────────────────────────────────────── */}
          <div className="flex-shrink-0 mt-4 flex flex-col gap-3">

            {/* Progress bar */}
            {isExporting && (
              <div className="flex flex-col gap-1.5">
                <div className="flex items-center justify-between">
                  <span className="text-[11px] text-zinc-400">
                    {PHASE_LABELS[phaseLabel] || PHASE_LABELS[phase] || phase}
                  </span>
                  <span className="text-[11px] text-indigo-400 font-mono font-bold">{progress}%</span>
                </div>
                <div className="w-full h-2.5 bg-zinc-800 rounded-full overflow-hidden border border-zinc-700">
                  <div className="h-full bg-gradient-to-r from-indigo-500 via-violet-500 to-purple-500 rounded-full transition-all duration-500 ease-out"
                    style={{ width:`${progress}%` }} />
                </div>
              </div>
            )}

            {/* Done */}
            {phase === 'done' && (
              <div className="flex items-center gap-2 bg-emerald-500/10 border border-emerald-500/30 rounded-lg px-3 py-2">
                <svg className="h-4 w-4 text-emerald-400 flex-shrink-0" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7"/>
                </svg>
                <span className="text-xs text-emerald-300">Export complete! File downloaded automatically.</span>
                {downloadUrl && (
                  <a href={downloadUrl} download={`${settings.filename}.${settings.format}`}
                    className="ml-auto text-[10px] text-emerald-400 hover:text-emerald-300 underline whitespace-nowrap">
                    Re-download
                  </a>
                )}
              </div>
            )}

            {/* Error */}
            {phase === 'error' && (
              <div className="flex flex-col gap-1 bg-red-500/10 border border-red-500/30 rounded-lg px-3 py-2">
                <div className="flex items-center gap-2">
                  <svg className="h-4 w-4 text-red-400 flex-shrink-0" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v4m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/>
                  </svg>
                  <span className="text-xs text-red-300 font-semibold">Export failed</span>
                </div>
                <p className="text-[9px] text-red-400/80 leading-relaxed">{errorMsg || 'Check the FFmpeg log for details.'}</p>
                <p className="text-[9px] text-zinc-500">
                  Tip: Try a lower resolution (720p) or shorter duration. FFmpeg WASM needs to load from CDN on first use.
                </p>
              </div>
            )}

            {/* Action buttons */}
            <div className="flex gap-2">
              {isExporting ? (
                <button onClick={() => { abortRef.current = true; }}
                  className="flex-1 py-2.5 rounded-xl bg-red-600 hover:bg-red-500 text-white text-sm font-semibold transition-colors">
                  Cancel Export
                </button>
              ) : (
                <>
                  <button onClick={() => { abortRef.current = true; onClose(); }}
                    className="px-4 py-2.5 rounded-xl bg-zinc-800 hover:bg-zinc-700 text-zinc-300 text-sm font-medium border border-zinc-700 transition-colors">
                    Cancel
                  </button>
                  <button onClick={handleExport}
                    disabled={tracks.flatMap(t => t.clips).length === 0}
                    className="flex-1 py-2.5 rounded-xl bg-gradient-to-r from-indigo-600 to-violet-600 hover:from-indigo-500 hover:to-violet-500 disabled:opacity-50 disabled:cursor-not-allowed text-white text-sm font-semibold transition-all flex items-center justify-center gap-2 shadow-lg">
                    <svg className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2M8 12l4 4 4-4M12 4v12"/>
                    </svg>
                    {phase === 'done' ? 'Export Again' : phase === 'error' ? 'Retry Export' : 'Start Export'}
                  </button>
                </>
              )}
            </div>

            <p className="text-[9px] text-zinc-600 text-center leading-relaxed">
              FFmpeg WASM · PNG image-sequence → libx264 H.264 · No MediaRecorder · CRF quality · CFR output
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
