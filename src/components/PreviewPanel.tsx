import { useEffect, useRef } from 'react';
import { TimelineClip, AspectRatio } from '../types/editor';
import { interpolateKF } from '../hooks/useEditorStore';

interface Props {
  currentTime: number;
  isPlaying: boolean;
  duration: number;
  activeClips: TimelineClip[];
  selectedClips: TimelineClip[];
  aspectRatio: AspectRatio;
  onPlay: () => void;
  onPause: () => void;
  onSeek: (t: number) => void;
  onAspectRatioChange: (r: AspectRatio) => void;
}

const ASPECT_RATIOS: { label: string; value: AspectRatio; w: number; h: number }[] = [
  { label: '16:9',  value: '16:9',  w: 16, h: 9  },
  { label: '9:16',  value: '9:16',  w: 9,  h: 16 },
  { label: '1:1',   value: '1:1',   w: 1,  h: 1  },
  { label: '4:3',   value: '4:3',   w: 4,  h: 3  },
  { label: '21:9',  value: '21:9',  w: 21, h: 9  },
];

function formatTime(sec: number) {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  const ms = Math.floor((sec % 1) * 10);
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}.${ms}`;
}

function computeClipTransform(clip: TimelineClip, currentTime: number) {
  const t = clip.transform;

  const scale = t.scaleKFs.length > 0
    ? interpolateKF(t.scaleKFs, t.scale, currentTime)
    : t.scale;

  const opacity = t.opacityKFs.length > 0
    ? interpolateKF(t.opacityKFs, t.opacity, currentTime)
    : t.opacity;

  const posX = t.posXKFs.length > 0
    ? interpolateKF(t.posXKFs, t.posX, currentTime)
    : t.posX;

  const posY = t.posYKFs.length > 0
    ? interpolateKF(t.posYKFs, t.posY, currentTime)
    : t.posY;

  return { scale, opacity, posX, posY };
}

// ── Per-clip video element — one ref per clip id ──────────────────────────────
function VideoLayer({
  clip,
  currentTime,
  isPlaying,
  hasAudio,
}: {
  clip: TimelineClip;
  currentTime: number;
  isPlaying: boolean;
  hasAudio: boolean;
}) {
  const ref = useRef<HTMLVideoElement>(null);
  const tf = computeClipTransform(clip, currentTime);

  useEffect(() => {
    const vid = ref.current;
    if (!vid) return;
    const offset = currentTime - clip.startTime + clip.trimStart;
    if (Math.abs(vid.currentTime - offset) > 0.3) {
      vid.currentTime = Math.max(0, offset);
    }
  }, [currentTime, clip]);

  useEffect(() => {
    const vid = ref.current;
    if (!vid) return;
    if (isPlaying) {
      vid.play().catch(() => {});
    } else {
      vid.pause();
    }
  }, [isPlaying]);

  return (
    <div
      style={{
        position: 'absolute',
        inset: 0,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        overflow: 'hidden',
        pointerEvents: 'none',
      }}
    >
      <video
        ref={ref}
        key={clip.url}
        src={clip.url}
        playsInline
        muted={hasAudio}
        style={{
          width: '100%',
          height: '100%',
          objectFit: 'contain',
          transform: `translate(${tf.posX}px, ${tf.posY}px) scale(${tf.scale})`,
          opacity: tf.opacity,
          transformOrigin: 'center center',
          transition: 'none',
        }}
      />
    </div>
  );
}

// ── Per-clip image layer ──────────────────────────────────────────────────────
function ImageLayer({ clip, currentTime }: { clip: TimelineClip; currentTime: number }) {
  const tf = computeClipTransform(clip, currentTime);
  return (
    <div
      style={{
        position: 'absolute',
        inset: 0,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        overflow: 'hidden',
        pointerEvents: 'none',
      }}
    >
      <img
        src={clip.url}
        alt=""
        style={{
          width: '100%',
          height: '100%',
          objectFit: 'contain',
          transform: `translate(${tf.posX}px, ${tf.posY}px) scale(${tf.scale})`,
          opacity: tf.opacity,
          transformOrigin: 'center center',
          transition: 'none',
        }}
      />
    </div>
  );
}

export default function PreviewPanel({
  currentTime,
  isPlaying,
  duration,
  activeClips,
  selectedClips,
  aspectRatio,
  onPlay,
  onPause,
  onSeek,
  onAspectRatioChange,
}: Props) {
  const audioRef = useRef<HTMLAudioElement>(null);

  const visualClips = [...activeClips]
    .filter(c => c.type !== 'audio')
    .sort((a, b) => b.trackIndex - a.trackIndex);

  const activeAudio = activeClips.find(c => c.type === 'audio');
  const hasAudioTrack = !!activeAudio;

  const progress = duration > 0 ? (currentTime / duration) * 100 : 0;

  const ratioObj = ASPECT_RATIOS.find(r => r.value === aspectRatio) ?? ASPECT_RATIOS[0];
  const ratioCss = `${ratioObj.w} / ${ratioObj.h}`;

  const debugClip = selectedClips.find(
    c => c.type === 'video' || c.type === 'image'
  );
  const debugT = debugClip ? computeClipTransform(debugClip, currentTime) : null;

  useEffect(() => {
    const aud = audioRef.current;
    if (!aud || !activeAudio) return;
    const offset = currentTime - activeAudio.startTime + activeAudio.trimStart;
    if (Math.abs(aud.currentTime - offset) > 0.3) {
      aud.currentTime = Math.max(0, offset);
    }
  }, [currentTime, activeAudio]);

  useEffect(() => {
    const aud = audioRef.current;
    if (!aud) return;
    if (isPlaying && activeAudio) {
      aud.play().catch(() => {});
    } else {
      aud.pause();
    }
  }, [isPlaying, activeAudio]);

  return (
    <div className="flex flex-col h-full bg-zinc-950 select-none">

      {/* ── Aspect Ratio Selector ─────────────────────────────────────────── */}
      <div className="flex items-center gap-1.5 px-3 py-1.5 bg-zinc-900 border-b border-zinc-800 flex-shrink-0">
        <span className="text-[10px] text-zinc-500 font-semibold uppercase tracking-wider mr-1">Ratio</span>
        {ASPECT_RATIOS.map(r => (
          <button
            key={r.value}
            onClick={() => onAspectRatioChange(r.value)}
            className={`text-[11px] font-medium px-2.5 py-0.5 rounded-md transition-colors border ${
              aspectRatio === r.value
                ? 'bg-indigo-600 border-indigo-500 text-white shadow'
                : 'bg-zinc-800 border-zinc-700 text-zinc-400 hover:text-zinc-200 hover:bg-zinc-700'
            }`}
          >
            {r.label}
          </button>
        ))}
        <span className="ml-auto text-[10px] text-zinc-600 font-mono">
          {ratioObj.w}:{ratioObj.h}
        </span>
      </div>

      {/* ── Preview Area ──────────────────────────────────────────────────── */}
      <div className="flex-1 flex items-center justify-center bg-zinc-950 relative overflow-hidden">
        <div
          className="relative bg-black shadow-2xl overflow-hidden"
          style={{
            aspectRatio: ratioCss,
            maxWidth: '100%',
            maxHeight: '100%',
            height: aspectRatio === '9:16' ? '90%' : undefined,
            width: aspectRatio === '9:16' ? 'auto' : '100%',
            outline: '2px solid rgba(99,102,241,0.25)',
          }}
        >
          {visualClips.map(clip => {
            if (clip.type === 'video') {
              return (
                <VideoLayer
                  key={clip.id}
                  clip={clip}
                  currentTime={currentTime}
                  isPlaying={isPlaying}
                  hasAudio={hasAudioTrack}
                />
              );
            }
            if (clip.type === 'image') {
              return (
                <ImageLayer
                  key={clip.id}
                  clip={clip}
                  currentTime={currentTime}
                />
              );
            }
            return null;
          })}

          {visualClips.length === 0 && activeAudio && (
            <AudioVisual isPlaying={isPlaying} color="#10b981" />
          )}

          {visualClips.length === 0 && !activeAudio && (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 text-zinc-700">
              <svg className="h-14 w-14" fill="none" stroke="currentColor" strokeWidth={1} viewBox="0 0 24 24">
                <rect x="2" y="4" width="20" height="16" rx="2" />
                <path strokeLinecap="round" strokeLinejoin="round" d="M10 9l5 3-5 3V9z" />
              </svg>
              <span className="text-xs text-center px-4">No media at current time</span>
            </div>
          )}

          {activeAudio && (
            <audio ref={audioRef} key={activeAudio.url} src={activeAudio.url} />
          )}

          <div className="absolute top-2 right-2 bg-black/70 rounded px-2 py-0.5 text-xs text-zinc-300 font-mono pointer-events-none z-10">
            {formatTime(currentTime)} / {formatTime(duration)}
          </div>

          {activeClips.length > 0 && (
            <div className="absolute bottom-2 left-2 flex gap-1 flex-wrap pointer-events-none z-10">
              {activeClips.map(c => (
                <span
                  key={c.id}
                  className="text-[10px] px-1.5 py-0.5 rounded-full text-white font-medium"
                  style={{ backgroundColor: c.color + 'cc' }}
                >
                  {c.name}
                </span>
              ))}
            </div>
          )}

          {debugT && debugClip && (
            <div className="absolute top-2 left-2 bg-black/75 rounded px-2 py-1 text-[9px] text-indigo-300 font-mono pointer-events-none z-10 flex flex-col gap-0.5">
              <span>Scale: {(debugT.scale * 100).toFixed(0)}%</span>
              <span>Opacity: {(debugT.opacity * 100).toFixed(0)}%</span>
              <span>X: {debugT.posX.toFixed(0)}px  Y: {debugT.posY.toFixed(0)}px</span>
            </div>
          )}
        </div>
      </div>

      {/* ── Controls ─────────────────────────────────────────────────────── */}
      <div className="bg-zinc-900 border-t border-zinc-800 px-4 py-2 flex flex-col gap-2 flex-shrink-0">
        <div
          className="w-full h-1.5 bg-zinc-700 rounded-full cursor-pointer relative group"
          onClick={e => {
            const rect = e.currentTarget.getBoundingClientRect();
            const ratio = (e.clientX - rect.left) / rect.width;
            onSeek(ratio * duration);
          }}
        >
          <div
            className="h-full bg-indigo-500 rounded-full"
            style={{ width: `${progress}%` }}
          />
          <div
            className="absolute top-1/2 -translate-y-1/2 w-3 h-3 bg-white rounded-full shadow -translate-x-1/2 opacity-0 group-hover:opacity-100 transition-opacity"
            style={{ left: `${progress}%` }}
          />
        </div>

        <div className="flex items-center gap-3">
          <div className="flex items-center justify-center gap-3 flex-1">
            <button onClick={() => onSeek(0)} className="text-zinc-400 hover:text-white transition-colors" title="Rewind">
              <svg className="h-4 w-4" fill="currentColor" viewBox="0 0 24 24">
                <path d="M6 6h2v12H6zm3.5 6 8.5 6V6z" />
              </svg>
            </button>

            <button onClick={() => onSeek(currentTime - 5)} className="text-zinc-400 hover:text-white transition-colors" title="-5s">
              <svg className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 8V4l-8 8 8 8v-4c3.314 0 6 2.686 6 6a6 6 0 01-6-6z" />
              </svg>
            </button>

            <button
              onClick={isPlaying ? onPause : onPlay}
              className="w-9 h-9 rounded-full bg-indigo-600 hover:bg-indigo-500 flex items-center justify-center text-white transition-colors shadow"
              title={isPlaying ? 'Pause (Space)' : 'Play (Space)'}
            >
              {isPlaying ? (
                <svg className="h-4 w-4" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M6 19h4V5H6zm8-14v14h4V5z" />
                </svg>
              ) : (
                <svg className="h-4 w-4 ml-0.5" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M8 5v14l11-7z" />
                </svg>
              )}
            </button>

            <button onClick={() => onSeek(currentTime + 5)} className="text-zinc-400 hover:text-white transition-colors" title="+5s">
              <svg className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 8V4l8 8-8 8v-4a6 6 0 00-6 6 6 6 0 016-6z" />
              </svg>
            </button>

            <button onClick={() => onSeek(duration)} className="text-zinc-400 hover:text-white transition-colors" title="End">
              <svg className="h-4 w-4" fill="currentColor" viewBox="0 0 24 24">
                <path d="M6 18l8.5-6L6 6v12zm2.5-6 5.5 3.9V8.1z" />
                <path d="M16 6h2v12h-2z" />
              </svg>
            </button>
          </div>

          <span className="text-[9px] text-zinc-600 font-mono">SPACE</span>
        </div>
      </div>
    </div>
  );
}

// ── Audio-only visual ─────────────────────────────────────────────────────────
function AudioVisual({ isPlaying, color }: { isPlaying: boolean; color: string }) {
  const bars = [0.4, 0.7, 1, 0.6, 0.9, 0.5, 0.8, 0.3, 0.75, 0.55, 0.95, 0.4];
  return (
    <div className="absolute inset-0 flex flex-col items-center justify-center gap-4 bg-zinc-900">
      <svg className="h-10 w-10 text-emerald-500 opacity-60" fill="currentColor" viewBox="0 0 24 24">
        <path d="M12 3v10.55A4 4 0 1014 17V7h4V3h-6z" />
      </svg>
      <div className="flex items-end gap-[3px] h-12">
        {bars.map((amp, i) => (
          <div
            key={i}
            className="w-[5px] rounded-full"
            style={{
              backgroundColor: color,
              height: isPlaying ? `${amp * 100}%` : '20%',
              opacity: 0.8,
              transition: 'height 0.15s ease',
              animation: isPlaying ? `audioBar 0.${6 + (i % 5)}s ease-in-out infinite alternate` : 'none',
              animationDelay: `${i * 0.07}s`,
            }}
          />
        ))}
      </div>
      <span className="text-xs text-zinc-500">Audio Only</span>
      <style>{`
        @keyframes audioBar {
          from { transform: scaleY(0.3); }
          to   { transform: scaleY(1);   }
        }
      `}</style>
    </div>
  );
}
