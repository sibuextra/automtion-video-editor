import { useRef, useCallback, useState, useEffect } from 'react';
import { Track, TimelineClip, MediaFile } from '../types/editor';
import { isMediaAllowedOnTrack } from '../hooks/useEditorStore';

interface Props {
  tracks: Track[];
  currentTime: number;
  duration: number;
  zoom: number;
  selectedClipIds: string[];
  mediaFiles: MediaFile[];
  isPlaying: boolean;
  snapEnabled: boolean;
  onSnapToggle: () => void;
  onSeek: (t: number) => void;
  onSelectClip: (id: string | null, addToSelection?: boolean) => void;
  onMoveClip: (clipId: string, newStart: number, newTrack: number) => void;
  onRemoveClip: (clipId: string) => void;
  onSplitAtPlayhead: (splitTime: number) => void;
  onAddClipToTrack: (media: MediaFile, trackIndex: number, startTime: number) => void;
  onZoomChange: (zoom: number) => void;
  onAddTrack: (type: 'video' | 'audio') => void;
  onRemoveTrack: (trackId: string) => void;
  onMoveKeyframe: (clipId: string, prop: 'scale' | 'opacity' | 'posX' | 'posY', kfId: string, newTime: number) => void;
  onPlay: () => void;
  onPause: () => void;
}

const TRACK_HEIGHT = 52;
const HEADER_WIDTH = 90;
const RULER_HEIGHT = 28;
const SNAP_THRESHOLD_PX = 10;

function formatRulerTime(sec: number) {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return m > 0 ? `${m}:${String(s).padStart(2, '0')}` : `${s}s`;
}

export default function Timeline({
  tracks, currentTime, duration, zoom, selectedClipIds, mediaFiles,
  isPlaying, snapEnabled, onSnapToggle, onSeek, onSelectClip,
  onMoveClip, onRemoveClip, onSplitAtPlayhead, onAddClipToTrack, onZoomChange,
  onAddTrack, onRemoveTrack, onMoveKeyframe, onPlay, onPause,
}: Props) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [isDraggingPlayhead, setIsDraggingPlayhead] = useState(false);
  const [snapIndicatorX, setSnapIndicatorX] = useState<number | null>(null);

  const totalWidth = Math.max(duration * zoom, 600);
  const tickInterval = zoom >= 100 ? 1 : zoom >= 50 ? 2 : zoom >= 20 ? 5 : 10;
  const ticks: number[] = [];
  for (let t = 0; t <= duration + tickInterval; t += tickInterval) ticks.push(t);

  const videoTracks = tracks.filter(t => t.type === 'video');
  const audioTracks = tracks.filter(t => t.type === 'audio');
  const lastVideoId = videoTracks[videoTracks.length - 1]?.id ?? null;
  const lastAudioId = audioTracks[audioTracks.length - 1]?.id ?? null;
  const extraVideoIds = new Set(videoTracks.slice(2).map(t => t.id));
  const extraAudioIds = new Set(audioTracks.slice(2).map(t => t.id));

  // ── Spacebar play/pause ───────────────────────────────────────────────────
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement).tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
      if (e.code === 'Space') { e.preventDefault(); if (isPlaying) onPause(); else onPlay(); }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [isPlaying, onPlay, onPause]);

  // ── Delete key → selected clips remove ───────────────────────────────────
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement).tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
      if (e.key === 'Delete' && selectedClipIds.length > 0) {
        e.preventDefault();
        selectedClipIds.forEach(id => onRemoveClip(id));
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [selectedClipIds, onRemoveClip]);

  // ── Ctrl+B → Split clip at playhead ──────────────────────────────────────
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement).tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
      if ((e.ctrlKey || e.metaKey) && e.key === 'b') {
        e.preventDefault();
        onSplitAtPlayhead(currentTime);
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [currentTime, onSplitAtPlayhead]);

  const getSnapPoints = useCallback((excludeClipIds: string[] = []): number[] => {
    const points: number[] = [0];
    for (const track of tracks) {
      for (const clip of track.clips) {
        if (excludeClipIds.includes(clip.id)) continue;
        points.push(clip.startTime);
        points.push(clip.startTime + clip.duration);
      }
    }
    return points;
  }, [tracks]);

  const snapTime = useCallback((rawTime: number, excludeClipIds: string[] = []): { time: number; snapped: boolean; snapX: number | null } => {
    if (!snapEnabled) return { time: rawTime, snapped: false, snapX: null };
    const points = getSnapPoints(excludeClipIds);
    const thresholdSec = SNAP_THRESHOLD_PX / zoom;
    let closest = rawTime, minDist = thresholdSec, didSnap = false;
    for (const pt of points) {
      const dist = Math.abs(rawTime - pt);
      if (dist < minDist) { minDist = dist; closest = pt; didSnap = true; }
    }
    return { time: closest, snapped: didSnap, snapX: didSnap ? closest * zoom : null };
  }, [snapEnabled, getSnapPoints, zoom]);

  const snapClip = useCallback((rawStart: number, clipDuration: number, excludeClipIds: string[] = []): { start: number; snapX: number | null } => {
    if (!snapEnabled) return { start: rawStart, snapX: null };
    const points = getSnapPoints(excludeClipIds);
    const thresholdSec = SNAP_THRESHOLD_PX / zoom;
    let bestStart = rawStart, minDist = thresholdSec, snapX: number | null = null;
    for (const pt of points) {
      const dist = Math.abs(rawStart - pt);
      if (dist < minDist) { minDist = dist; bestStart = pt; snapX = pt * zoom; }
    }
    const rawEnd = rawStart + clipDuration;
    for (const pt of points) {
      const dist = Math.abs(rawEnd - pt);
      if (dist < minDist) { minDist = dist; bestStart = pt - clipDuration; snapX = pt * zoom; }
    }
    return { start: Math.max(0, bestStart), snapX };
  }, [snapEnabled, getSnapPoints, zoom]);

  const clientXToTime = useCallback((clientX: number): number => {
    const el = scrollRef.current;
    if (!el) return 0;
    const rect = el.getBoundingClientRect();
    const x = clientX - rect.left - HEADER_WIDTH + el.scrollLeft;
    return Math.max(0, Math.min(x / zoom, duration));
  }, [zoom, duration]);

  const handleRulerMouseDown = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    if ((e.target as HTMLElement).closest('[data-ruler-header]')) return;
    e.preventDefault();
    setIsDraggingPlayhead(true);
    const rawTime = clientXToTime(e.clientX);
    const { time, snapX } = snapTime(rawTime);
    onSeek(time);
    setSnapIndicatorX(snapX);
  }, [clientXToTime, onSeek, snapTime]);

  const handlePlayheadMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault(); e.stopPropagation(); setIsDraggingPlayhead(true);
  }, []);

  useEffect(() => {
    if (!isDraggingPlayhead) { setSnapIndicatorX(null); return; }
    const onMove = (e: MouseEvent) => {
      const rawTime = clientXToTime(e.clientX);
      const { time, snapX } = snapTime(rawTime);
      onSeek(time); setSnapIndicatorX(snapX);
    };
    const onUp = () => { setIsDraggingPlayhead(false); setSnapIndicatorX(null); };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    return () => { window.removeEventListener('mousemove', onMove); window.removeEventListener('mouseup', onUp); };
  }, [isDraggingPlayhead, clientXToTime, onSeek, snapTime]);

  const handleClipMouseDown = useCallback((e: React.MouseEvent, clip: TimelineClip, isCtrl: boolean) => {
    e.stopPropagation();
    const alreadySelected = selectedClipIds.includes(clip.id);
    if (isCtrl) { onSelectClip(clip.id, true); }
    else if (!alreadySelected) { onSelectClip(clip.id, false); }

    const startX = e.clientX, startY = e.clientY;
    let didMove = false;

    const currentlySelected = isCtrl
      ? selectedClipIds.includes(clip.id) ? selectedClipIds : [...selectedClipIds, clip.id]
      : alreadySelected ? selectedClipIds : [clip.id];

    const allClips = tracks.flatMap(t => t.clips);
    const origPositions = new Map<string, { startTime: number; trackIndex: number }>();
    for (const id of currentlySelected) {
      const c = allClips.find(cl => cl.id === id);
      if (c) origPositions.set(id, { startTime: c.startTime, trackIndex: c.trackIndex });
    }
    const primaryOrig = origPositions.get(clip.id) ?? { startTime: clip.startTime, trackIndex: clip.trackIndex };

    const onMouseMove = (ev: MouseEvent) => {
      const dx = ev.clientX - startX, dy = ev.clientY - startY;
      if (!didMove && (Math.abs(dx) > 3 || Math.abs(dy) > 3)) didMove = true;
      if (!didMove) return;
      const dt = dx / zoom;
      const el = scrollRef.current;
      if (!el) return;
      const rect = el.getBoundingClientRect();
      const y = ev.clientY - rect.top + el.scrollTop - RULER_HEIGHT;
      const rawTrackIdx = Math.floor(y / TRACK_HEIGHT);
      const trackDelta = rawTrackIdx - primaryOrig.trackIndex;

      for (const [id, orig] of origPositions) {
        const rawStart = Math.max(0, orig.startTime + dt);
        const newTrackIdx = Math.max(0, Math.min(tracks.length - 1, orig.trackIndex + trackDelta));
        const c = allClips.find(cl => cl.id === id);
        if (!c) continue;
        if (!isMediaAllowedOnTrack(c.type, tracks[newTrackIdx].type)) continue;
        if (id === clip.id) {
          const { start: snappedStart, snapX } = snapClip(rawStart, c.duration, currentlySelected);
          onMoveClip(id, snappedStart, newTrackIdx);
          setSnapIndicatorX(snapX);
        } else {
          const primaryRawStart = Math.max(0, primaryOrig.startTime + dt);
          const primarySnapped = snapClip(primaryRawStart, allClips.find(cl => cl.id === clip.id)?.duration ?? 1, currentlySelected);
          const snapDelta = primarySnapped.start - primaryRawStart;
          onMoveClip(id, Math.max(0, rawStart + snapDelta), newTrackIdx);
        }
      }
    };

    const onMouseUp = () => {
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', onMouseUp);
      setSnapIndicatorX(null);
    };
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);
  }, [zoom, tracks, onMoveClip, onSelectClip, selectedClipIds, snapClip]);

  const handleDrop = useCallback((e: React.DragEvent, trackIndex: number) => {
    e.preventDefault();
    const mediaId = e.dataTransfer.getData('mediaId');
    if (!mediaId) return;
    const media = mediaFiles.find(m => m.id === mediaId);
    if (!media) return;
    if (!isMediaAllowedOnTrack(media.type, tracks[trackIndex].type)) return;
    const el = scrollRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const x = e.clientX - rect.left - HEADER_WIDTH + el.scrollLeft;
    const rawStart = Math.max(0, x / zoom);
    const clipDur = media.duration ?? 5;
    const { start: snappedStart } = snapClip(rawStart, clipDur);
    onAddClipToTrack(media, trackIndex, snappedStart);
  }, [mediaFiles, zoom, tracks, onAddClipToTrack, snapClip]);

  const handleDragOver = useCallback((e: React.DragEvent, trackIndex: number) => {
    const mediaId = e.dataTransfer.getData('mediaId');
    if (mediaId) {
      const media = mediaFiles.find(m => m.id === mediaId);
      if (media && !isMediaAllowedOnTrack(media.type, tracks[trackIndex].type)) return;
    }
    e.preventDefault();
  }, [mediaFiles, tracks]);

  const playheadX = currentTime * zoom;

  return (
    <div className="flex flex-col h-full bg-zinc-950 select-none">
      {/* Toolbar */}
      <div className="flex items-center gap-3 px-3 py-1.5 border-b border-zinc-800 bg-zinc-900 flex-shrink-0">
        <span className="text-xs text-zinc-500 font-medium">Timeline</span>
        {selectedClipIds.length > 0 && (
          <span className="text-[10px] bg-indigo-600/30 text-indigo-300 px-2 py-0.5 rounded-full border border-indigo-500/30">
            {selectedClipIds.length} selected
          </span>
        )}
        {selectedClipIds.length > 0 && <span className="text-[9px] text-zinc-600">Del to remove</span>}

        {/* Ctrl+B split button */}
        <button
          onClick={() => onSplitAtPlayhead(currentTime)}
          title="Split clip at playhead (Ctrl+B)"
          className="flex items-center gap-1.5 text-[10px] font-semibold px-2 py-0.5 rounded border transition-colors bg-zinc-800 border-zinc-700 text-zinc-500 hover:bg-amber-600/30 hover:border-amber-500/60 hover:text-amber-300"
        >
          <svg className="h-3 w-3" fill="none" stroke="currentColor" strokeWidth={2.5} viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 3v18M3 12h18" />
          </svg>
          Split
          <span className="text-[9px] opacity-60">Ctrl+B</span>
        </button>

        <button onClick={onSnapToggle}
          className={`flex items-center gap-1.5 text-[10px] font-semibold px-2 py-0.5 rounded border transition-colors ${
            snapEnabled ? 'bg-indigo-600/30 border-indigo-500/60 text-indigo-300' : 'bg-zinc-800 border-zinc-700 text-zinc-500 hover:bg-zinc-700 hover:text-zinc-300'
          }`}>
          <svg className="h-3 w-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2.5} strokeLinecap="round" strokeLinejoin="round">
            <path d="M6 15A6 6 0 0 0 18 15" /><path d="M6 15V5" /><path d="M18 15V5" /><path d="M3 5h6" /><path d="M15 5h6" />
          </svg>
          Snap {snapEnabled ? 'ON' : 'OFF'}
        </button>

        <div className="flex items-center gap-2 ml-auto">
          <span className="text-[11px] text-zinc-500">Zoom</span>
          <input type="range" min={10} max={200} value={zoom} onChange={e => onZoomChange(Number(e.target.value))}
            className="w-24 accent-indigo-500 h-1 cursor-pointer" />
          <span className="text-[11px] text-zinc-400 w-10">{zoom}px/s</span>
        </div>
      </div>

      {/* Scrollable body — click on empty area to deselect */}
      <div ref={scrollRef} className="flex-1 overflow-auto relative"
        style={{ cursor: isDraggingPlayhead ? 'col-resize' : 'default' }}
        onClick={e => {
          // Sirf tab deselect karo jab clip pe click na ho
          // Clip block ka onClick e.stopPropagation() call karta hai
          // Toh agar event yahan tak aaya matlab empty area pe click hua
          const target = e.target as HTMLElement;
          const isClipArea = target.closest('[data-clip-block]');
          if (!isClipArea) onSelectClip(null);
        }}>
        <div style={{ width: totalWidth + HEADER_WIDTH, minHeight: '100%', position: 'relative' }}>

          {/* Ruler */}
          <div className="flex sticky top-0 z-20 bg-zinc-900 border-b border-zinc-800 cursor-crosshair"
            style={{ height: RULER_HEIGHT }} onMouseDown={handleRulerMouseDown}>
            <div data-ruler-header="true" style={{ width: HEADER_WIDTH }}
              className="flex-shrink-0 border-r border-zinc-800 cursor-default" />
            <div className="relative flex-1" style={{ height: RULER_HEIGHT }}>
              {ticks.map(t => (
                <div key={t} className="absolute top-0 flex flex-col items-center pointer-events-none" style={{ left: t * zoom }}>
                  <div className="w-px h-2 bg-zinc-600 mt-1" />
                  <span className="text-[9px] text-zinc-500 mt-0.5 whitespace-nowrap select-none">{formatRulerTime(t)}</span>
                </div>
              ))}
              {snapEnabled && snapIndicatorX !== null && (
                <div className="absolute top-0 bottom-0 w-px bg-yellow-400/70 pointer-events-none z-40" style={{ left: snapIndicatorX }} />
              )}
              <div className="absolute top-0 bottom-0 z-30 flex flex-col items-center"
                style={{ left: playheadX, transform: 'translateX(-50%)' }}>
                <div className="w-3 h-3 cursor-col-resize flex-shrink-0" style={{ marginTop: 2 }}
                  onMouseDown={handlePlayheadMouseDown}>
                  <svg viewBox="0 0 12 10" className="w-full h-full fill-indigo-400"><polygon points="6,10 0,0 12,0" /></svg>
                </div>
                <div className="w-px flex-1 bg-indigo-400 pointer-events-none" />
              </div>
            </div>
          </div>

          {/* Tracks */}
          {tracks.map((track, trackIndex) => {
            const isLastVideo = track.id === lastVideoId;
            const isLastAudio = track.id === lastAudioId;
            const isExtraVideo = extraVideoIds.has(track.id);
            const isExtraAudio = extraAudioIds.has(track.id);
            const showPlus = isLastVideo || isLastAudio;
            const showMinus = isExtraVideo || isExtraAudio;

            return (
              <div key={track.id} className="flex border-b border-zinc-800/60" style={{ height: TRACK_HEIGHT }}
                onDragOver={e => handleDragOver(e, trackIndex)} onDrop={e => handleDrop(e, trackIndex)}>

                <div className="flex-shrink-0 flex items-center px-1.5 gap-1 border-r border-zinc-800 bg-zinc-900/80 sticky left-0 z-10"
                  style={{ width: HEADER_WIDTH }}>
                  <div className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${track.type === 'audio' ? 'bg-emerald-500' : 'bg-indigo-500'}`} />
                  <span className="text-[10px] text-zinc-400 truncate font-medium flex-1">{track.label}</span>
                  {showPlus && (
                    <button className={`w-4 h-4 flex items-center justify-center rounded-sm transition-colors flex-shrink-0 ${
                      track.type === 'audio' ? 'bg-emerald-700/50 hover:bg-emerald-600 text-emerald-300' : 'bg-indigo-700/50 hover:bg-indigo-600 text-indigo-300'
                    }`} onClick={e => { e.stopPropagation(); onAddTrack(track.type as 'video' | 'audio'); }}>
                      <svg className="h-2.5 w-2.5" fill="none" stroke="currentColor" strokeWidth={2.5} viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" d="M12 4v16m8-8H4" />
                      </svg>
                    </button>
                  )}
                  {showMinus && (
                    <button className="w-4 h-4 flex items-center justify-center rounded-sm bg-red-700/50 hover:bg-red-600 text-red-300 flex-shrink-0 transition-colors"
                      onClick={e => { e.stopPropagation(); onRemoveTrack(track.id); }}>
                      <svg className="h-2.5 w-2.5" fill="none" stroke="currentColor" strokeWidth={2.5} viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" d="M20 12H4" />
                      </svg>
                    </button>
                  )}
                </div>

                <div className="relative flex-1 bg-zinc-950/50" style={{ minWidth: totalWidth }}>
                  {ticks.map(t => (
                    <div key={t} className="absolute top-0 bottom-0 w-px bg-zinc-800/40 pointer-events-none" style={{ left: t * zoom }} />
                  ))}
                  {snapEnabled && snapIndicatorX !== null && (
                    <div className="absolute top-0 bottom-0 w-px bg-yellow-400/50 pointer-events-none z-30" style={{ left: snapIndicatorX }} />
                  )}
                  {track.clips.map(clip => (
                    <ClipBlock key={clip.id} clip={clip} zoom={zoom}
                      selected={selectedClipIds.includes(clip.id)} trackHeight={TRACK_HEIGHT}
                      currentTime={currentTime} onSeek={onSeek}
                      onMouseDown={e => handleClipMouseDown(e, clip, e.ctrlKey || e.metaKey)}
                      onRemove={() => onRemoveClip(clip.id)}
                      onMoveKeyframe={(prop, kfId, newTime) => onMoveKeyframe(clip.id, prop, kfId, newTime)} />
                  ))}
                  <div className="absolute top-0 bottom-0 w-px bg-indigo-400/60 z-10 pointer-events-none" style={{ left: playheadX }} />
                </div>
              </div>
            );
          })}

          {tracks.every(t => t.clips.length === 0) && (
            <div className="absolute inset-0 flex items-center justify-center pointer-events-none mt-6">
              <p className="text-xs text-zinc-700">
                Drag media from panel • Ctrl+Click to multi-select • Space to play/pause • Del to remove • <span className="text-amber-600">Ctrl+B to split</span>
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function WaveformBar({ waveformData, width, height, color }: { waveformData: number[]; width: number; height: number; color: string }) {
  if (!waveformData.length || width < 4) return null;
  const barW = 2, gap = 1;
  const totalBars = Math.floor(width / (barW + gap));
  const samples = Math.max(totalBars, 1);
  const step = waveformData.length / samples;
  const bars: number[] = [];
  for (let i = 0; i < samples; i++) {
    const idx = Math.floor(i * step);
    bars.push(waveformData[Math.min(idx, waveformData.length - 1)]);
  }
  const midY = height / 2, maxAmp = midY - 2;
  return (
    <svg className="absolute inset-0 w-full h-full pointer-events-none" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
      {bars.map((amp, i) => {
        const x = i * (barW + gap), barH = Math.max(amp * maxAmp, 1);
        return (
          <g key={i}>
            <rect x={x} y={midY - barH} width={barW} height={barH} fill={color} opacity={0.75} rx={0.5} />
            <rect x={x} y={midY} width={barW} height={barH} fill={color} opacity={0.75} rx={0.5} />
          </g>
        );
      })}
      <line x1={0} y1={midY} x2={width} y2={midY} stroke={color} strokeWidth={0.5} opacity={0.3} />
    </svg>
  );
}

type KFProp = 'scale' | 'opacity' | 'posX' | 'posY';

interface ClipBlockProps {
  clip: TimelineClip;
  zoom: number;
  selected: boolean;
  trackHeight: number;
  currentTime: number;
  onMouseDown: (e: React.MouseEvent) => void;
  onRemove: () => void;
  onSeek: (t: number) => void;
  onMoveKeyframe: (prop: KFProp, kfId: string, newTime: number) => void;
}

function ClipBlock({ clip, zoom, selected, onMouseDown, onRemove, onSeek, onMoveKeyframe, currentTime }: ClipBlockProps) {
  const left = clip.startTime * zoom;
  const width = Math.max(clip.duration * zoom, 4);
  const isAudio = clip.type === 'audio';
  const t = clip.transform;

  const allKFProps: { prop: KFProp; color: string }[] = [
    { prop: 'scale', color: '#a78bfa' }, { prop: 'opacity', color: '#34d399' },
    { prop: 'posX', color: '#fb923c' }, { prop: 'posY', color: '#f472b6' },
  ];

  const kfDots: { id: string; time: number; prop: KFProp; color: string }[] = [];
  if (!isAudio) {
    for (const { prop, color } of allKFProps) {
      const kfProp = `${prop}KFs` as keyof typeof t;
      const kfs = t[kfProp] as Array<{ id: string; time: number; value: number }>;
      for (const kf of kfs) kfDots.push({ id: kf.id, time: kf.time, prop, color });
    }
  }

  const speed = clip.speed ?? 1;
  const showSpeedBadge = Math.abs(speed - 1) > 0.01;

  const handleKFDotMouseDown = (e: React.MouseEvent, kf: { id: string; time: number; prop: KFProp }) => {
    e.stopPropagation(); e.preventDefault();
    const startX = e.clientX, origTime = kf.time;
    const onMove = (ev: MouseEvent) => {
      const dt = (ev.clientX - startX) / zoom;
      const newTime = Math.max(clip.startTime, Math.min(clip.startTime + clip.duration, origTime + dt));
      onMoveKeyframe(kf.prop, kf.id, newTime);
      onSeek(newTime);
    };
    const onUp = () => { window.removeEventListener('mousemove', onMove); window.removeEventListener('mouseup', onUp); };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
  };

  const playheadRelX = (currentTime - clip.startTime) * zoom;

  return (
    <div
      data-clip-block="true"
      className="absolute top-1 bottom-1 rounded-md cursor-grab active:cursor-grabbing flex items-center overflow-hidden group"
      style={{
        left, width,
        backgroundColor: clip.color + (isAudio ? '22' : '33'),
        border: `1.5px solid ${clip.color}${selected ? 'ff' : '88'}`,
        boxShadow: selected ? `0 0 0 1px ${clip.color}` : undefined,
      }}
      onMouseDown={onMouseDown}
      onClick={e => e.stopPropagation()}>

      {clip.thumbnail && !isAudio && width > 40 && (
        <img src={clip.thumbnail} alt="" className="absolute inset-0 w-full h-full object-cover opacity-30 pointer-events-none" />
      )}

      {isAudio && clip.waveformData && clip.waveformData.length > 0 && (
        <WaveformBar waveformData={clip.waveformData} width={width} height={40} color={clip.color} />
      )}

      {width > 30 && (
        <div className="relative z-10 flex items-center w-full px-1.5 gap-1">
          <span className="text-[10px] font-semibold truncate drop-shadow"
            style={{ color: isAudio ? '#d1fae5' : 'white', maxWidth: width - 36, textShadow: '0 1px 3px rgba(0,0,0,0.9)' }}>
            {clip.name}
          </span>
          {showSpeedBadge && width > 60 && (
            <span className="text-[8px] font-bold px-1 py-0 rounded flex-shrink-0"
              style={{ backgroundColor: '#f59e0b44', color: '#fbbf24', border: '1px solid #f59e0b66' }}>
              {speed < 1 ? `${Math.round(speed * 100)}%` : `${speed.toFixed(1)}×`}
            </span>
          )}
        </div>
      )}

      {(!isAudio || !clip.waveformData) && (
        <div className="absolute right-1.5 top-1/2 -translate-y-1/2 w-1.5 h-1.5 rounded-full" style={{ backgroundColor: clip.color }} />
      )}

      {kfDots.map(kf => {
        const relX = (kf.time - clip.startTime) * zoom;
        if (relX < 0 || relX > width) return null;
        const isAtPlayhead = Math.abs(kf.time - currentTime) < 0.06;
        return (
          <div key={kf.id} className="absolute z-20 cursor-col-resize" style={{ left: relX - 5, bottom: 2 }}
            onMouseDown={e => handleKFDotMouseDown(e, kf)} onClick={e => { e.stopPropagation(); onSeek(kf.time); }}>
            <div className="w-2.5 h-2.5 rounded-full border-2"
              style={{
                backgroundColor: isAtPlayhead ? '#facc15' : kf.color,
                borderColor: isAtPlayhead ? '#fef08a' : 'rgba(0,0,0,0.6)',
                boxShadow: isAtPlayhead ? '0 0 4px #facc15' : `0 0 3px ${kf.color}88`,
              }} />
          </div>
        );
      })}

      {selected && playheadRelX >= 0 && playheadRelX <= width && (
        <div className="absolute top-0 bottom-0 w-px bg-white/40 z-10 pointer-events-none" style={{ left: playheadRelX }} />
      )}

      <button className="absolute top-0.5 right-0.5 opacity-0 group-hover:opacity-100 transition-opacity w-4 h-4 flex items-center justify-center rounded-sm bg-red-600/80 hover:bg-red-500 text-white z-20"
        onMouseDown={e => e.stopPropagation()} onClick={e => { e.stopPropagation(); onRemove(); }}>
        <svg className="h-2.5 w-2.5" fill="none" stroke="currentColor" strokeWidth={2.5} viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
        </svg>
      </button>
    </div>
  );
}
