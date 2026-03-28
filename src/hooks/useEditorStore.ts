import { useState, useCallback, useRef } from 'react';
import {
  MediaFile, TimelineClip, Track, AspectRatio,
  ClipTransform, Keyframe, defaultTransform,
  AutomationSettings, defaultAutomationSettings,
} from '../types/editor';

const CLIP_COLORS: Record<string, string> = {
  video: '#6366f1',
  audio: '#10b981',
  image: '#f59e0b',
};

function generateId() {
  return Math.random().toString(36).slice(2, 10);
}

function initTracks(): Track[] {
  return [
    { id: generateId(), label: 'Video 1', type: 'video', clips: [] },
    { id: generateId(), label: 'Video 2', type: 'video', clips: [] },
    { id: generateId(), label: 'Audio 1', type: 'audio', clips: [] },
    { id: generateId(), label: 'Audio 2', type: 'audio', clips: [] },
  ];
}

/** Returns true if this media type is allowed on the given track */
export function isMediaAllowedOnTrack(
  mediaType: MediaFile['type'],
  trackType: Track['type']
): boolean {
  if (trackType === 'mixed') return true;
  if (trackType === 'audio') return mediaType === 'audio';
  return mediaType === 'video' || mediaType === 'image';
}

/**
 * Given existing clips in a track and a desired startTime + duration,
 * returns the first non-overlapping startTime >= desired start.
 */
function resolveNonOverlappingStart(
  clips: TimelineClip[],
  desiredStart: number,
  duration: number
): number {
  const sorted = [...clips].sort((a, b) => a.startTime - b.startTime);
  let st = Math.max(0, desiredStart);
  let changed = true;
  while (changed) {
    changed = false;
    for (const c of sorted) {
      const cEnd = c.startTime + c.duration;
      const newEnd = st + duration;
      if (!(newEnd <= c.startTime || st >= cEnd)) {
        st = cEnd;
        changed = true;
        break;
      }
    }
  }
  return st;
}

/** Interpolate keyframes at a given time (linear) */
export function interpolateKF(kfs: Keyframe[], fallback: number, time: number): number {
  if (kfs.length === 0) return fallback;
  const sorted = [...kfs].sort((a, b) => a.time - b.time);
  if (time <= sorted[0].time) return sorted[0].value;
  if (time >= sorted[sorted.length - 1].time) return sorted[sorted.length - 1].value;
  for (let i = 0; i < sorted.length - 1; i++) {
    const a = sorted[i];
    const b = sorted[i + 1];
    if (time >= a.time && time <= b.time) {
      const t = (time - a.time) / (b.time - a.time);
      return a.value + (b.value - a.value) * t;
    }
  }
  return fallback;
}

// ── History state snapshot ────────────────────────────────────────────────────
interface HistorySnapshot {
  tracks: Track[];
  mediaFiles: MediaFile[];
  duration: number;
}

const MAX_HISTORY = 50;

export function useEditorStore() {
  const [mediaFiles, setMediaFiles] = useState<MediaFile[]>([]);
  const [tracks, setTracks] = useState<Track[]>(initTracks);
  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [duration, setDuration] = useState(30);
  const [selectedClipIds, setSelectedClipIds] = useState<string[]>([]);
  const [zoom, setZoom] = useState(50);
  const [aspectRatio, setAspectRatio] = useState<AspectRatio>('16:9');

  // ── Automation Settings State ─────────────────────────────────────────────
  const [automationSettings, setAutomationSettings] = useState<AutomationSettings>(
    defaultAutomationSettings
  );

  const playIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // ── Undo/Redo history stacks ──────────────────────────────────────────────
  const historyRef = useRef<HistorySnapshot[]>([]);
  const futureRef = useRef<HistorySnapshot[]>([]);

  // Internal refs to always have current values for history snapshot
  const tracksRef = useRef(tracks);
  const mediaFilesRef = useRef(mediaFiles);
  const durationRef = useRef(duration);

  // Keep refs in sync
  tracksRef.current = tracks;
  mediaFilesRef.current = mediaFiles;
  durationRef.current = duration;

  // Automation ref — always latest
  const automationRef = useRef(automationSettings);
  automationRef.current = automationSettings;

  // selectedClipIds ref — split function ke liye latest value
  const selectedClipIdsRef = useRef(selectedClipIds);
  selectedClipIdsRef.current = selectedClipIds;

  /** Push current state onto history stack before a change */
  const pushHistory = useCallback(() => {
    const snapshot: HistorySnapshot = {
      tracks: JSON.parse(JSON.stringify(tracksRef.current)),
      mediaFiles: JSON.parse(JSON.stringify(mediaFilesRef.current)),
      duration: durationRef.current,
    };
    historyRef.current = [...historyRef.current, snapshot].slice(-MAX_HISTORY);
    futureRef.current = []; // nayi action ke baad redo clear ho jaata hai
  }, []);

  /** Undo — history stack se pichla snapshot restore karo */
  const undo = useCallback(() => {
    const history = historyRef.current;
    if (history.length === 0) return;
    const prev = history[history.length - 1];
    // Current state ko future mein push karo
    futureRef.current = [
      {
        tracks: JSON.parse(JSON.stringify(tracksRef.current)),
        mediaFiles: JSON.parse(JSON.stringify(mediaFilesRef.current)),
        duration: durationRef.current,
      },
      ...futureRef.current,
    ];
    historyRef.current = history.slice(0, -1);
    setTracks(prev.tracks);
    setMediaFiles(prev.mediaFiles);
    setDuration(prev.duration);
    setSelectedClipIds([]);
  }, []);

  /** Redo — future stack se agla snapshot restore karo */
  const redo = useCallback(() => {
    const future = futureRef.current;
    if (future.length === 0) return;
    const next = future[0];
    // Current state ko history mein push karo
    historyRef.current = [
      ...historyRef.current,
      {
        tracks: JSON.parse(JSON.stringify(tracksRef.current)),
        mediaFiles: JSON.parse(JSON.stringify(mediaFilesRef.current)),
        duration: durationRef.current,
      },
    ].slice(-MAX_HISTORY);
    futureRef.current = future.slice(1);
    setTracks(next.tracks);
    setMediaFiles(next.mediaFiles);
    setDuration(next.duration);
    setSelectedClipIds([]);
  }, []);

  // ── Keyboard shortcuts: Ctrl+Z undo, Ctrl+Shift+Z redo ───────────────────
  const shortcutsRegistered = useRef(false);
  if (!shortcutsRegistered.current) {
    shortcutsRegistered.current = true;
    setTimeout(() => {
      window.addEventListener('keydown', (e: KeyboardEvent) => {
        const tag = (e.target as HTMLElement).tagName;
        if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
        if ((e.ctrlKey || e.metaKey) && e.key === 'z' && !e.shiftKey) {
          e.preventDefault();
          undoRef.current();
        } else if (
          (e.ctrlKey || e.metaKey) &&
          (e.key === 'y' || (e.key === 'z' && e.shiftKey))
        ) {
          e.preventDefault();
          redoRef.current();
        }
      });
    }, 0);
  }

  // Refs so the event listener always calls latest undo/redo
  const undoRef = useRef(undo);
  const redoRef = useRef(redo);
  undoRef.current = undo;
  redoRef.current = redo;

  const selectedClipId = selectedClipIds[0] ?? null;

  // ── Media files ──────────────────────────────────────────────────────────
  const addMediaFile = useCallback((file: MediaFile) => {
    pushHistory();
    setMediaFiles(prev => [...prev, file]);
  }, [pushHistory]);

  const removeMediaFile = useCallback((id: string) => {
    pushHistory();
    setMediaFiles(prev => prev.filter(f => f.id !== id));
    setTracks(prev =>
      prev.map(track => ({
        ...track,
        clips: track.clips.filter(c => c.mediaId !== id),
      }))
    );
  }, [pushHistory]);

  // ── Add clip to track — Automation settings apply hoti hain yahan ────────
  const addClipToTrack = useCallback(
    (media: MediaFile, trackIndex: number, startTime?: number) => {
      pushHistory();
      setTracks(prev => {
        if (!prev[trackIndex]) return prev;
        const track = prev[trackIndex];
        if (!isMediaAllowedOnTrack(media.type, track.type)) return prev;

        const automation = automationRef.current;

        // ── Duration decide karo — Automation ON hai tabhi apply ho ──────────
        let clipDuration: number;
        if (automation.enabled && media.type === 'image') {
          clipDuration = Math.max(0.1, automation.imageDuration);
        } else if (automation.enabled && media.type === 'video') {
          if (automation.useCustomVideoDuration && automation.videoDuration > 0) {
            clipDuration = Math.max(0.1, automation.videoDuration);
          } else {
            clipDuration = media.duration ?? 5;
          }
        } else {
          clipDuration = media.duration ?? 5;
        }

        let rawStart: number;
        if (startTime !== undefined) {
          rawStart = startTime;
        } else {
          rawStart = track.clips.reduce(
            (max, c) => Math.max(max, c.startTime + c.duration), 0
          );
        }
        const st = resolveNonOverlappingStart(track.clips, rawStart, clipDuration);

        // ── Base transform ───────────────────────────────────────────────────
        const transform = defaultTransform();

        // ── Auto Keyframes — Automation ON aur settings se ──────────────────
        const akSettings = automation.autoKeyframes;
        const shouldApplyKF =
          automation.enabled &&
          ((media.type === 'image' && akSettings.applyToImages) ||
          (media.type === 'video' && akSettings.applyToVideos));

        if (shouldApplyKF) {
          const sv = akSettings.startValues;
          const ev = akSettings.endValues;

          const kfStart = st + Math.min(akSettings.startTime, clipDuration);
          const kfEnd   = st + Math.min(akSettings.endTime,   clipDuration);

          if (kfStart < kfEnd) {
            const makeKF = (time: number, value: number): Keyframe => ({
              id: generateId(),
              time,
              value,
            });

            transform.scaleKFs   = [makeKF(kfStart, sv.scale),   makeKF(kfEnd, ev.scale)];
            transform.opacityKFs = [makeKF(kfStart, sv.opacity),  makeKF(kfEnd, ev.opacity)];
            transform.posXKFs    = [makeKF(kfStart, sv.posX),     makeKF(kfEnd, ev.posX)];
            transform.posYKFs    = [makeKF(kfStart, sv.posY),     makeKF(kfEnd, ev.posY)];

            const effectiveSpeed = sv.speed;
            if (Math.abs(effectiveSpeed - 1) > 0.01) {
              const origDur = media.duration ?? clipDuration;
              clipDuration = origDur / effectiveSpeed;
            }

            transform.scale   = sv.scale;
            transform.opacity = sv.opacity;
            transform.posX    = sv.posX;
            transform.posY    = sv.posY;
          }
        }

        const newClip: TimelineClip = {
          id: generateId(),
          mediaId: media.id,
          trackIndex,
          startTime: st,
          duration: clipDuration,
          originalDuration: media.duration ?? clipDuration,
          trimStart: 0,
          trimEnd: 0,
          name: media.name,
          type: media.type,
          color: CLIP_COLORS[media.type],
          url: media.url,
          thumbnail: media.thumbnail,
          waveformData: media.waveformData,
          transform,
          speed: 1,
        };

        const updated = [...prev];
        updated[trackIndex] = {
          ...updated[trackIndex],
          clips: [...updated[trackIndex].clips, newClip],
        };

        const newEnd = st + clipDuration;
        setDuration(d => Math.max(d, newEnd + 5));

        return updated;
      });
    },
    [pushHistory]
  );

  // ── Move clip — keyframes bhi clip ke saath shift hote hain ─────────────
  const moveClip = useCallback(
    (clipId: string, newStartTime: number, newTrackIndex: number) => {
      setTracks(prev => {
        const clip = prev.flatMap(t => t.clips).find(c => c.id === clipId);
        if (!clip || !prev[newTrackIndex]) return prev;
        if (!isMediaAllowedOnTrack(clip.type, prev[newTrackIndex].type)) return prev;

        const updated = prev.map(track => ({
          ...track,
          clips: track.clips.filter(c => c.id !== clipId),
        }));

        const targetClips = updated[newTrackIndex].clips;
        const st = resolveNonOverlappingStart(
          targetClips,
          Math.max(0, newStartTime),
          clip.duration
        );

        const timeDelta = st - clip.startTime;
        const shiftKFs = (kfs: Keyframe[]): Keyframe[] =>
          kfs.map(kf => ({ ...kf, time: kf.time + timeDelta }));

        const movedClip: TimelineClip = {
          ...clip,
          startTime: st,
          trackIndex: newTrackIndex,
          transform: {
            ...clip.transform,
            scaleKFs:   shiftKFs(clip.transform.scaleKFs),
            opacityKFs: shiftKFs(clip.transform.opacityKFs),
            posXKFs:    shiftKFs(clip.transform.posXKFs),
            posYKFs:    shiftKFs(clip.transform.posYKFs),
          },
        };

        updated[newTrackIndex] = {
          ...updated[newTrackIndex],
          clips: [...updated[newTrackIndex].clips, movedClip],
        };

        return updated;
      });
    },
    []
  );

  // ── Move clip with history (mouseup pe call karo) ────────────────────────
  const moveClipWithHistory = useCallback(
    (clipId: string, newStartTime: number, newTrackIndex: number) => {
      pushHistory();
      moveClip(clipId, newStartTime, newTrackIndex);
    },
    [pushHistory, moveClip]
  );

  // ── Remove clip ──────────────────────────────────────────────────────────
  const removeClip = useCallback((clipId: string) => {
    pushHistory();
    setTracks(prev =>
      prev.map(track => ({
        ...track,
        clips: track.clips.filter(c => c.id !== clipId),
      }))
    );
    setSelectedClipIds(prev => prev.filter(id => id !== clipId));
  }, [pushHistory]);

  // ── Split clip at playhead (Ctrl+B) ──────────────────────────────────────
  // Clip ko currentTime pe do alag clips mein tod deta hai
  const splitClip = useCallback(
    (clipId: string, splitTime: number) => {
      pushHistory();
      setTracks(prev =>
        prev.map(track => {
          const clipIdx = track.clips.findIndex(c => c.id === clipId);
          if (clipIdx === -1) return track;
          const clip = track.clips[clipIdx];

          // Split time clip ke andar hona chahiye
          if (splitTime <= clip.startTime || splitTime >= clip.startTime + clip.duration) {
            return track;
          }

          const leftDuration  = splitTime - clip.startTime;
          const rightDuration = clip.duration - leftDuration;

          // ── Left clip — original clip start se splitTime tak ─────────────
          const leftClip: TimelineClip = {
            ...clip,
            id: generateId(),
            duration: leftDuration,
            originalDuration: leftDuration,
            trimEnd: clip.trimEnd + rightDuration,
            // Keyframes jo left part mein hain woh rakho
            transform: {
              ...clip.transform,
              scaleKFs:   clip.transform.scaleKFs.filter(kf => kf.time < splitTime),
              opacityKFs: clip.transform.opacityKFs.filter(kf => kf.time < splitTime),
              posXKFs:    clip.transform.posXKFs.filter(kf => kf.time < splitTime),
              posYKFs:    clip.transform.posYKFs.filter(kf => kf.time < splitTime),
            },
          };

          // ── Right clip — splitTime se clip end tak ────────────────────────
          const rightClip: TimelineClip = {
            ...clip,
            id: generateId(),
            startTime: splitTime,
            duration: rightDuration,
            originalDuration: rightDuration,
            trimStart: clip.trimStart + leftDuration,
            // Keyframes jo right part mein hain woh rakho
            transform: {
              ...clip.transform,
              scaleKFs:   clip.transform.scaleKFs.filter(kf => kf.time >= splitTime),
              opacityKFs: clip.transform.opacityKFs.filter(kf => kf.time >= splitTime),
              posXKFs:    clip.transform.posXKFs.filter(kf => kf.time >= splitTime),
              posYKFs:    clip.transform.posYKFs.filter(kf => kf.time >= splitTime),
            },
          };

          // Original clip ko remove karo, dono naye clips add karo
          const newClips = [...track.clips];
          newClips.splice(clipIdx, 1, leftClip, rightClip);

          return { ...track, clips: newClips };
        })
      );

      // Split ke baad dono clips deselect karo
      setSelectedClipIds([]);
    },
    [pushHistory]
  );

  // ── Split all selected clips at currentTime (Ctrl+B handler) ─────────────
  const splitSelectedClipsAtPlayhead = useCallback(
    (splitTime: number) => {
      // Agar koi clip selected hai toh sirf unhe split karo
      // Agar koi selected nahi hai toh playhead pe jo bhi clips hain unhe split karo
      const currentTracks = tracksRef.current;
      const allClips = currentTracks.flatMap(t => t.clips);
      const selectedIds = selectedClipIdsRef.current;

      let clipsToSplit: string[];
      if (selectedIds.length > 0) {
        // Selected clips mein se jo splitTime pe hain
        clipsToSplit = allClips
          .filter(c =>
            selectedIds.includes(c.id) &&
            splitTime > c.startTime &&
            splitTime < c.startTime + c.duration
          )
          .map(c => c.id);
      } else {
        // Koi selected nahi — playhead pe jo bhi clips hain
        clipsToSplit = allClips
          .filter(c =>
            splitTime > c.startTime &&
            splitTime < c.startTime + c.duration
          )
          .map(c => c.id);
      }

      if (clipsToSplit.length === 0) return;

      // Ek pushHistory for the whole split operation
      pushHistory();

      setTracks(prev =>
        prev.map(track => {
          const anyToSplit = track.clips.some(c => clipsToSplit.includes(c.id));
          if (!anyToSplit) return track;

          const newClips: TimelineClip[] = [];
          for (const clip of track.clips) {
            if (!clipsToSplit.includes(clip.id)) {
              newClips.push(clip);
              continue;
            }
            const leftDuration  = splitTime - clip.startTime;
            const rightDuration = clip.duration - leftDuration;

            const leftClip: TimelineClip = {
              ...clip,
              id: generateId(),
              duration: leftDuration,
              originalDuration: leftDuration,
              trimEnd: clip.trimEnd + rightDuration,
              transform: {
                ...clip.transform,
                scaleKFs:   clip.transform.scaleKFs.filter(kf => kf.time < splitTime),
                opacityKFs: clip.transform.opacityKFs.filter(kf => kf.time < splitTime),
                posXKFs:    clip.transform.posXKFs.filter(kf => kf.time < splitTime),
                posYKFs:    clip.transform.posYKFs.filter(kf => kf.time < splitTime),
              },
            };

            const rightClip: TimelineClip = {
              ...clip,
              id: generateId(),
              startTime: splitTime,
              duration: rightDuration,
              originalDuration: rightDuration,
              trimStart: clip.trimStart + leftDuration,
              transform: {
                ...clip.transform,
                scaleKFs:   clip.transform.scaleKFs.filter(kf => kf.time >= splitTime),
                opacityKFs: clip.transform.opacityKFs.filter(kf => kf.time >= splitTime),
                posXKFs:    clip.transform.posXKFs.filter(kf => kf.time >= splitTime),
                posYKFs:    clip.transform.posYKFs.filter(kf => kf.time >= splitTime),
              },
            };

            newClips.push(leftClip, rightClip);
          }
          return { ...track, clips: newClips };
        })
      );

      setSelectedClipIds([]);
    },
    [pushHistory]
  );

  // ── Add track ────────────────────────────────────────────────────────────
  const addTrack = useCallback((type: 'video' | 'audio') => {
    pushHistory();
    setTracks(prev => {
      const existing = prev.filter(t => t.type === type);
      const num = existing.length + 1;
      const label = type === 'video' ? `Video ${num}` : `Audio ${num}`;
      const newTrack: Track = {
        id: generateId(),
        label,
        type,
        clips: [],
      };
      const lastIdx = prev.reduce(
        (acc, t, i) => (t.type === type ? i : acc),
        -1
      );
      const updated = [...prev];
      updated.splice(lastIdx + 1, 0, newTrack);
      return updated;
    });
  }, [pushHistory]);

  // ── Remove track ─────────────────────────────────────────────────────────
  const removeTrack = useCallback((trackId: string) => {
    pushHistory();
    setTracks(prev => {
      const track = prev.find(t => t.id === trackId);
      if (!track) return prev;
      const clipIds = track.clips.map(c => c.id);
      setSelectedClipIds(s => s.filter(id => !clipIds.includes(id)));
      return prev.filter(t => t.id !== trackId);
    });
  }, [pushHistory]);

  // ── Selection ────────────────────────────────────────────────────────────
  const selectClip = useCallback((clipId: string | null, addToSelection = false) => {
    if (clipId === null) {
      setSelectedClipIds([]);
      return;
    }
    if (addToSelection) {
      setSelectedClipIds(prev =>
        prev.includes(clipId)
          ? prev.filter(id => id !== clipId)
          : [...prev, clipId]
      );
    } else {
      setSelectedClipIds([clipId]);
    }
  }, []);

  // ── Update transform (static values) ─────────────────────────────────────
  const updateTransform = useCallback(
    (clipIds: string[], partial: Partial<ClipTransform>) => {
      setTracks(prev =>
        prev.map(track => ({
          ...track,
          clips: track.clips.map(clip => {
            if (!clipIds.includes(clip.id)) return clip;
            return {
              ...clip,
              transform: { ...clip.transform, ...partial },
            };
          }),
        }))
      );
    },
    []
  );

  // ── Add keyframe at a time for a property ───────────────────────────────
  const addKeyframe = useCallback(
    (
      clipIds: string[],
      prop: 'scale' | 'opacity' | 'posX' | 'posY',
      time: number,
      value?: number
    ) => {
      pushHistory();
      setTracks(prev =>
        prev.map(track => ({
          ...track,
          clips: track.clips.map(clip => {
            if (!clipIds.includes(clip.id)) return clip;
            const clampedTime = Math.max(
              clip.startTime,
              Math.min(clip.startTime + clip.duration, time)
            );
            const kfProp = `${prop}KFs` as 'scaleKFs' | 'opacityKFs' | 'posXKFs' | 'posYKFs';
            const existingKFs = clip.transform[kfProp] as Keyframe[];

            let currentVal: number;
            if (value !== undefined) {
              currentVal = value;
            } else if (existingKFs.length > 0) {
              currentVal = interpolateKF(existingKFs, clip.transform[prop] as number, clampedTime);
            } else {
              currentVal = clip.transform[prop] as number;
            }

            const filtered = existingKFs.filter(kf => Math.abs(kf.time - clampedTime) > 0.05);
            const newKF: Keyframe = {
              id: generateId(),
              time: clampedTime,
              value: currentVal,
            };
            return {
              ...clip,
              transform: {
                ...clip.transform,
                [kfProp]: [...filtered, newKF].sort((a, b) => a.time - b.time),
              },
            };
          }),
        }))
      );
    },
    [pushHistory]
  );

  // ── Remove keyframe ──────────────────────────────────────────────────────
  const removeKeyframe = useCallback(
    (clipId: string, prop: 'scale' | 'opacity' | 'posX' | 'posY', kfId: string) => {
      pushHistory();
      setTracks(prev =>
        prev.map(track => ({
          ...track,
          clips: track.clips.map(clip => {
            if (clip.id !== clipId) return clip;
            const kfProp = `${prop}KFs` as 'scaleKFs' | 'opacityKFs' | 'posXKFs' | 'posYKFs';
            return {
              ...clip,
              transform: {
                ...clip.transform,
                [kfProp]: (clip.transform[kfProp] as Keyframe[]).filter(kf => kf.id !== kfId),
              },
            };
          }),
        }))
      );
    },
    [pushHistory]
  );

  // ── Update clip speed — duration bhi adjust hoti hai ─────────────────────
  const updateClipSpeed = useCallback(
    (clipIds: string[], newSpeed: number) => {
      const speed = Math.max(0.1, Math.min(10, newSpeed));
      pushHistory();
      setTracks(prev =>
        prev.map(track => ({
          ...track,
          clips: track.clips.map(clip => {
            if (!clipIds.includes(clip.id)) return clip;
            const origDur = clip.originalDuration ?? clip.duration;
            const newDuration = origDur / speed;
            const scaleKFTime = (kf: Keyframe): Keyframe => ({
              ...kf,
              time: clip.startTime + (kf.time - clip.startTime) * (clip.speed / speed),
            });
            return {
              ...clip,
              speed,
              originalDuration: origDur,
              duration: newDuration,
              transform: {
                ...clip.transform,
                scaleKFs:   clip.transform.scaleKFs.map(scaleKFTime),
                opacityKFs: clip.transform.opacityKFs.map(scaleKFTime),
                posXKFs:    clip.transform.posXKFs.map(scaleKFTime),
                posYKFs:    clip.transform.posYKFs.map(scaleKFTime),
              },
            };
          }),
        }))
      );
    },
    [pushHistory]
  );

  // ── Move keyframe ────────────────────────────────────────────────────────
  const moveKeyframe = useCallback(
    (clipId: string, prop: 'scale' | 'opacity' | 'posX' | 'posY', kfId: string, newTime: number) => {
      setTracks(prev =>
        prev.map(track => ({
          ...track,
          clips: track.clips.map(clip => {
            if (clip.id !== clipId) return clip;
            const kfProp = `${prop}KFs` as 'scaleKFs' | 'opacityKFs' | 'posXKFs' | 'posYKFs';
            const clampedTime = Math.max(
              clip.startTime,
              Math.min(clip.startTime + clip.duration, newTime)
            );
            const updated = (clip.transform[kfProp] as Keyframe[])
              .map(kf => kf.id === kfId ? { ...kf, time: clampedTime } : kf)
              .sort((a, b) => a.time - b.time);
            return {
              ...clip,
              transform: { ...clip.transform, [kfProp]: updated },
            };
          }),
        }))
      );
    },
    []
  );

  // ── Move keyframe with history ────────────────────────────────────────────
  const moveKeyframeWithHistory = useCallback(
    (clipId: string, prop: 'scale' | 'opacity' | 'posX' | 'posY', kfId: string, newTime: number) => {
      pushHistory();
      moveKeyframe(clipId, prop, kfId, newTime);
    },
    [pushHistory, moveKeyframe]
  );

  // ── Update automation settings ────────────────────────────────────────────
  const updateAutomationSettings = useCallback(
    (partial: Partial<AutomationSettings>) => {
      setAutomationSettings(prev => ({ ...prev, ...partial }));
    },
    []
  );

  // ── Playback — 60fps (1000ms / 60 = 16.667ms interval) ──────────────────
  const FPS = 60;
  const FRAME_MS = 1000 / FPS;           // 16.667ms
  const FRAME_SEC = 1 / FPS;             // 0.01667s

  const play = useCallback(() => {
    setIsPlaying(true);
    // requestAnimationFrame-style 60fps loop via setInterval at ~16.67ms
    playIntervalRef.current = setInterval(() => {
      setCurrentTime(prev => {
        if (prev >= durationRef.current) {
          clearInterval(playIntervalRef.current!);
          setIsPlaying(false);
          return 0;
        }
        return prev + FRAME_SEC;
      });
    }, FRAME_MS);
  }, [FRAME_MS, FRAME_SEC]);

  const pause = useCallback(() => {
    setIsPlaying(false);
    if (playIntervalRef.current) clearInterval(playIntervalRef.current);
  }, []);

  const seek = useCallback(
    (time: number) => {
      setCurrentTime(Math.max(0, Math.min(time, durationRef.current)));
    },
    []
  );

  const getActiveClipsAt = useCallback(
    (time: number) => {
      return tracks.flatMap(track =>
        track.clips.filter(c => time >= c.startTime && time < c.startTime + c.duration)
      );
    },
    [tracks]
  );

  return {
    mediaFiles,
    tracks,
    currentTime,
    isPlaying,
    duration,
    selectedClipId,
    selectedClipIds,
    zoom,
    aspectRatio,
    automationSettings,
    addMediaFile,
    removeMediaFile,
    addClipToTrack,
    moveClip,
    moveClipWithHistory,
    removeClip,
    splitClip,
    splitSelectedClipsAtPlayhead,
    addTrack,
    removeTrack,
    selectClip,
    updateTransform,
    updateClipSpeed,
    addKeyframe,
    removeKeyframe,
    moveKeyframe,
    moveKeyframeWithHistory,
    updateAutomationSettings,
    undo,
    redo,
    play,
    pause,
    seek,
    setCurrentTime,
    setSelectedClipIds,
    setZoom,
    setDuration,
    setAspectRatio,
    getActiveClipsAt,
    pushHistory,
  };
}
