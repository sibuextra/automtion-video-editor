export type MediaType = 'video' | 'audio' | 'image';

export type AspectRatio = '16:9' | '9:16' | '1:1' | '4:3' | '21:9';

export interface MediaFile {
  id: string;
  name: string;
  type: MediaType;
  url: string;
  duration?: number;
  width?: number;
  height?: number;
  thumbnail?: string;
  waveformData?: number[];
  file: File;
}

// ── Keyframe for a single property at a specific time ──────────────────────
export interface Keyframe {
  id: string;
  time: number;   // seconds on timeline (absolute)
  value: number;
}

// ── Per-clip transform data ────────────────────────────────────────────────
export interface ClipTransform {
  // Static values (used when no keyframes exist for that property)
  scale: number;       // 0..5 (1 = 100%)
  opacity: number;     // 0..1
  posX: number;        // -1000..1000 px offset
  posY: number;        // -1000..1000 px offset

  // Keyframe arrays
  scaleKFs: Keyframe[];
  opacityKFs: Keyframe[];
  posXKFs: Keyframe[];
  posYKFs: Keyframe[];
}

export function defaultTransform(): ClipTransform {
  return {
    scale: 1,
    opacity: 1,
    posX: 0,
    posY: 0,
    scaleKFs: [],
    opacityKFs: [],
    posXKFs: [],
    posYKFs: [],
  };
}

export interface TimelineClip {
  id: string;
  mediaId: string;
  trackIndex: number;
  startTime: number;
  duration: number;        // timeline pe actual duration (originalDuration / speed)
  originalDuration: number; // original media duration (speed change pe nahi badlega)
  trimStart: number;
  trimEnd: number;
  name: string;
  type: MediaType;
  color: string;
  url: string;
  thumbnail?: string;
  waveformData?: number[];
  transform: ClipTransform;
  speed: number;           // 0.1 = 10% (slow), 1 = normal, 4 = 400% (fast)
}

export interface Track {
  id: string;
  label: string;
  type: 'video' | 'audio' | 'mixed';
  clips: TimelineClip[];
}

// ── Automation Settings ────────────────────────────────────────────────────
// ── Per-keyframe transform values (start aur end ke alag alag) ───────────────
export interface KFPointValues {
  scale: number;    // 0..5
  opacity: number;  // 0..1
  posX: number;     // -1000..1000
  posY: number;     // -1000..1000
  speed: number;    // 0.1..4
}

export function defaultKFPointValues(): KFPointValues {
  return { scale: 1, opacity: 1, posX: 0, posY: 0, speed: 1 };
}

export interface AutoKeyframeSettings {
  applyToImages: boolean;
  applyToVideos: boolean;
  startTime: number;       // seconds from clip begin
  endTime: number;         // seconds from clip begin
  startValues: KFPointValues;  // start keyframe ki values
  endValues: KFPointValues;    // end keyframe ki values
}

export interface AutomationSettings {
  enabled: boolean;                // global ON/OFF toggle
  imageDuration: number;           // seconds — images ki default duration
  videoDuration: number;           // seconds — videos ki default duration
  useCustomVideoDuration: boolean; // true = videoDuration use karo, false = original
  autoKeyframes: AutoKeyframeSettings;
}

export function defaultAutomationSettings(): AutomationSettings {
  return {
    enabled: false,
    imageDuration: 5,
    videoDuration: 0,
    useCustomVideoDuration: false,
    autoKeyframes: {
      applyToImages: false,
      applyToVideos: false,
      startTime: 0,
      endTime: 5,
      startValues: defaultKFPointValues(),
      endValues: defaultKFPointValues(),
    },
  };
}
