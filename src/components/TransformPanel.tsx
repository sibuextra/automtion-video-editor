import { useState, useEffect } from 'react';
import { TimelineClip, ClipTransform, Keyframe } from '../types/editor';
import { interpolateKF } from '../hooks/useEditorStore';

type KFProp = 'scale' | 'opacity' | 'posX' | 'posY';

interface Props {
  selectedClips: TimelineClip[];
  currentTime: number;
  onUpdateTransform: (clipIds: string[], partial: Partial<ClipTransform>) => void;
  onUpdateSpeed: (clipIds: string[], speed: number) => void;
  onAddKeyframe: (clipIds: string[], prop: KFProp, time: number, value?: number) => void;
  onRemoveKeyframe: (clipId: string, prop: KFProp, kfId: string) => void;
  onMoveKeyframe: (clipId: string, prop: KFProp, kfId: string, newTime: number) => void;
}

function getDisplayValue(clip: TimelineClip, prop: KFProp, currentTime: number): number {
  const t = clip.transform;
  const kfProp = `${prop}KFs` as keyof ClipTransform;
  const kfs = t[kfProp] as Keyframe[];
  if (kfs.length > 0) return interpolateKF(kfs, t[prop] as number, currentTime);
  return t[prop] as number;
}

function getMultiValue(clips: TimelineClip[], prop: KFProp, currentTime: number): number {
  if (clips.length === 0) return prop === 'scale' ? 1 : prop === 'opacity' ? 1 : 0;
  const sum = clips.reduce((acc, c) => acc + getDisplayValue(c, prop, currentTime), 0);
  return sum / clips.length;
}

function hasAnyKeyframes(clips: TimelineClip[], prop: KFProp): boolean {
  const kfProp = `${prop}KFs` as keyof ClipTransform;
  return clips.some(c => (c.transform[kfProp] as Keyframe[]).length > 0);
}

function KFDot({ kfs, currentTime }: { kfs: Keyframe[]; currentTime: number }) {
  const hasKFAtTime = kfs.some(kf => Math.abs(kf.time - currentTime) < 0.06);
  return (
    <div
      className={`w-3 h-3 rounded-full border-2 flex-shrink-0 transition-colors ${
        hasKFAtTime
          ? 'bg-yellow-400 border-yellow-300 shadow-sm'
          : kfs.length > 0
          ? 'bg-zinc-600 border-indigo-400'
          : 'bg-zinc-700 border-zinc-600'
      }`}
    />
  );
}

interface SliderRowProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  displayValue: string;
  kfs: Keyframe[];
  hasKFs: boolean;
  currentTime: number;
  disabled: boolean;
  onChange: (v: number) => void;
  onAddKF: () => void;
  onInputChange: (v: string) => void;
}

function SliderRow({
  label, value, min, max, step, displayValue,
  kfs, hasKFs, currentTime, disabled,
  onChange, onAddKF, onInputChange,
}: SliderRowProps) {
  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1.5">
          <span className="text-[10px] text-zinc-400 font-semibold uppercase tracking-wider">{label}</span>
          {hasKFs && (
            <span className="text-[8px] bg-indigo-600/30 border border-indigo-500/40 text-indigo-300 px-1 rounded">KF</span>
          )}
        </div>
        <div className="flex items-center gap-1.5">
          <KFDot kfs={kfs} currentTime={currentTime} />
          <button
            disabled={disabled}
            onClick={onAddKF}
            className={`text-[9px] px-1.5 py-0.5 rounded border transition-colors ${
              disabled
                ? 'bg-zinc-800 border-zinc-700 text-zinc-600 cursor-not-allowed'
                : 'bg-zinc-800 border-zinc-700 text-zinc-400 hover:bg-indigo-600/30 hover:border-indigo-500 hover:text-indigo-300 cursor-pointer'
            }`}
          >
            ◆ KF
          </button>
        </div>
      </div>
      <div className="flex items-center gap-2">
        <input type="range" min={min} max={max} step={step} value={value} disabled={disabled}
          onChange={e => onChange(Number(e.target.value))}
          className="flex-1 accent-indigo-500 h-1 cursor-pointer disabled:opacity-40" />
        <input type="number" value={displayValue} disabled={disabled} min={min} max={max} step={step}
          onChange={e => onInputChange(e.target.value)}
          className="w-14 text-[11px] bg-zinc-800 border border-zinc-700 text-zinc-200 rounded px-1.5 py-0.5 text-right disabled:opacity-40" />
      </div>
    </div>
  );
}

function SpeedRow({ speed, disabled, onChange }: { speed: number; disabled: boolean; onChange: (v: number) => void }) {
  const displayPct = Math.round(speed * 100);
  const presets = [{ label: '25%', val: 0.25 }, { label: '50%', val: 0.5 }, { label: '1×', val: 1 }, { label: '2×', val: 2 }, { label: '4×', val: 4 }];

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1.5">
          <svg className="h-3 w-3 text-amber-400" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
          <span className="text-[10px] text-zinc-400 font-semibold uppercase tracking-wider">Speed</span>
        </div>
        <span className="text-[10px] text-amber-300 font-mono bg-amber-500/10 border border-amber-500/20 px-1.5 py-0.5 rounded">{displayPct}%</span>
      </div>
      <div className="flex items-center gap-2">
        <input type="range" min={0.1} max={4} step={0.05} value={speed} disabled={disabled}
          onChange={e => onChange(Number(e.target.value))}
          className="flex-1 h-1 cursor-pointer disabled:opacity-40" style={{ accentColor: '#f59e0b' }} />
        <input type="number" value={displayPct} disabled={disabled} min={10} max={400} step={5}
          onChange={e => { const pct = parseFloat(e.target.value); if (!isNaN(pct)) onChange(Math.max(0.1, Math.min(4, pct / 100))); }}
          className="w-14 text-[11px] bg-zinc-800 border border-zinc-700 text-zinc-200 rounded px-1.5 py-0.5 text-right disabled:opacity-40" />
      </div>
      <div className="flex items-center gap-1 flex-wrap">
        {presets.map(p => (
          <button key={p.label} disabled={disabled} onClick={() => onChange(p.val)}
            className={`text-[9px] px-2 py-0.5 rounded border transition-colors ${
              Math.abs(speed - p.val) < 0.01
                ? 'bg-amber-500/30 border-amber-500/60 text-amber-300'
                : 'bg-zinc-800 border-zinc-700 text-zinc-500 hover:bg-zinc-700 hover:text-zinc-300 disabled:opacity-40 disabled:cursor-not-allowed'
            }`}>
            {p.label}
          </button>
        ))}
      </div>
    </div>
  );
}

interface KFItemProps {
  kf: { id: string; time: number; value: number; prop: KFProp; propLabel: string };
  clip: TimelineClip;
  isSelected: boolean;
  onSelect: () => void;
  onRemove: () => void;
  onMove: (newTime: number) => void;
}

function KFItem({ kf, clip, isSelected, onSelect, onRemove, onMove }: KFItemProps) {
  const formatVal = (prop: KFProp, val: number) => {
    if (prop === 'scale' || prop === 'opacity') return `${(val * 100).toFixed(0)}%`;
    return `${val.toFixed(0)}px`;
  };

  const handleTimeDragMouseDown = (e: React.MouseEvent) => {
    e.stopPropagation();
    e.preventDefault();
    const startX = e.clientX;
    const origTime = kf.time;
    const onMove_ = (ev: MouseEvent) => {
      const dt = (ev.clientX - startX) * 0.02;
      const newTime = Math.max(clip.startTime, Math.min(clip.startTime + clip.duration, origTime + dt));
      onMove(newTime);
    };
    const onUp = () => { window.removeEventListener('mousemove', onMove_); window.removeEventListener('mouseup', onUp); };
    window.addEventListener('mousemove', onMove_);
    window.addEventListener('mouseup', onUp);
  };

  return (
    <div
      className={`flex items-center justify-between rounded px-2 py-1 cursor-pointer select-none transition-colors ${
        isSelected ? 'bg-yellow-500/20 border border-yellow-500/50' : 'bg-zinc-800/50 border border-transparent hover:bg-zinc-800'
      }`}
      onClick={onSelect}
    >
      <div className="flex items-center gap-1.5">
        <div className={`w-1.5 h-1.5 rounded-full ${isSelected ? 'bg-yellow-400' : 'bg-yellow-500/70'}`} />
        <span className="text-[10px] text-zinc-400">{kf.propLabel}</span>
      </div>
      <div className="flex items-center gap-1">
        <span className="text-[10px] text-zinc-500 font-mono cursor-col-resize bg-zinc-700/60 rounded px-1 hover:bg-zinc-600/60"
          onMouseDown={handleTimeDragMouseDown} onClick={e => e.stopPropagation()}>
          {kf.time.toFixed(2)}s
        </span>
        <span className="text-[10px] text-indigo-300 font-mono">{formatVal(kf.prop, kf.value)}</span>
        <button className="w-3.5 h-3.5 flex items-center justify-center rounded bg-red-600/50 hover:bg-red-500 text-white transition-colors ml-0.5"
          onClick={e => { e.stopPropagation(); onRemove(); }}>
          <svg className="h-2 w-2" fill="none" stroke="currentColor" strokeWidth={3} viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>
    </div>
  );
}

export default function TransformPanel({
  selectedClips, currentTime, onUpdateTransform, onUpdateSpeed,
  onAddKeyframe, onRemoveKeyframe, onMoveKeyframe,
}: Props) {
  const validClips = selectedClips.filter(c => c.type !== 'audio');
  const allSelected = selectedClips;
  const isDisabled = validClips.length === 0;
  const noClipsSelected = selectedClips.length === 0;

  const [selectedKF, setSelectedKF] = useState<{ clipId: string; prop: KFProp; kfId: string } | null>(null);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.key === 'Backspace' || e.key === 'Delete') && selectedKF) {
        const tag = (e.target as HTMLElement).tagName;
        if (tag === 'INPUT' || tag === 'TEXTAREA') return;
        e.preventDefault();
        onRemoveKeyframe(selectedKF.clipId, selectedKF.prop, selectedKF.kfId);
        setSelectedKF(null);
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [selectedKF, onRemoveKeyframe]);

  if (noClipsSelected) {
    return (
      <div className="flex flex-col h-full bg-zinc-900">
        <div className="px-3 py-2 border-b border-zinc-800 flex-shrink-0">
          <span className="text-[11px] font-semibold text-zinc-500 uppercase tracking-wider">Transformation</span>
        </div>
        <div className="flex-1 flex flex-col items-center justify-center gap-3 text-zinc-600 p-4 text-center">
          <svg className="h-10 w-10 text-zinc-700" fill="none" stroke="currentColor" strokeWidth={1.5} viewBox="0 0 24 24">
            <circle cx="12" cy="12" r="3" /><path strokeLinecap="round" strokeLinejoin="round" d="M12 1v4M12 19v4M4.22 4.22l2.83 2.83M16.95 16.95l2.83 2.83M1 12h4M19 12h4M4.22 19.78l2.83-2.83M16.95 7.05l2.83-2.83" />
          </svg>
          <p className="text-xs leading-relaxed">Click a clip on the<br />timeline to select it</p>
          <p className="text-[10px] text-zinc-700">Ctrl+Click for multi-select</p>
        </div>
      </div>
    );
  }

  const isAudioOnly = validClips.length === 0 && selectedClips.length > 0;
  const avgSpeed = allSelected.reduce((acc, c) => acc + (c.speed ?? 1), 0) / allSelected.length;

  const scaleVal   = isDisabled ? 1 : getMultiValue(validClips, 'scale', currentTime);
  const opacityVal = isDisabled ? 1 : getMultiValue(validClips, 'opacity', currentTime);
  const posXVal    = isDisabled ? 0 : getMultiValue(validClips, 'posX', currentTime);
  const posYVal    = isDisabled ? 0 : getMultiValue(validClips, 'posY', currentTime);

  const firstClip  = validClips[0];
  const scaleKFs   = firstClip?.transform.scaleKFs   ?? [];
  const opacityKFs = firstClip?.transform.opacityKFs ?? [];
  const posXKFs    = firstClip?.transform.posXKFs    ?? [];
  const posYKFs    = firstClip?.transform.posYKFs    ?? [];

  const scaleHasKF   = hasAnyKeyframes(validClips, 'scale');
  const opacityHasKF = hasAnyKeyframes(validClips, 'opacity');
  const posXHasKF    = hasAnyKeyframes(validClips, 'posX');
  const posYHasKF    = hasAnyKeyframes(validClips, 'posY');

  const validIds = validClips.map(c => c.id);
  const allIds   = allSelected.map(c => c.id);

  const handleChange = (prop: KFProp, value: number) => {
    if (isDisabled) return;
    const kfProp = `${prop}KFs` as keyof typeof firstClip.transform;
    const kfsForProp = firstClip ? (firstClip.transform[kfProp] as Keyframe[]) : [];
    if (kfsForProp.length > 0) {
      onAddKeyframe(validIds, prop, currentTime, value);
    } else {
      onUpdateTransform(validIds, { [prop]: value });
    }
  };

  const allKFs: { id: string; time: number; value: number; prop: KFProp; propLabel: string }[] = firstClip
    ? [
        ...firstClip.transform.scaleKFs.map(k   => ({ ...k, prop: 'scale'   as KFProp, propLabel: 'Scale'   })),
        ...firstClip.transform.opacityKFs.map(k  => ({ ...k, prop: 'opacity' as KFProp, propLabel: 'Opacity' })),
        ...firstClip.transform.posXKFs.map(k     => ({ ...k, prop: 'posX'    as KFProp, propLabel: 'Pos X'   })),
        ...firstClip.transform.posYKFs.map(k     => ({ ...k, prop: 'posY'    as KFProp, propLabel: 'Pos Y'   })),
      ].sort((a, b) => a.time - b.time)
    : [];

  return (
    <div className="flex flex-col h-full bg-zinc-900">
      <div className="px-3 py-2 border-b border-zinc-800 flex-shrink-0 flex items-center justify-between">
        <span className="text-[11px] font-semibold text-zinc-500 uppercase tracking-wider">Transformation</span>
        {selectedClips.length > 0 && (
          <span className="text-[10px] text-indigo-400 bg-indigo-600/20 px-1.5 py-0.5 rounded border border-indigo-500/30">
            {selectedClips.length === 1 ? selectedClips[0].name.slice(0, 12) : `${selectedClips.length} clips`}
          </span>
        )}
      </div>

      <div className="flex-1 overflow-y-auto p-3 flex flex-col gap-4">
        {isAudioOnly && (
          <div className="bg-zinc-800/50 rounded-lg p-3 text-center">
            <p className="text-[11px] text-zinc-500">Audio clips don't have visual transform properties.<br />Speed control is available below.</p>
          </div>
        )}

        <SpeedRow speed={avgSpeed} disabled={allSelected.length === 0} onChange={v => onUpdateSpeed(allIds, v)} />

        {!isAudioOnly && (
          <>
            <div className="h-px bg-zinc-800" />
            <div className="flex flex-col gap-2">
              <SliderRow label="Scale" value={scaleVal} min={0} max={5} step={0.01}
                displayValue={(scaleVal * 100).toFixed(0)} kfs={scaleKFs} hasKFs={scaleHasKF}
                currentTime={currentTime} disabled={isDisabled}
                onChange={v => handleChange('scale', v)} onAddKF={() => onAddKeyframe(validIds, 'scale', currentTime)}
                onInputChange={v => { const n = parseFloat(v); if (!isNaN(n)) handleChange('scale', Math.max(0, Math.min(5, n / 100))); }} />
            </div>
            <div className="h-px bg-zinc-800" />
            <div className="flex flex-col gap-2">
              <SliderRow label="Opacity" value={opacityVal} min={0} max={1} step={0.01}
                displayValue={(opacityVal * 100).toFixed(0)} kfs={opacityKFs} hasKFs={opacityHasKF}
                currentTime={currentTime} disabled={isDisabled}
                onChange={v => handleChange('opacity', v)} onAddKF={() => onAddKeyframe(validIds, 'opacity', currentTime)}
                onInputChange={v => { const n = parseFloat(v); if (!isNaN(n)) handleChange('opacity', Math.max(0, Math.min(1, n / 100))); }} />
            </div>
            <div className="h-px bg-zinc-800" />
            <div className="flex flex-col gap-3">
              <span className="text-[10px] text-zinc-400 font-semibold uppercase tracking-wider">Position</span>
              <SliderRow label="X" value={posXVal} min={-1000} max={1000} step={1}
                displayValue={posXVal.toFixed(0)} kfs={posXKFs} hasKFs={posXHasKF}
                currentTime={currentTime} disabled={isDisabled}
                onChange={v => handleChange('posX', v)} onAddKF={() => onAddKeyframe(validIds, 'posX', currentTime)}
                onInputChange={v => { const n = parseFloat(v); if (!isNaN(n)) handleChange('posX', Math.max(-1000, Math.min(1000, n))); }} />
              <SliderRow label="Y" value={posYVal} min={-1000} max={1000} step={1}
                displayValue={posYVal.toFixed(0)} kfs={posYKFs} hasKFs={posYHasKF}
                currentTime={currentTime} disabled={isDisabled}
                onChange={v => handleChange('posY', v)} onAddKF={() => onAddKeyframe(validIds, 'posY', currentTime)}
                onInputChange={v => { const n = parseFloat(v); if (!isNaN(n)) handleChange('posY', Math.max(-1000, Math.min(1000, n))); }} />
            </div>

            {!isDisabled && (
              <>
                <div className="h-px bg-zinc-800" />
                <button
                  onClick={() => { onUpdateTransform(validIds, { scale: 1, opacity: 1, posX: 0, posY: 0, scaleKFs: [], opacityKFs: [], posXKFs: [], posYKFs: [] }); setSelectedKF(null); }}
                  className="text-[11px] text-zinc-500 hover:text-zinc-300 bg-zinc-800 hover:bg-zinc-700 border border-zinc-700 rounded-lg px-3 py-1.5 transition-colors text-center">
                  Reset Transform
                </button>
              </>
            )}

            {firstClip && !isDisabled && allKFs.length > 0 && (
              <>
                <div className="h-px bg-zinc-800" />
                <div className="flex flex-col gap-1.5">
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] text-zinc-500 uppercase tracking-wider font-semibold">Keyframes ({allKFs.length})</span>
                    <span className="text-[9px] text-zinc-600">← drag time | Backspace=del</span>
                  </div>
                  <div className="flex flex-col gap-1 max-h-40 overflow-y-auto">
                    {allKFs.map((kf) => (
                      <KFItem key={kf.id} kf={kf} clip={firstClip}
                        isSelected={selectedKF?.kfId === kf.id}
                        onSelect={() => setSelectedKF(prev => prev?.kfId === kf.id ? null : { clipId: firstClip.id, prop: kf.prop, kfId: kf.id })}
                        onRemove={() => { onRemoveKeyframe(firstClip.id, kf.prop, kf.id); if (selectedKF?.kfId === kf.id) setSelectedKF(null); }}
                        onMove={newTime => onMoveKeyframe(firstClip.id, kf.prop, kf.id, newTime)} />
                    ))}
                  </div>
                </div>
              </>
            )}
          </>
        )}
      </div>
    </div>
  );
}
