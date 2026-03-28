import { useState } from 'react';
import { AutomationSettings, KFPointValues, defaultKFPointValues } from '../types/editor';

interface Props {
  settings: AutomationSettings;
  onChange: (partial: Partial<AutomationSettings>) => void;
}

function secToHMS(sec: number): { h: number; m: number; s: number } {
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = Math.floor(sec % 60);
  return { h, m, s };
}

function hmsToSec(h: number, m: number, s: number): number {
  return h * 3600 + m * 60 + s;
}

function formatDisplay(sec: number): string {
  const { h, m, s } = secToHMS(sec);
  if (h > 0) return `${h}h ${m}m ${s}s`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

interface TimeInputProps {
  label: string;
  value: number;
  min?: number;
  max?: number;
  onChange: (sec: number) => void;
  accentColor?: string;
}

function TimeInput({ label, value, min = 0, max = 3600 * 24, onChange, accentColor = '#6366f1' }: TimeInputProps) {
  const { h, m, s } = secToHMS(value);

  const update = (newH: number, newM: number, newS: number) => {
    const total = hmsToSec(newH, newM, newS);
    onChange(Math.max(min, Math.min(max, total)));
  };

  return (
    <div className="flex flex-col gap-1.5">
      <span className="text-[10px] text-zinc-400 font-semibold uppercase tracking-wider">{label}</span>
      <div className="flex items-center gap-1">
        <div className="flex flex-col items-center">
          <span className="text-[8px] text-zinc-600 mb-0.5">HH</span>
          <input
            type="number" min={0} max={23} value={h}
            onChange={e => update(Math.max(0, Math.min(23, parseInt(e.target.value) || 0)), m, s)}
            className="w-10 text-center text-[11px] bg-zinc-800 border border-zinc-700 text-zinc-200 rounded px-1 py-0.5 focus:border-indigo-500 focus:outline-none"
            style={{ borderColor: h > 0 ? accentColor + '88' : undefined }}
          />
        </div>
        <span className="text-zinc-600 text-xs mt-3">:</span>
        <div className="flex flex-col items-center">
          <span className="text-[8px] text-zinc-600 mb-0.5">MM</span>
          <input
            type="number" min={0} max={59} value={m}
            onChange={e => update(h, Math.max(0, Math.min(59, parseInt(e.target.value) || 0)), s)}
            className="w-10 text-center text-[11px] bg-zinc-800 border border-zinc-700 text-zinc-200 rounded px-1 py-0.5 focus:border-indigo-500 focus:outline-none"
            style={{ borderColor: m > 0 ? accentColor + '88' : undefined }}
          />
        </div>
        <span className="text-zinc-600 text-xs mt-3">:</span>
        <div className="flex flex-col items-center">
          <span className="text-[8px] text-zinc-600 mb-0.5">SS</span>
          <input
            type="number" min={0} max={59} value={s}
            onChange={e => update(h, m, Math.max(0, Math.min(59, parseInt(e.target.value) || 0)))}
            className="w-10 text-center text-[11px] bg-zinc-800 border border-zinc-700 text-zinc-200 rounded px-1 py-0.5 focus:border-indigo-500 focus:outline-none"
            style={{ borderColor: s > 0 ? accentColor + '88' : undefined }}
          />
        </div>
        <div className="ml-1 text-[10px] text-zinc-500 font-mono bg-zinc-800/60 rounded px-1.5 py-0.5 min-w-[44px] text-center">
          {formatDisplay(value)}
        </div>
      </div>
      <input
        type="range" min={min} max={Math.min(max, 300)} step={1}
        value={Math.min(value, 300)}
        onChange={e => onChange(Math.max(min, Number(e.target.value)))}
        className="w-full h-1 cursor-pointer rounded"
        style={{ accentColor }}
      />
      <div className="flex justify-between text-[8px] text-zinc-700">
        <span>0s</span><span>1m</span><span>2m</span><span>5m+</span>
      </div>
    </div>
  );
}

function SectionHeader({ icon, title }: { icon: React.ReactNode; title: string }) {
  return (
    <div className="flex items-center gap-2">
      <div className="w-6 h-6 rounded-md bg-zinc-800 flex items-center justify-center flex-shrink-0">
        {icon}
      </div>
      <p className="text-[11px] font-semibold text-zinc-300">{title}</p>
    </div>
  );
}

function ToggleCheck({
  checked, onChange, label, color = '#6366f1',
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label: string;
  color?: string;
}) {
  return (
    <label className="flex items-center gap-2 cursor-pointer group">
      <div
        className={`w-4 h-4 rounded border-2 flex items-center justify-center transition-colors flex-shrink-0 ${
          checked ? 'border-transparent' : 'border-zinc-600 bg-zinc-800'
        }`}
        style={{ backgroundColor: checked ? color : undefined, borderColor: checked ? color : undefined }}
        onClick={() => onChange(!checked)}
      >
        {checked && (
          <svg className="h-2.5 w-2.5 text-white" fill="none" stroke="currentColor" strokeWidth={3} viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
          </svg>
        )}
      </div>
      <span className="text-[11px] text-zinc-300 group-hover:text-zinc-100 transition-colors select-none">{label}</span>
    </label>
  );
}

function MiniSlider({
  label, value, min, max, step, displayFn, onChange, accentColor = '#6366f1', disabled = false,
}: {
  label: string; value: number; min: number; max: number; step: number;
  displayFn: (v: number) => string; onChange: (v: number) => void;
  accentColor?: string; disabled?: boolean;
}) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-[10px] text-zinc-500 w-12 flex-shrink-0 font-medium">{label}</span>
      <input
        type="range" min={min} max={max} step={step} value={value}
        disabled={disabled}
        onChange={e => onChange(Number(e.target.value))}
        className="flex-1 h-1 cursor-pointer disabled:opacity-30"
        style={{ accentColor }}
      />
      <input
        type="number"
        value={parseFloat(displayFn(value))}
        disabled={disabled}
        min={min} max={max} step={step}
        onChange={e => {
          const n = parseFloat(e.target.value);
          if (!isNaN(n)) onChange(Math.max(min, Math.min(max, n)));
        }}
        className="w-14 text-[10px] bg-zinc-800 border border-zinc-700 text-zinc-200 rounded px-1.5 py-0.5 text-right disabled:opacity-30"
      />
    </div>
  );
}

function KFPointEditor({
  label, accentColor, values, onChange,
}: {
  label: string;
  accentColor: string;
  values: KFPointValues;
  onChange: (v: KFPointValues) => void;
}) {
  const [open, setOpen] = useState(false);

  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center justify-between">
        <span className="text-[10px] text-zinc-400 font-semibold">{label}</span>
        <button
          onClick={() => setOpen(v => !v)}
          className={`flex items-center gap-1 text-[10px] px-2 py-0.5 rounded border transition-colors ${
            open
              ? 'bg-zinc-700 border-zinc-600 text-zinc-200'
              : 'bg-zinc-800 border-zinc-700 text-zinc-500 hover:bg-zinc-700 hover:text-zinc-300'
          }`}
        >
          <span className="font-mono text-[11px]">{open ? '↑' : '↓'}</span>
          <span>Settings</span>
        </button>
      </div>

      {open && (
        <div className="flex flex-col gap-2 bg-zinc-800/50 rounded-lg p-2.5 border border-zinc-700/50 mt-0.5">
          <MiniSlider label="Scale" value={values.scale} min={0} max={5} step={0.01}
            displayFn={v => (v * 100).toFixed(0)} onChange={v => onChange({ ...values, scale: v })} accentColor="#a78bfa" />
          <MiniSlider label="Opacity" value={values.opacity} min={0} max={1} step={0.01}
            displayFn={v => (v * 100).toFixed(0)} onChange={v => onChange({ ...values, opacity: v })} accentColor="#34d399" />
          <MiniSlider label="Pos X" value={values.posX} min={-500} max={500} step={1}
            displayFn={v => v.toFixed(0)} onChange={v => onChange({ ...values, posX: v })} accentColor="#fb923c" />
          <MiniSlider label="Pos Y" value={values.posY} min={-500} max={500} step={1}
            displayFn={v => v.toFixed(0)} onChange={v => onChange({ ...values, posY: v })} accentColor="#f472b6" />
          <MiniSlider label="Speed" value={values.speed} min={0.1} max={4} step={0.05}
            displayFn={v => (v * 100).toFixed(0)} onChange={v => onChange({ ...values, speed: v })} accentColor="#fbbf24" />
          <button
            onClick={() => onChange(defaultKFPointValues())}
            className="text-[9px] text-zinc-600 hover:text-zinc-400 bg-zinc-800/60 hover:bg-zinc-800 border border-zinc-700/60 rounded px-2 py-0.5 transition-colors text-center mt-0.5"
          >
            Reset to Default
          </button>
        </div>
      )}

      {!open && (
        <div className="flex gap-2 text-[9px] text-zinc-600 font-mono px-1 flex-wrap">
          <span style={{ color: accentColor + 'cc' }}>Sc:{(values.scale * 100).toFixed(0)}%</span>
          <span style={{ color: accentColor + 'cc' }}>Op:{(values.opacity * 100).toFixed(0)}%</span>
          <span style={{ color: accentColor + 'cc' }}>X:{values.posX.toFixed(0)}</span>
          <span style={{ color: accentColor + 'cc' }}>Y:{values.posY.toFixed(0)}</span>
          <span style={{ color: accentColor + 'cc' }}>Sp:{(values.speed * 100).toFixed(0)}%</span>
        </div>
      )}
    </div>
  );
}

export default function AutomationPanel({ settings, onChange }: Props) {
  const ak = settings.autoKeyframes;
  const isOn = settings.enabled;

  const updateAK = (partial: Partial<typeof ak>) => {
    onChange({ autoKeyframes: { ...ak, ...partial } });
  };

  const isAKEnabled = ak.applyToImages || ak.applyToVideos;
  const rangeValid = ak.endTime > ak.startTime;

  return (
    <div className="flex flex-col h-full bg-zinc-900">
      <div className="px-3 py-2 border-b border-zinc-800 flex-shrink-0">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <svg className="h-3.5 w-3.5 text-violet-400" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 3H5a2 2 0 00-2 2v4m6-6h10a2 2 0 012 2v4M9 3v18m0 0h10a2 2 0 002-2V9M9 21H5a2 2 0 01-2-2V9m0 0h18" />
            </svg>
            <span className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider">Automation</span>
          </div>
          <button
            onClick={() => onChange({ enabled: !isOn })}
            className="flex items-center gap-1.5 group"
          >
            <span className={`text-[10px] font-semibold transition-colors ${isOn ? 'text-violet-300' : 'text-zinc-600'}`}>
              {isOn ? 'ON' : 'OFF'}
            </span>
            <div className={`relative w-9 h-5 rounded-full transition-colors duration-200 flex-shrink-0 ${isOn ? 'bg-violet-600' : 'bg-zinc-700'}`}>
              <div className={`absolute top-0.5 w-4 h-4 rounded-full bg-white shadow transition-transform duration-200 ${isOn ? 'translate-x-4' : 'translate-x-0.5'}`} />
            </div>
          </button>
        </div>
      </div>

      {!isOn && (
        <div className="mx-3 mt-3 bg-zinc-800/50 border border-zinc-700/50 rounded-lg px-3 py-3 flex flex-col items-center gap-2 text-center">
          <p className="text-[11px] text-zinc-500 font-medium">Automation is OFF</p>
          <p className="text-[10px] text-zinc-600 leading-relaxed">
            Turn ON to auto-apply duration &amp; keyframes when media is dropped on timeline
          </p>
          <button onClick={() => onChange({ enabled: true })}
            className="mt-1 text-[11px] px-3 py-1.5 rounded-lg bg-violet-600 hover:bg-violet-500 text-white font-semibold transition-colors">
            Turn ON
          </button>
        </div>
      )}

      <div className={`flex-1 overflow-y-auto p-3 flex flex-col gap-5 ${!isOn ? 'opacity-30 pointer-events-none' : ''}`}>
        {/* Image Duration */}
        <div className="flex flex-col gap-3">
          <SectionHeader
            icon={<svg className="h-3.5 w-3.5 text-amber-400" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24"><rect x="3" y="3" width="18" height="18" rx="2" /><circle cx="8.5" cy="8.5" r="1.5" /><path strokeLinecap="round" strokeLinejoin="round" d="M21 15l-5-5L5 21" /></svg>}
            title="Image Duration"
          />
          <TimeInput label="Duration" value={settings.imageDuration} min={0.1} max={3600 * 24}
            onChange={v => onChange({ imageDuration: v })} accentColor="#f59e0b" />
          <div className="flex flex-wrap gap-1">
            {[1, 2, 3, 5, 10, 15, 30, 60].map(sec => (
              <button key={sec} onClick={() => onChange({ imageDuration: sec })}
                className={`text-[9px] px-2 py-0.5 rounded border transition-colors ${
                  settings.imageDuration === sec
                    ? 'bg-amber-500/30 border-amber-500/60 text-amber-300'
                    : 'bg-zinc-800 border-zinc-700 text-zinc-500 hover:bg-zinc-700 hover:text-zinc-300'
                }`}>
                {sec >= 60 ? `${sec / 60}m` : `${sec}s`}
              </button>
            ))}
          </div>
        </div>

        <div className="h-px bg-zinc-800" />

        {/* Video Duration */}
        <div className="flex flex-col gap-3">
          <SectionHeader
            icon={<svg className="h-3.5 w-3.5 text-indigo-400" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24"><rect x="2" y="4" width="20" height="16" rx="2" /><path strokeLinecap="round" strokeLinejoin="round" d="M10 9l5 3-5 3V9z" /></svg>}
            title="Video Duration"
          />
          <ToggleCheck checked={settings.useCustomVideoDuration}
            onChange={v => onChange({ useCustomVideoDuration: v })}
            label="Use custom duration (override original)" color="#6366f1" />
          <div className={settings.useCustomVideoDuration ? '' : 'opacity-40 pointer-events-none'}>
            <TimeInput label="Custom Duration" value={settings.videoDuration} min={0.1} max={3600 * 24}
              onChange={v => onChange({ videoDuration: v })} accentColor="#6366f1" />
          </div>
        </div>

        <div className="h-px bg-zinc-800" />

        {/* Auto Keyframes */}
        <div className="flex flex-col gap-3">
          <SectionHeader
            icon={<svg className="h-3.5 w-3.5 text-violet-400" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M15 15l-2 5L9 9l11 4-5 2zm0 0l5 5" /></svg>}
            title="Auto Keyframes"
          />
          <div className="flex flex-col gap-2 bg-zinc-800/40 rounded-lg p-2.5 border border-zinc-700/50">
            <p className="text-[9px] text-zinc-500 font-semibold uppercase tracking-wider mb-1">Apply to</p>
            <ToggleCheck checked={ak.applyToImages} onChange={v => updateAK({ applyToImages: v })}
              label="Images (jpg, png, webp, gif…)" color="#f59e0b" />
            <ToggleCheck checked={ak.applyToVideos} onChange={v => updateAK({ applyToVideos: v })}
              label="Videos (mp4, mov, webm…)" color="#6366f1" />
          </div>

          <div className={`flex flex-col gap-3 transition-opacity ${isAKEnabled ? '' : 'opacity-40 pointer-events-none'}`}>
            <div className="flex items-center justify-between">
              <p className="text-[10px] text-zinc-400 font-semibold">Keyframe Time Range</p>
              {rangeValid && isAKEnabled && (
                <span className="text-[9px] text-violet-300 font-mono bg-violet-500/10 border border-violet-500/20 px-1.5 py-0.5 rounded">
                  {formatDisplay(ak.startTime)} → {formatDisplay(ak.endTime)}
                </span>
              )}
            </div>

            <div className="flex flex-col gap-2 bg-zinc-800/30 rounded-lg p-2.5 border border-zinc-700/40">
              <TimeInput label="▶ Start" value={ak.startTime} min={0} max={3600}
                onChange={v => updateAK({ startTime: v })} accentColor="#a78bfa" />
              <KFPointEditor label="Start Keyframe Values" accentColor="#a78bfa"
                values={ak.startValues} onChange={v => updateAK({ startValues: v })} />
            </div>

            <div className="flex flex-col gap-2 bg-zinc-800/30 rounded-lg p-2.5 border border-zinc-700/40">
              <TimeInput label="⏹ End" value={ak.endTime} min={0} max={3600}
                onChange={v => updateAK({ endTime: v })} accentColor="#7c3aed" />
              <KFPointEditor label="End Keyframe Values" accentColor="#7c3aed"
                values={ak.endValues} onChange={v => updateAK({ endValues: v })} />
            </div>
          </div>
        </div>

        <div className="h-4" />
      </div>
    </div>
  );
}
