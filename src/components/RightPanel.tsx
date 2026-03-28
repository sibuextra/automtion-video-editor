import { useState } from 'react';
import TransformPanel from './TransformPanel';
import AutomationPanel from './AutomationPanel';
import { TimelineClip, ClipTransform, AutomationSettings } from '../types/editor';

type Tab = 'transform' | 'automation';

interface Props {
  selectedClips: TimelineClip[];
  currentTime: number;
  onUpdateTransform: (clipIds: string[], partial: Partial<ClipTransform>) => void;
  onUpdateSpeed: (clipIds: string[], speed: number) => void;
  onAddKeyframe: (clipIds: string[], prop: 'scale' | 'opacity' | 'posX' | 'posY', time: number, value?: number) => void;
  onRemoveKeyframe: (clipId: string, prop: 'scale' | 'opacity' | 'posX' | 'posY', kfId: string) => void;
  onMoveKeyframe: (clipId: string, prop: 'scale' | 'opacity' | 'posX' | 'posY', kfId: string, newTime: number) => void;
  automationSettings: AutomationSettings;
  onAutomationChange: (partial: Partial<AutomationSettings>) => void;
}

export default function RightPanel({
  selectedClips, currentTime, onUpdateTransform, onUpdateSpeed,
  onAddKeyframe, onRemoveKeyframe, onMoveKeyframe,
  automationSettings, onAutomationChange,
}: Props) {
  const [activeTab, setActiveTab] = useState<Tab>('transform');

  const autoKFActive = automationSettings.autoKeyframes.applyToImages || automationSettings.autoKeyframes.applyToVideos;
  const videoOverride = automationSettings.useCustomVideoDuration;

  return (
    <div className="flex flex-col h-full bg-zinc-900 border-l border-zinc-800">
      <div className="flex flex-shrink-0 border-b border-zinc-800 bg-zinc-900">
        <button
          onClick={() => setActiveTab('transform')}
          className={`flex-1 flex items-center justify-center gap-1.5 py-2 px-2 text-[11px] font-semibold transition-colors border-b-2 ${
            activeTab === 'transform'
              ? 'border-indigo-500 text-indigo-400 bg-indigo-500/5'
              : 'border-transparent text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800/50'
          }`}
        >
          <svg className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
            <circle cx="12" cy="12" r="3" /><path strokeLinecap="round" strokeLinejoin="round" d="M12 1v4M12 19v4M4.22 4.22l2.83 2.83M16.95 16.95l2.83 2.83M1 12h4M19 12h4M4.22 19.78l2.83-2.83M16.95 7.05l2.83-2.83" />
          </svg>
          <span>Transform</span>
        </button>

        <button
          onClick={() => setActiveTab('automation')}
          className={`flex-1 flex items-center justify-center gap-1.5 py-2 px-2 text-[11px] font-semibold transition-colors border-b-2 relative ${
            activeTab === 'automation'
              ? 'border-violet-500 text-violet-400 bg-violet-500/5'
              : 'border-transparent text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800/50'
          }`}
        >
          <svg className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M9 3H5a2 2 0 00-2 2v4m6-6h10a2 2 0 012 2v4M9 3v18m0 0h10a2 2 0 002-2V9M9 21H5a2 2 0 01-2-2V9m0 0h18" />
          </svg>
          <span>Automation</span>
          {(autoKFActive || videoOverride) && (
            <div className={`absolute top-1.5 right-2 w-1.5 h-1.5 rounded-full ${activeTab === 'automation' ? 'bg-violet-400' : 'bg-violet-500'}`} />
          )}
        </button>
      </div>

      <div className="flex-1 overflow-hidden min-h-0">
        {activeTab === 'transform' ? (
          <TransformPanel
            selectedClips={selectedClips} currentTime={currentTime}
            onUpdateTransform={onUpdateTransform} onUpdateSpeed={onUpdateSpeed}
            onAddKeyframe={onAddKeyframe} onRemoveKeyframe={onRemoveKeyframe}
            onMoveKeyframe={onMoveKeyframe}
          />
        ) : (
          <AutomationPanel settings={automationSettings} onChange={onAutomationChange} />
        )}
      </div>
    </div>
  );
}
