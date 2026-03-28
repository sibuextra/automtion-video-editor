import { useCallback, useState, useRef } from 'react';
import { useEditorStore } from './hooks/useEditorStore';
import ImportButton from './components/ImportButton';
import MediaPanel from './components/MediaPanel';
import PreviewPanel from './components/PreviewPanel';
import Timeline from './components/Timeline';
import RightPanel from './components/RightPanel';
import ExportModal from './components/ExportModal';
import { MediaFile, ClipTransform } from './types/editor';

const MEDIA_MIN = 140;
const MEDIA_MAX = 400;
const TRANSFORM_MIN = 180;
const TRANSFORM_MAX = 480;
const TIMELINE_MIN = 140;
const TIMELINE_MAX = 520;

export default function App() {
  const store = useEditorStore();

  const {
    mediaFiles,
    tracks,
    currentTime,
    isPlaying,
    duration,
    selectedClipIds,
    zoom,
    aspectRatio,
    automationSettings,
    addMediaFile,
    removeMediaFile,
    addClipToTrack,
    moveClip,
    removeClip,
    splitSelectedClipsAtPlayhead,
    addTrack,
    removeTrack,
    selectClip,
    updateTransform,
    updateClipSpeed,
    addKeyframe,
    removeKeyframe,
    moveKeyframeWithHistory,
    updateAutomationSettings,
    play,
    pause,
    seek,
    setAspectRatio,
    getActiveClipsAt,
    setZoom,
  } = store;

  const [mediaPanelW, setMediaPanelW]         = useState(192);
  const [transformPanelW, setTransformPanelW] = useState(224);
  const [timelineH, setTimelineH]             = useState(240);
  const [snapEnabled, setSnapEnabled]         = useState(true);
  const [exportOpen, setExportOpen]           = useState(false);

  const resizingRef    = useRef<null | 'media' | 'transform' | 'timeline'>(null);
  const resizeStartRef = useRef<{ x: number; y: number; startVal: number }>({ x: 0, y: 0, startVal: 0 });

  const startResize = useCallback((which: 'media' | 'transform' | 'timeline', e: React.MouseEvent) => {
    e.preventDefault();
    resizingRef.current = which;
    resizeStartRef.current = {
      x: e.clientX, y: e.clientY,
      startVal: which === 'media' ? mediaPanelW : which === 'transform' ? transformPanelW : timelineH,
    };

    const onMove = (ev: MouseEvent) => {
      const { x, y, startVal } = resizeStartRef.current;
      if (resizingRef.current === 'media') {
        setMediaPanelW(Math.max(MEDIA_MIN, Math.min(MEDIA_MAX, startVal + ev.clientX - x)));
      } else if (resizingRef.current === 'transform') {
        setTransformPanelW(Math.max(TRANSFORM_MIN, Math.min(TRANSFORM_MAX, startVal - (ev.clientX - x))));
      } else if (resizingRef.current === 'timeline') {
        setTimelineH(Math.max(TIMELINE_MIN, Math.min(TIMELINE_MAX, startVal - (ev.clientY - y))));
      }
    };
    const onUp = () => {
      resizingRef.current = null;
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
  }, [mediaPanelW, transformPanelW, timelineH]);

  const handleImport = useCallback((file: MediaFile) => { addMediaFile(file); }, [addMediaFile]);

  const activeClips = getActiveClipsAt(currentTime);
  const selectedClipObjects = tracks.flatMap(t => t.clips).filter(c => selectedClipIds.includes(c.id));

  const handleUpdateTransform = useCallback(
    (clipIds: string[], partial: Partial<ClipTransform>) => { updateTransform(clipIds, partial); },
    [updateTransform]
  );

  const handleAddKeyframe = useCallback(
    (clipIds: string[], prop: 'scale' | 'opacity' | 'posX' | 'posY', time: number, value?: number) => {
      addKeyframe(clipIds, prop, time, value);
    },
    [addKeyframe]
  );

  const handleRemoveKeyframe = useCallback(
    (clipId: string, prop: 'scale' | 'opacity' | 'posX' | 'posY', kfId: string) => {
      removeKeyframe(clipId, prop, kfId);
    },
    [removeKeyframe]
  );

  const handleMoveKeyframe = useCallback(
    (clipId: string, prop: 'scale' | 'opacity' | 'posX' | 'posY', kfId: string, newTime: number) => {
      moveKeyframeWithHistory(clipId, prop, kfId, newTime);
    },
    [moveKeyframeWithHistory]
  );

  const handleUpdateSpeed = useCallback(
    (clipIds: string[], speed: number) => { updateClipSpeed(clipIds, speed); },
    [updateClipSpeed]
  );

  return (
    <div className="flex flex-col h-screen bg-zinc-950 text-zinc-100 overflow-hidden font-sans select-none">

      {/* ── Top Bar ── */}
      <header className="flex items-center gap-3 px-4 py-2 bg-zinc-900 border-b border-zinc-800 flex-shrink-0 z-30">
        <div className="flex items-center gap-2 mr-2">
          <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-indigo-500 to-violet-600 flex items-center justify-center shadow">
            <svg className="h-4 w-4 text-white" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M15 10l4.553-2.277A1 1 0 0121 8.677v6.646a1 1 0 01-1.447.894L15 14M3 8a2 2 0 012-2h8a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2V8z" />
            </svg>
          </div>
          <span className="text-sm font-semibold text-white tracking-tight">VideoForge</span>
        </div>

        <ImportButton onImport={handleImport} />

        <div className="flex-1" />

        {mediaFiles.length > 0 && (
          <div className="flex items-center gap-1.5 text-[11px] text-zinc-500">
            <span className="bg-zinc-800 rounded px-1.5 py-0.5">{mediaFiles.filter(f => f.type === 'video').length} video</span>
            <span className="bg-zinc-800 rounded px-1.5 py-0.5">{mediaFiles.filter(f => f.type === 'audio').length} audio</span>
            <span className="bg-zinc-800 rounded px-1.5 py-0.5">{mediaFiles.filter(f => f.type === 'image').length} image</span>
          </div>
        )}

        <button
          onClick={() => setExportOpen(true)}
          className="flex items-center gap-1.5 text-xs bg-indigo-600 hover:bg-indigo-500 active:bg-indigo-700 text-white px-3 py-1.5 rounded-lg transition-colors border border-indigo-500 shadow-sm font-medium"
        >
          <svg className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2M8 12l4 4 4-4M12 4v12" />
          </svg>
          Export
        </button>
      </header>

      {/* ── Main Area ── */}
      <div className="flex flex-1 overflow-hidden min-h-0">

        {/* Left: Media Panel */}
        <aside className="flex-shrink-0 border-r border-zinc-800 bg-zinc-900 flex flex-col relative" style={{ width: mediaPanelW }}>
          <div className="px-3 py-2 border-b border-zinc-800 flex-shrink-0">
            <span className="text-[11px] font-semibold text-zinc-500 uppercase tracking-wider">Media</span>
          </div>
          <div className="flex-1 overflow-hidden">
            <MediaPanel mediaFiles={mediaFiles} tracks={tracks} onRemove={removeMediaFile} onAddToTrack={addClipToTrack} />
          </div>
          <div className="absolute top-0 right-0 bottom-0 w-1 cursor-col-resize hover:bg-indigo-500/40 transition-colors z-20 group"
            onMouseDown={e => startResize('media', e)}>
            <div className="absolute inset-y-0 right-0 w-0.5 bg-zinc-700 group-hover:bg-indigo-400 transition-colors" />
          </div>
        </aside>

        {/* Center: Preview */}
        <main className="flex-1 flex flex-col overflow-hidden min-w-0 relative">
          <div className="flex-1 overflow-hidden min-h-0">
            <PreviewPanel
              currentTime={currentTime} isPlaying={isPlaying} duration={duration}
              activeClips={activeClips} selectedClips={selectedClipObjects}
              aspectRatio={aspectRatio}
              onPlay={play} onPause={pause} onSeek={seek} onAspectRatioChange={setAspectRatio}
            />
          </div>
          <div className="absolute bottom-0 left-0 right-0 h-1 cursor-row-resize hover:bg-indigo-500/40 transition-colors z-20 group"
            onMouseDown={e => startResize('timeline', e)}>
            <div className="absolute inset-x-0 bottom-0 h-0.5 bg-zinc-700 group-hover:bg-indigo-400 transition-colors" />
          </div>
        </main>

        {/* Right: Transform + Automation */}
        <aside className="flex-shrink-0 relative" style={{ width: transformPanelW }}>
          <div className="absolute top-0 left-0 bottom-0 w-1 cursor-col-resize hover:bg-indigo-500/40 transition-colors z-20 group"
            onMouseDown={e => startResize('transform', e)}>
            <div className="absolute inset-y-0 left-0 w-0.5 bg-zinc-700 group-hover:bg-indigo-400 transition-colors" />
          </div>
          <RightPanel
            selectedClips={selectedClipObjects} currentTime={currentTime}
            onUpdateTransform={handleUpdateTransform} onUpdateSpeed={handleUpdateSpeed}
            onAddKeyframe={handleAddKeyframe} onRemoveKeyframe={handleRemoveKeyframe}
            onMoveKeyframe={handleMoveKeyframe}
            automationSettings={automationSettings} onAutomationChange={updateAutomationSettings}
          />
        </aside>
      </div>

      {/* Bottom: Timeline */}
      <div className="flex-shrink-0 border-t border-zinc-800 relative" style={{ height: timelineH }}>
        <div className="absolute top-0 left-0 right-0 h-1 cursor-row-resize hover:bg-indigo-500/40 transition-colors z-20 group"
          onMouseDown={e => startResize('timeline', e)}>
          <div className="absolute inset-x-0 top-0 h-0.5 bg-zinc-700 group-hover:bg-indigo-400 transition-colors" />
        </div>
        <Timeline
          tracks={tracks} currentTime={currentTime} duration={duration}
          zoom={zoom} selectedClipIds={selectedClipIds} mediaFiles={mediaFiles}
          isPlaying={isPlaying} snapEnabled={snapEnabled}
          onSnapToggle={() => setSnapEnabled(v => !v)}
          onSeek={seek} onSelectClip={selectClip}
          onMoveClip={moveClip} onRemoveClip={removeClip}
          onSplitAtPlayhead={splitSelectedClipsAtPlayhead}
          onAddClipToTrack={addClipToTrack} onZoomChange={setZoom}
          onAddTrack={addTrack} onRemoveTrack={removeTrack}
          onMoveKeyframe={handleMoveKeyframe}
          onPlay={play} onPause={pause}
        />
      </div>

      {/* Export Modal */}
      <ExportModal
        isOpen={exportOpen}
        onClose={() => setExportOpen(false)}
        tracks={tracks}
        duration={duration}
        currentAspectRatio={aspectRatio}
        projectName="VideoForge Export"
      />
    </div>
  );
}
