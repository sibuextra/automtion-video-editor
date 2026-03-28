import { MediaFile, Track } from '../types/editor';
import { isMediaAllowedOnTrack } from '../hooks/useEditorStore';

interface Props {
  mediaFiles: MediaFile[];
  tracks: Track[];
  onRemove: (id: string) => void;
  onAddToTrack: (media: MediaFile, trackIndex: number) => void;
}

function formatDuration(sec?: number) {
  if (!sec) return '--:--';
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${String(s).padStart(2, '0')}`;
}

const typeIcon = {
  video: (
    <svg className="h-4 w-4 text-indigo-400" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
      <rect x="2" y="4" width="20" height="16" rx="2" />
      <path strokeLinecap="round" strokeLinejoin="round" d="M10 9l5 3-5 3V9z" />
    </svg>
  ),
  audio: (
    <svg className="h-4 w-4 text-emerald-400" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" d="M9 18V5l12-2v13" />
      <circle cx="6" cy="18" r="3" />
      <circle cx="18" cy="16" r="3" />
    </svg>
  ),
  image: (
    <svg className="h-4 w-4 text-amber-400" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
      <rect x="3" y="3" width="18" height="18" rx="2" />
      <circle cx="8.5" cy="8.5" r="1.5" />
      <path strokeLinecap="round" strokeLinejoin="round" d="M21 15l-5-5L5 21" />
    </svg>
  ),
};

export default function MediaPanel({ mediaFiles, tracks, onRemove, onAddToTrack }: Props) {
  if (mediaFiles.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-3 text-zinc-500 p-6 text-center">
        <svg className="h-10 w-10 text-zinc-600" fill="none" stroke="currentColor" strokeWidth={1.5} viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
        </svg>
        <p className="text-sm">Import media to get started</p>
        <p className="text-xs text-zinc-600">Video, Audio & Images supported</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-1 p-2 overflow-y-auto h-full">
      {mediaFiles.map(media => {
        const compatibleTracks = tracks
          .map((t, i) => ({ track: t, index: i }))
          .filter(({ track }) => isMediaAllowedOnTrack(media.type, track.type));

        return (
          <div
            key={media.id}
            className="group flex items-center gap-2 rounded-lg bg-zinc-800 hover:bg-zinc-700 p-2 cursor-grab active:cursor-grabbing transition-colors"
            draggable
            onDragStart={e => {
              e.dataTransfer.setData('mediaId', media.id);
            }}
          >
            {/* Thumbnail / Icon */}
            <div className="w-14 h-9 rounded bg-zinc-900 flex items-center justify-center flex-shrink-0 overflow-hidden">
              {media.thumbnail ? (
                <img src={media.thumbnail} alt="" className="w-full h-full object-cover" />
              ) : (
                <div className="scale-150">{typeIcon[media.type]}</div>
              )}
            </div>

            {/* Info */}
            <div className="flex-1 min-w-0">
              <p className="text-xs text-zinc-200 truncate font-medium">{media.name}</p>
              <div className="flex items-center gap-1 mt-0.5">
                {typeIcon[media.type]}
                <span className="text-[10px] text-zinc-500 uppercase">{media.type}</span>
                {media.duration && (
                  <span className="text-[10px] text-zinc-500 ml-1">{formatDuration(media.duration)}</span>
                )}
              </div>
            </div>

            {/* Actions */}
            <div className="flex flex-col gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
              <select
                className="text-[10px] bg-zinc-700 text-zinc-200 rounded px-1 py-0.5 border border-zinc-600 cursor-pointer"
                defaultValue=""
                onChange={e => {
                  const idx = parseInt(e.target.value);
                  if (!isNaN(idx)) {
                    onAddToTrack(media, idx);
                    e.target.value = '';
                  }
                }}
              >
                <option value="" disabled>+ Track</option>
                {compatibleTracks.map(({ track, index }) => (
                  <option key={track.id} value={index}>{track.label}</option>
                ))}
              </select>
              <button
                onClick={() => onRemove(media.id)}
                className="text-[10px] text-red-400 hover:text-red-300 text-center"
              >
                Remove
              </button>
            </div>
          </div>
        );
      })}
    </div>
  );
}
