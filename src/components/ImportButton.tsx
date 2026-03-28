import React, { useRef, useState } from 'react';
import { MediaFile } from '../types/editor';

interface Props {
  onImport: (file: MediaFile) => void;
}

function generateId() {
  return Math.random().toString(36).slice(2, 10);
}

function getMediaType(file: File): 'video' | 'audio' | 'image' | null {
  const mime = file.type.toLowerCase();
  if (mime.startsWith('video/')) return 'video';
  if (mime.startsWith('audio/')) return 'audio';
  if (mime.startsWith('image/')) return 'image';

  const ext = file.name.split('.').pop()?.toLowerCase() ?? '';
  const videoExts = ['mp4', 'mov', 'avi', 'mkv', 'webm', 'flv', 'wmv', 'm4v', 'ogv', '3gp'];
  const audioExts = ['mp3', 'wav', 'aac', 'ogg', 'flac', 'm4a', 'wma', 'opus', 'aiff'];
  const imageExts = ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp', 'svg', 'tiff', 'ico', 'avif'];
  if (videoExts.includes(ext)) return 'video';
  if (audioExts.includes(ext)) return 'audio';
  if (imageExts.includes(ext)) return 'image';
  return null;
}

function getMediaDuration(url: string, type: 'video' | 'audio'): Promise<number> {
  return new Promise(resolve => {
    const el = document.createElement(type === 'video' ? 'video' : 'audio');
    el.src = url;
    el.onloadedmetadata = () => resolve(el.duration || 5);
    el.onerror = () => resolve(5);
  });
}

function getVideoThumbnail(url: string): Promise<string> {
  return new Promise(resolve => {
    const video = document.createElement('video');
    video.src = url;
    video.currentTime = 0.5;
    video.onloadeddata = () => {
      const canvas = document.createElement('canvas');
      canvas.width = 120;
      canvas.height = 68;
      const ctx = canvas.getContext('2d');
      ctx?.drawImage(video, 0, 0, 120, 68);
      resolve(canvas.toDataURL('image/jpeg', 0.7));
    };
    video.onerror = () => resolve('');
  });
}

async function generateWaveformData(file: File, samples = 300): Promise<number[]> {
  try {
    const arrayBuffer = await file.arrayBuffer();
    const audioCtx = new OfflineAudioContext(1, 1, 44100);
    const decoded = await audioCtx.decodeAudioData(arrayBuffer);
    const rawData = decoded.getChannelData(0);
    const blockSize = Math.floor(rawData.length / samples);
    const waveform: number[] = [];
    for (let i = 0; i < samples; i++) {
      let sum = 0;
      const start = i * blockSize;
      for (let j = 0; j < blockSize; j++) {
        sum += Math.abs(rawData[start + j] || 0);
      }
      waveform.push(sum / blockSize);
    }
    const max = Math.max(...waveform, 0.001);
    return waveform.map(v => v / max);
  } catch {
    return Array.from({ length: samples }, () => Math.random() * 0.8 + 0.1);
  }
}

export default function ImportButton({ onImport }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [loading, setLoading] = useState(false);

  const handleFiles = async (files: FileList | null) => {
    if (!files) return;
    setLoading(true);
    for (const file of Array.from(files)) {
      const type = getMediaType(file);
      if (!type) continue;
      const url = URL.createObjectURL(file);
      let duration: number | undefined;
      let thumbnail: string | undefined;
      let waveformData: number[] | undefined;

      if (type === 'video') {
        duration = await getMediaDuration(url, 'video');
        thumbnail = await getVideoThumbnail(url);
      } else if (type === 'audio') {
        duration = await getMediaDuration(url, 'audio');
        waveformData = await generateWaveformData(file);
      } else {
        duration = 5;
        thumbnail = url;
      }

      const mediaFile: MediaFile = {
        id: generateId(),
        name: file.name,
        type,
        url,
        duration,
        thumbnail,
        waveformData,
        file,
      };
      onImport(mediaFile);
    }
    setLoading(false);
    if (inputRef.current) inputRef.current.value = '';
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    handleFiles(e.dataTransfer.files);
  };

  return (
    <div
      className="flex items-center gap-2"
      onDrop={handleDrop}
      onDragOver={e => e.preventDefault()}
    >
      <input
        ref={inputRef}
        type="file"
        multiple
        accept="video/*,audio/*,image/*,.mp4,.mov,.avi,.mkv,.webm,.flv,.wmv,.m4v,.mp3,.wav,.aac,.ogg,.flac,.m4a,.wma,.opus,.png,.jpg,.jpeg,.gif,.bmp,.webp,.svg,.tiff"
        className="hidden"
        onChange={e => handleFiles(e.target.files)}
      />
      <button
        onClick={() => inputRef.current?.click()}
        disabled={loading}
        className="flex items-center gap-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 active:bg-indigo-700 text-white text-sm font-medium px-4 py-2 transition-colors duration-150 disabled:opacity-60 disabled:cursor-not-allowed shadow-sm"
      >
        {loading ? (
          <>
            <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"/>
            </svg>
            Importing...
          </>
        ) : (
          <>
            <svg className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2M12 4v12m-4-4l4 4 4-4" />
            </svg>
            Import Media
          </>
        )}
      </button>
    </div>
  );
}
