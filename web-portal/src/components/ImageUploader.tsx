'use client';

import { useCallback, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import { Upload, ImageIcon, Loader2, Sparkles } from 'lucide-react';

interface ImageUploaderProps {
  onImageSelect: (file: File) => void;
  isProcessing: boolean;
}

const MAX_SIZE = 10 * 1024 * 1024; // 10 MB
const ACCEPTED = ['image/jpeg', 'image/png', 'image/webp'];

export default function ImageUploader({ onImageSelect, isProcessing }: ImageUploaderProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(
    (file: File) => {
      setError(null);

      if (!ACCEPTED.includes(file.type)) {
        setError('Please upload a JPEG, PNG, or WebP image.');
        return;
      }
      if (file.size > MAX_SIZE) {
        setError('File too large — maximum 10 MB.');
        return;
      }

      setFileName(file.name);
      const url = URL.createObjectURL(file);
      setPreview(url);
      onImageSelect(file);
    },
    [onImageSelect],
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile],
  );

  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const onDragLeave = useCallback(() => setIsDragging(false), []);

  const onInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) handleFile(file);
    },
    [handleFile],
  );

  const handleTrySample = useCallback(async () => {
    setError(null);
    try {
      const res = await fetch('/samples/sample_cityscapes.jpg');
      if (!res.ok) {
        setError('Sample image not available. Upload your own image.');
        return;
      }
      const blob = await res.blob();
      const file = new File([blob], 'sample_cityscapes.jpg', { type: 'image/jpeg' });
      handleFile(file);
    } catch {
      setError('Failed to load sample image.');
    }
  }, [handleFile]);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: 'easeOut' }}
      className="w-full"
    >
      <div
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onClick={() => inputRef.current?.click()}
        className={`
          relative group cursor-pointer rounded-2xl border-2 border-dashed
          transition-all duration-300 overflow-hidden
          ${isDragging
            ? 'border-purple-400 bg-purple-500/10 scale-[1.01]'
            : 'border-white/20 bg-white/5 hover:border-purple-400/60 hover:bg-white/[0.07]'
          }
          ${isProcessing ? 'pointer-events-none opacity-60' : ''}
        `}
      >
        {/* Animated gradient border glow */}
        <div className={`
          absolute inset-0 rounded-2xl opacity-0 transition-opacity duration-300
          ${isDragging ? 'opacity-100' : 'group-hover:opacity-50'}
          bg-gradient-to-r from-purple-500/20 via-blue-500/20 to-purple-500/20
        `} />

        <div className="relative z-10 flex flex-col items-center justify-center p-10 md:p-16 gap-4">
          {isProcessing ? (
            <>
              <Loader2 className="w-12 h-12 text-purple-400 animate-spin" />
              <p className="text-lg text-white/80 font-medium">Processing image…</p>
              <div className="w-64 h-1.5 bg-white/10 rounded-full overflow-hidden">
                <motion.div
                  className="h-full bg-gradient-to-r from-purple-500 to-blue-500 rounded-full"
                  initial={{ width: '0%' }}
                  animate={{ width: '100%' }}
                  transition={{ duration: 8, ease: 'linear' }}
                />
              </div>
            </>
          ) : preview ? (
            <div className="flex flex-col items-center gap-4">
              <div className="relative w-48 h-28 rounded-lg overflow-hidden ring-2 ring-purple-400/50">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={preview}
                  alt="Preview"
                  className="w-full h-full object-cover"
                />
              </div>
              <p className="text-sm text-white/60 font-mono">{fileName}</p>
              <p className="text-xs text-white/40">Click or drag to replace</p>
            </div>
          ) : (
            <>
              <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-purple-500/20 to-blue-500/20 flex items-center justify-center border border-white/10">
                <Upload className="w-8 h-8 text-purple-400" />
              </div>
              <div className="text-center">
                <p className="text-lg text-white/90 font-medium">
                  Drop a street-scene image here
                </p>
                <p className="text-sm text-white/50 mt-1">
                  or click to browse — JPEG, PNG, WebP up to 10 MB
                </p>
              </div>
            </>
          )}
        </div>

        <input
          ref={inputRef}
          type="file"
          accept="image/jpeg,image/png,image/webp"
          onChange={onInputChange}
          className="hidden"
        />
      </div>

      {/* Error message */}
      {error && (
        <motion.p
          initial={{ opacity: 0, y: -5 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-3 text-sm text-red-400 text-center"
        >
          {error}
        </motion.p>
      )}

      {/* Try Sample button */}
      {!preview && !isProcessing && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3 }}
          className="mt-4 flex justify-center"
        >
          <button
            onClick={(e) => {
              e.stopPropagation();
              handleTrySample();
            }}
            className="
              flex items-center gap-2 px-5 py-2.5 rounded-xl
              bg-gradient-to-r from-purple-500/20 to-blue-500/20
              border border-white/10 text-sm text-white/80
              hover:border-purple-400/50 hover:text-white
              transition-all duration-200
            "
          >
            <Sparkles className="w-4 h-4" />
            Try Sample Image
          </button>
        </motion.div>
      )}
    </motion.div>
  );
}
