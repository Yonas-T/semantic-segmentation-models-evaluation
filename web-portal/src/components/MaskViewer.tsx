'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import { Eye, Layers, Clock } from 'lucide-react';
import { CITYSCAPES_CLASSES, CITYSCAPES_COLORS, colorToRgbString, classIdToColor } from '@/lib/cityscapesPalette';

interface MaskViewerProps {
  originalImage: string | null;
  segformerMask: string | null;
  deeplabMask: string | null;
  segformerLatency: number;
  deeplabLatency: number;
}

type ViewMode = 'mask' | 'overlay';

export default function MaskViewer({
  originalImage,
  segformerMask,
  deeplabMask,
  segformerLatency,
  deeplabLatency,
}: MaskViewerProps) {
  const [viewMode, setViewMode] = useState<ViewMode>('mask');
  const [opacity, setOpacity] = useState(0.6);
  const [hoveredClass, setHoveredClass] = useState<string | null>(null);
  const [showLegend, setShowLegend] = useState(false);

  const panels = [
    { title: 'Original Image', src: originalImage, latency: null, model: null },
    { title: 'SegFormer-B2', src: segformerMask, latency: segformerLatency, model: 'segformer' },
    { title: 'DeepLabV3+', src: deeplabMask, latency: deeplabLatency, model: 'deeplabv3' },
  ];

  if (!originalImage && !segformerMask && !deeplabMask) {
    return (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="glass-card p-12 text-center"
      >
        <Layers className="w-12 h-12 text-white/20 mx-auto mb-4" />
        <p className="text-white/40 text-lg">
          Upload an image to see segmentation results
        </p>
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      {/* Controls bar */}
      <div className="flex flex-wrap items-center justify-between gap-4 mb-6">
        <div className="flex items-center gap-3">
          {/* View mode toggle */}
          <div className="flex bg-white/5 rounded-xl border border-white/10 p-1">
            <button
              onClick={() => setViewMode('mask')}
              className={`px-4 py-1.5 rounded-lg text-sm font-medium transition-all ${
                viewMode === 'mask'
                  ? 'bg-purple-500/30 text-purple-300 border border-purple-400/30'
                  : 'text-white/50 hover:text-white/80'
              }`}
            >
              <Eye className="w-4 h-4 inline mr-1.5" />
              Mask Only
            </button>
            <button
              onClick={() => setViewMode('overlay')}
              className={`px-4 py-1.5 rounded-lg text-sm font-medium transition-all ${
                viewMode === 'overlay'
                  ? 'bg-purple-500/30 text-purple-300 border border-purple-400/30'
                  : 'text-white/50 hover:text-white/80'
              }`}
            >
              <Layers className="w-4 h-4 inline mr-1.5" />
              Overlay
            </button>
          </div>

          {/* Opacity slider (overlay mode) */}
          {viewMode === 'overlay' && (
            <div className="flex items-center gap-2 text-sm text-white/50">
              <span>Opacity</span>
              <input
                type="range"
                min="0.2"
                max="0.9"
                step="0.05"
                value={opacity}
                onChange={(e) => setOpacity(parseFloat(e.target.value))}
                className="w-24 accent-purple-400"
              />
              <span className="text-white/70 font-mono w-8">{Math.round(opacity * 100)}%</span>
            </div>
          )}
        </div>

        {/* Legend toggle */}
        <button
          onClick={() => setShowLegend(!showLegend)}
          className="px-3 py-1.5 rounded-lg text-sm text-white/50 border border-white/10 hover:border-white/20 hover:text-white/70 transition-all"
        >
          {showLegend ? 'Hide' : 'Show'} Legend
        </button>
      </div>

      {/* Legend */}
      {showLegend && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          exit={{ opacity: 0, height: 0 }}
          className="mb-6 glass-card p-4"
        >
          <div className="grid grid-cols-4 sm:grid-cols-5 md:grid-cols-7 lg:grid-cols-10 gap-2">
            {CITYSCAPES_CLASSES.map((cls, i) => (
              <div
                key={cls}
                className="flex items-center gap-1.5 text-xs text-white/70"
              >
                <div
                  className="w-3 h-3 rounded-sm flex-shrink-0"
                  style={{ backgroundColor: colorToRgbString(CITYSCAPES_COLORS[i]) }}
                />
                <span className="truncate">{cls}</span>
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Panels */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {panels.map((panel, idx) => (
          <motion.div
            key={panel.title}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: idx * 0.1 }}
            className="glass-card overflow-hidden"
          >
            {/* Panel header */}
            <div className="flex items-center justify-between px-4 py-3 border-b border-white/10">
              <h3 className="text-sm font-semibold text-white/90">{panel.title}</h3>
              {panel.latency !== null && panel.latency > 0 && (
                <span className="flex items-center gap-1 text-xs text-white/50 font-mono">
                  <Clock className="w-3 h-3" />
                  {panel.latency}ms
                </span>
              )}
            </div>

            {/* Image display */}
            <div className="relative aspect-[2/1] bg-black/40">
              {panel.src ? (
                viewMode === 'overlay' && idx > 0 && originalImage ? (
                  <OverlayPanel
                    original={originalImage}
                    mask={panel.src}
                    opacity={opacity}
                    onHoverClass={setHoveredClass}
                  />
                ) : (
                  <ImagePanel
                    src={panel.src}
                    alt={panel.title}
                    isMask={idx > 0}
                    onHoverClass={idx > 0 ? setHoveredClass : undefined}
                  />
                )
              ) : (
                <div className="w-full h-full flex items-center justify-center">
                  <p className="text-white/20 text-sm">Waiting for result…</p>
                </div>
              )}

              {/* Hover tooltip */}
              {hoveredClass && idx > 0 && (
                <div className="absolute bottom-2 left-2 bg-black/80 backdrop-blur-sm rounded-lg px-3 py-1.5 text-xs text-white/90 border border-white/10">
                  {hoveredClass}
                </div>
              )}
            </div>
          </motion.div>
        ))}
      </div>
    </motion.div>
  );
}

// ─── Sub-components ─────────────────────────────────────────────────────────

function ImagePanel({
  src,
  alt,
  isMask,
  onHoverClass,
}: {
  src: string;
  alt: string;
  isMask: boolean;
  onHoverClass?: (cls: string | null) => void;
}) {
  const imgSrc = isMask && !src.startsWith('data:') && !src.startsWith('blob:')
    ? `data:image/png;base64,${src}`
    : src;

  return (
    // eslint-disable-next-line @next/next/no-img-element
    <img
      src={imgSrc}
      alt={alt}
      className="w-full h-full object-contain"
      onMouseLeave={() => onHoverClass?.(null)}
    />
  );
}

function OverlayPanel({
  original,
  mask,
  opacity,
  onHoverClass,
}: {
  original: string;
  mask: string;
  opacity: number;
  onHoverClass: (cls: string | null) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const maskSrc = !mask.startsWith('data:') && !mask.startsWith('blob:')
    ? `data:image/png;base64,${mask}`
    : mask;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const origImg = new Image();
    const maskImg = new Image();
    let loaded = 0;

    const draw = () => {
      loaded++;
      if (loaded < 2) return;

      canvas.width = origImg.width;
      canvas.height = origImg.height;

      ctx.drawImage(origImg, 0, 0);
      ctx.globalAlpha = opacity;
      ctx.drawImage(maskImg, 0, 0, origImg.width, origImg.height);
      ctx.globalAlpha = 1.0;
    };

    origImg.onload = draw;
    maskImg.onload = draw;
    origImg.crossOrigin = 'anonymous';
    maskImg.crossOrigin = 'anonymous';
    origImg.src = original;
    maskImg.src = maskSrc;
  }, [original, maskSrc, opacity]);

  return (
    <canvas
      ref={canvasRef}
      className="w-full h-full object-contain"
      onMouseLeave={() => onHoverClass(null)}
    />
  );
}
