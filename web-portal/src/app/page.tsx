'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import { ExternalLink, Code2 } from 'lucide-react';
import ImageUploader from '@/components/ImageUploader';
import MaskViewer from '@/components/MaskViewer';
import MetricsTable, { type MetricsData } from '@/components/MetricsTable';

export default function DashboardPage() {
  // ── State ──
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [originalImageUrl, setOriginalImageUrl] = useState<string | null>(null);
  const [segformerMask, setSegformerMask] = useState<string | null>(null);
  const [deeplabMask, setDeeplabMask] = useState<string | null>(null);
  const [segformerLatency, setSegformerLatency] = useState(0);
  const [deeplabLatency, setDeeplabLatency] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);
  const [metrics, setMetrics] = useState<MetricsData | null>(null);
  const [error, setError] = useState<string | null>(null);

  const resultsRef = useRef<HTMLDivElement>(null);

  // ── Load pre-computed metrics on mount ──
  useEffect(() => {
    fetch('/metrics.json')
      .then((res) => res.json())
      .then((data: MetricsData) => setMetrics(data))
      .catch((err) => console.warn('Could not load metrics.json:', err));
  }, []);

  // ── Handle image selection + inference ──
  const handleImageSelect = useCallback(async (file: File) => {
    setSelectedImage(file);
    setError(null);

    // Create preview URL
    const url = URL.createObjectURL(file);
    setOriginalImageUrl(url);

    // Reset previous results
    setSegformerMask(null);
    setDeeplabMask(null);
    setSegformerLatency(0);
    setDeeplabLatency(0);
    setIsProcessing(true);

    try {
      const formData = new FormData();
      formData.append('image', file);

      const res = await fetch('/api/inference', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({ error: 'Unknown error' }));
        throw new Error(errData.error || `Server error: ${res.status}`);
      }

      const data = await res.json();

      // Handle mock mode
      if (data.message) {
        setError(data.message);
      }

      if (data.segformer?.mask) {
        setSegformerMask(data.segformer.mask);
        setSegformerLatency(data.segformer.latencyMs);
      }
      if (data.deeplabv3?.mask) {
        setDeeplabMask(data.deeplabv3.mask);
        setDeeplabLatency(data.deeplabv3.latencyMs);
      }

      // Scroll to results
      setTimeout(() => {
        resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }, 300);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Inference failed';
      setError(message);
      console.error('[Inference]', err);
    } finally {
      setIsProcessing(false);
    }
  }, []);

  return (
    <div className="min-h-screen">
      {/* ─── Hero ──────────────────────────────────────────── */}
      <header className="relative overflow-hidden">
        {/* Decorative glow */}
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[600px] h-[300px] bg-gradient-to-b from-purple-500/10 to-transparent rounded-full blur-3xl -z-10" />

        <div className="max-w-6xl mx-auto px-4 sm:px-6 pt-16 pb-10 text-center">
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-white/5 border border-white/10 text-xs text-white/60 mb-6">
              <div className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
              Cityscapes · 19 Classes · 512×1024
            </div>

            <h1 className="text-4xl sm:text-5xl md:text-6xl font-extrabold tracking-tight leading-tight">
              <span className="gradient-text">Semantic Segmentation</span>
              <br />
              <span className="text-white/90">Benchmark</span>
            </h1>

            <p className="mt-5 text-base md:text-lg text-white/50 max-w-2xl mx-auto leading-relaxed">
              Compare <span className="text-blue-400 font-medium">SegFormer-B2</span>{' '}
              <span className="text-white/30">(Transformer)</span> against{' '}
              <span className="text-purple-400 font-medium">DeepLabV3+</span>{' '}
              <span className="text-white/30">(CNN)</span> on urban street scenes —
              upload an image to see pixel-level predictions side by side.
            </p>
          </motion.div>
        </div>
      </header>

      {/* ─── Main Content ─────────────────────────────────── */}
      <main className="max-w-6xl mx-auto px-4 sm:px-6 pb-24 space-y-16">
        {/* Upload section */}
        <section id="upload">
          <ImageUploader
            onImageSelect={handleImageSelect}
            isProcessing={isProcessing}
          />
          {error && (
            <motion.div
              initial={{ opacity: 0, y: -5 }}
              animate={{ opacity: 1, y: 0 }}
              className="mt-4 glass-card p-4 border-amber-500/20 text-center"
            >
              <p className="text-amber-400/80 text-sm">{error}</p>
            </motion.div>
          )}
        </section>

        {/* Results section */}
        {(originalImageUrl || segformerMask || deeplabMask) && (
          <section id="results" ref={resultsRef}>
            <MaskViewer
              originalImage={originalImageUrl}
              segformerMask={segformerMask}
              deeplabMask={deeplabMask}
              segformerLatency={segformerLatency}
              deeplabLatency={deeplabLatency}
            />
          </section>
        )}

        {/* Metrics section */}
        {/* <section id="metrics">
          <MetricsTable metrics={metrics} />
        </section> */}
      </main>

      {/* ─── Footer ───────────────────────────────────────── */}
      <footer className="border-t border-white/5 py-8">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 flex flex-col sm:flex-row items-center justify-between gap-4 text-sm text-white/30">
          <p>
            Semantic Segmentation Benchmark
          </p>
          <div className="flex items-center gap-4">
            <a
              href="https://www.cityscapes-dataset.com/"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1 hover:text-white/60 transition-colors"
            >
              <ExternalLink className="w-3.5 h-3.5" />
              Cityscapes
            </a>
            <a
              href="https://huggingface.co/nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1 hover:text-white/60 transition-colors"
            >
              <ExternalLink className="w-3.5 h-3.5" />
              SegFormer
            </a>
            <a
              href="https://github.com"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1 hover:text-white/60 transition-colors"
            >
              <Code2 className="w-3.5 h-3.5" />
              Source
            </a>
          </div>
        </div>
      </footer>
    </div>
  );
}
