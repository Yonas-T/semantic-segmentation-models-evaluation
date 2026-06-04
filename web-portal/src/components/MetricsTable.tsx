'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis,
  PolarRadiusAxis, Radar,
} from 'recharts';
import { Activity, Zap, HardDrive, Cpu, ChevronUp, ChevronDown } from 'lucide-react';
import { CITYSCAPES_CLASSES, CITYSCAPES_COLORS, colorToHex } from '@/lib/cityscapesPalette';

// ─── Types ──────────────────────────────────────────────────────────────────

export interface ModelMetrics {
  name: string;
  mIoU: number;
  pixelAccuracy: number;
  perClassIoU: Record<string, number>;
  latencyMs: { mean: number; p50: number; p95: number };
  fps: number;
  modelSizeMB: number;
  paramCount: number;
  peakMemoryMB: number;
}

export interface MetricsData {
  models: {
    segformer: ModelMetrics;
    deeplabv3: ModelMetrics;
  };
  metadata: {
    dataset: string;
    numClasses: number;
    evalResolution: string;
    evalImages: number;
    gpuName: string;
    timestamp: string;
  };
}

interface MetricsTableProps {
  metrics: MetricsData | null;
}

type SortKey = 'class' | 'segformer' | 'deeplabv3' | 'delta';
type SortDir = 'asc' | 'desc';

// ─── Summary Card ───────────────────────────────────────────────────────────

function SummaryCard({
  icon: Icon,
  title,
  segValue,
  dlValue,
  segLabel,
  dlLabel,
  format = 'number',
  higherBetter = true,
  delay = 0,
}: {
  icon: React.ElementType;
  title: string;
  segValue: number;
  dlValue: number;
  segLabel: string;
  dlLabel: string;
  format?: 'number' | 'percent' | 'ms' | 'mb' | 'fps';
  higherBetter?: boolean;
  delay?: number;
}) {
  const segBetter = higherBetter ? segValue > dlValue : segValue < dlValue;

  const fmt = (v: number) => {
    switch (format) {
      case 'percent': return `${(v * 100).toFixed(1)}%`;
      case 'ms': return `${v} ms`;
      case 'mb': return `${v.toFixed(1)} MB`;
      case 'fps': return `${v.toFixed(1)}`;
      default: return v.toLocaleString();
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay }}
      className="glass-card-hover p-5 relative overflow-hidden group"
    >
      {/* Gradient accent line */}
      <div className="absolute top-0 left-0 right-0 h-0.5 bg-gradient-to-r from-purple-500 to-blue-500 opacity-50 group-hover:opacity-100 transition-opacity" />

      <div className="flex items-start justify-between mb-3">
        <div className="p-2 rounded-xl bg-gradient-to-br from-purple-500/10 to-blue-500/10 border border-white/5">
          <Icon className="w-4 h-4 text-purple-400" />
        </div>
        <span className="text-xs text-white/40 font-medium uppercase tracking-wider">{title}</span>
      </div>

      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-xs text-blue-300/70">{segLabel}</span>
          <span className={`text-lg font-bold font-mono ${segBetter ? 'text-emerald-400' : 'text-white/80'}`}>
            {fmt(segValue)}
          </span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-xs text-purple-300/70">{dlLabel}</span>
          <span className={`text-lg font-bold font-mono ${!segBetter ? 'text-emerald-400' : 'text-white/80'}`}>
            {fmt(dlValue)}
          </span>
        </div>
      </div>
    </motion.div>
  );
}

// ─── Custom Tooltip ─────────────────────────────────────────────────────────

/* eslint-disable @typescript-eslint/no-explicit-any */
function CustomBarTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-[#1a1a2e]/95 backdrop-blur-md border border-white/10 rounded-xl p-3 shadow-xl">
      <p className="text-white/90 font-medium mb-2">{label}</p>
      {payload.map((entry: any) => (
        <div key={entry.dataKey} className="flex items-center gap-2 text-sm">
          <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: entry.color }} />
          <span className="text-white/60">{entry.name}:</span>
          <span className="text-white/90 font-mono">{(entry.value * 100).toFixed(1)}%</span>
        </div>
      ))}
    </div>
  );
}
/* eslint-enable @typescript-eslint/no-explicit-any */

// ─── Main Component ─────────────────────────────────────────────────────────

export default function MetricsTable({ metrics }: MetricsTableProps) {
  const [sortKey, setSortKey] = useState<SortKey>('class');
  const [sortDir, setSortDir] = useState<SortDir>('asc');

  if (!metrics) {
    return (
      <div className="glass-card p-12 text-center">
        <Activity className="w-12 h-12 text-white/20 mx-auto mb-4" />
        <p className="text-white/40 text-lg">Loading metrics…</p>
      </div>
    );
  }

  const { segformer, deeplabv3 } = metrics.models;

  // ── Per-class data for charts ──
  const perClassData = CITYSCAPES_CLASSES.map((cls, i) => ({
    name: cls,
    segformer: segformer.perClassIoU[cls] ?? 0,
    deeplabv3: deeplabv3.perClassIoU[cls] ?? 0,
    delta: (segformer.perClassIoU[cls] ?? 0) - (deeplabv3.perClassIoU[cls] ?? 0),
    color: colorToHex(CITYSCAPES_COLORS[i]),
  }));

  // ── Sort table ──
  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortDir(d => d === 'asc' ? 'desc' : 'asc');
    } else {
      setSortKey(key);
      setSortDir('desc');
    }
  };

  const sortedData = [...perClassData].sort((a, b) => {
    let va: number | string, vb: number | string;
    switch (sortKey) {
      case 'class': va = a.name; vb = b.name; break;
      case 'segformer': va = a.segformer; vb = b.segformer; break;
      case 'deeplabv3': va = a.deeplabv3; vb = b.deeplabv3; break;
      case 'delta': va = a.delta; vb = b.delta; break;
      default: va = a.name; vb = b.name;
    }
    if (typeof va === 'string') {
      return sortDir === 'asc' ? va.localeCompare(vb as string) : (vb as string).localeCompare(va);
    }
    return sortDir === 'asc' ? (va as number) - (vb as number) : (vb as number) - (va as number);
  });

  // ── Radar chart data ──
  const maxLatency = Math.max(segformer.latencyMs.mean, deeplabv3.latencyMs.mean);
  const maxSize = Math.max(segformer.modelSizeMB, deeplabv3.modelSizeMB);
  const maxMem = Math.max(segformer.peakMemoryMB, deeplabv3.peakMemoryMB);

  const radarData = [
    {
      metric: 'mIoU',
      segformer: segformer.mIoU * 100,
      deeplabv3: deeplabv3.mIoU * 100,
    },
    {
      metric: 'Speed',
      segformer: (1 - segformer.latencyMs.mean / maxLatency / 1.5) * 100,
      deeplabv3: (1 - deeplabv3.latencyMs.mean / maxLatency / 1.5) * 100,
    },
    {
      metric: 'Memory Eff.',
      segformer: (1 - segformer.peakMemoryMB / maxMem / 1.2) * 100,
      deeplabv3: (1 - deeplabv3.peakMemoryMB / maxMem / 1.2) * 100,
    },
    {
      metric: 'Size Eff.',
      segformer: (1 - segformer.modelSizeMB / maxSize / 1.2) * 100,
      deeplabv3: (1 - deeplabv3.modelSizeMB / maxSize / 1.2) * 100,
    },
    {
      metric: 'Pixel Acc.',
      segformer: segformer.pixelAccuracy * 100,
      deeplabv3: deeplabv3.pixelAccuracy * 100,
    },
  ];

  const SortIcon = ({ active, dir }: { active: boolean; dir: SortDir }) => {
    if (!active) return <ChevronUp className="w-3 h-3 text-white/20" />;
    return dir === 'asc'
      ? <ChevronUp className="w-3 h-3 text-purple-400" />
      : <ChevronDown className="w-3 h-3 text-purple-400" />;
  };

  return (
    <div className="space-y-8">
      {/* ─── Section title ──────────────────────────────────── */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="text-center"
      >
        <h2 className="text-2xl md:text-3xl font-bold gradient-text">
          Benchmark Analytics
        </h2>
        <p className="text-white/40 mt-2 text-sm">
          {metrics.metadata.evalImages} images • {metrics.metadata.evalResolution} • {metrics.metadata.gpuName}
        </p>
      </motion.div>

      {/* ─── Summary cards ──────────────────────────────────── */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <SummaryCard
          icon={Activity}
          title="mIoU"
          segValue={segformer.mIoU}
          dlValue={deeplabv3.mIoU}
          segLabel="SegFormer"
          dlLabel="DeepLabV3+"
          format="percent"
          delay={0}
        />
        <SummaryCard
          icon={Zap}
          title="Latency"
          segValue={segformer.latencyMs.mean}
          dlValue={deeplabv3.latencyMs.mean}
          segLabel="SegFormer"
          dlLabel="DeepLabV3+"
          format="ms"
          higherBetter={false}
          delay={0.1}
        />
        <SummaryCard
          icon={Cpu}
          title="FPS"
          segValue={segformer.fps}
          dlValue={deeplabv3.fps}
          segLabel="SegFormer"
          dlLabel="DeepLabV3+"
          format="fps"
          delay={0.2}
        />
        <SummaryCard
          icon={HardDrive}
          title="Model Size"
          segValue={segformer.modelSizeMB}
          dlValue={deeplabv3.modelSizeMB}
          segLabel="SegFormer"
          dlLabel="DeepLabV3+"
          format="mb"
          higherBetter={false}
          delay={0.3}
        />
      </div>

      {/* ─── Charts row ─────────────────────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Per-class IoU bar chart (takes 2/3 width) */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="lg:col-span-2 glass-card p-6"
        >
          <h3 className="text-lg font-semibold text-white/90 mb-4">Per-Class IoU</h3>
          <ResponsiveContainer width="100%" height={350}>
            <BarChart data={perClassData} margin={{ top: 5, right: 5, bottom: 60, left: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
              <XAxis
                dataKey="name"
                tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 10 }}
                angle={-45}
                textAnchor="end"
                height={60}
              />
              <YAxis
                tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 11 }}
                domain={[0, 1]}
                tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
              />
              <Tooltip content={<CustomBarTooltip />} />
              <Legend
                wrapperStyle={{ paddingTop: 10, fontSize: 12 }}
                iconType="circle"
              />
              <Bar dataKey="segformer" name="SegFormer-B2" fill="#3b82f6" radius={[3, 3, 0, 0]} />
              <Bar dataKey="deeplabv3" name="DeepLabV3+" fill="#a855f7" radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </motion.div>

        {/* Radar chart (1/3 width) */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="glass-card p-6"
        >
          <h3 className="text-lg font-semibold text-white/90 mb-4">Performance Profile</h3>
          <ResponsiveContainer width="100%" height={350}>
            <RadarChart data={radarData} cx="50%" cy="50%" outerRadius="70%">
              <PolarGrid stroke="rgba(255,255,255,0.1)" />
              <PolarAngleAxis
                dataKey="metric"
                tick={{ fill: 'rgba(255,255,255,0.6)', fontSize: 11 }}
              />
              <PolarRadiusAxis
                angle={90}
                domain={[0, 100]}
                tick={{ fill: 'rgba(255,255,255,0.3)', fontSize: 9 }}
              />
              <Radar
                name="SegFormer-B2"
                dataKey="segformer"
                stroke="#3b82f6"
                fill="#3b82f6"
                fillOpacity={0.2}
                strokeWidth={2}
              />
              <Radar
                name="DeepLabV3+"
                dataKey="deeplabv3"
                stroke="#a855f7"
                fill="#a855f7"
                fillOpacity={0.2}
                strokeWidth={2}
              />
              <Legend
                wrapperStyle={{ fontSize: 12, paddingTop: 10 }}
                iconType="circle"
              />
            </RadarChart>
          </ResponsiveContainer>
        </motion.div>
      </div>

      {/* ─── Detailed table ─────────────────────────────────── */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.6 }}
        className="glass-card overflow-hidden"
      >
        <div className="p-5 border-b border-white/10">
          <h3 className="text-lg font-semibold text-white/90">Detailed Per-Class Comparison</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-white/10">
                <th className="text-left px-5 py-3 text-white/50 font-medium">
                  <button onClick={() => handleSort('class')} className="flex items-center gap-1 hover:text-white/80 transition-colors">
                    Class <SortIcon active={sortKey === 'class'} dir={sortDir} />
                  </button>
                </th>
                <th className="text-right px-5 py-3 text-blue-400/70 font-medium">
                  <button onClick={() => handleSort('segformer')} className="flex items-center gap-1 ml-auto hover:text-blue-300 transition-colors">
                    SegFormer IoU <SortIcon active={sortKey === 'segformer'} dir={sortDir} />
                  </button>
                </th>
                <th className="text-right px-5 py-3 text-purple-400/70 font-medium">
                  <button onClick={() => handleSort('deeplabv3')} className="flex items-center gap-1 ml-auto hover:text-purple-300 transition-colors">
                    DeepLabV3+ IoU <SortIcon active={sortKey === 'deeplabv3'} dir={sortDir} />
                  </button>
                </th>
                <th className="text-right px-5 py-3 text-white/50 font-medium">
                  <button onClick={() => handleSort('delta')} className="flex items-center gap-1 ml-auto hover:text-white/80 transition-colors">
                    Δ <SortIcon active={sortKey === 'delta'} dir={sortDir} />
                  </button>
                </th>
                <th className="text-center px-5 py-3 text-white/50 font-medium">Winner</th>
              </tr>
            </thead>
            <tbody>
              {sortedData.map((row, i) => {
                const segWins = row.segformer > row.deeplabv3;
                return (
                  <tr
                    key={row.name}
                    className={`border-b border-white/5 ${
                      i % 2 === 0 ? 'bg-white/[0.02]' : ''
                    } hover:bg-white/[0.05] transition-colors`}
                  >
                    <td className="px-5 py-3">
                      <div className="flex items-center gap-2">
                        <div
                          className="w-3 h-3 rounded-sm flex-shrink-0"
                          style={{ backgroundColor: row.color }}
                        />
                        <span className="text-white/80">{row.name}</span>
                      </div>
                    </td>
                    <td className={`text-right px-5 py-3 font-mono ${segWins ? 'text-emerald-400' : 'text-white/60'}`}>
                      {(row.segformer * 100).toFixed(1)}%
                    </td>
                    <td className={`text-right px-5 py-3 font-mono ${!segWins ? 'text-emerald-400' : 'text-white/60'}`}>
                      {(row.deeplabv3 * 100).toFixed(1)}%
                    </td>
                    <td className={`text-right px-5 py-3 font-mono text-xs ${
                      row.delta > 0 ? 'text-blue-400' : row.delta < 0 ? 'text-purple-400' : 'text-white/30'
                    }`}>
                      {row.delta > 0 ? '+' : ''}{(row.delta * 100).toFixed(1)}%
                    </td>
                    <td className="text-center px-5 py-3">
                      <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${
                        segWins
                          ? 'bg-blue-500/20 text-blue-300'
                          : 'bg-purple-500/20 text-purple-300'
                      }`}>
                        {segWins ? 'SF' : 'DL'}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </motion.div>
    </div>
  );
}
