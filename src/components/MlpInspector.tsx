import React, { useState } from 'react';
import { gelu, TransformerRun } from '../engine/transformerEngine';
import { VectorStrip } from './VectorStrip';

interface MlpInspectorProps {
  run: TransformerRun;
  selectedTokenIdx: number;
  selectedLayer: number;
}

export const MlpInspector: React.FC<MlpInspectorProps> = ({
  run,
  selectedTokenIdx,
  selectedLayer,
}) => {
  const [hoveredNeuron, setHoveredNeuron] = useState<number | null>(null);

  const safeIdx = Math.min(selectedTokenIdx, run.tokens.length - 1);
  const tok = run.tokens[safeIdx];
  const layer = run.layers[selectedLayer] ?? run.layers[0];
  const mlp = layer.mlpStates[safeIdx] ?? layer.mlpStates[0];

  // Plot geometry for GELU curve (-3 to +3 on x, -0.5 to +3.1 on y)
  const plotW = 420;
  const plotH = 220;
  const padL = 42;
  const padR = 18;
  const padT = 18;
  const padB = 34;

  const toX = (x: number) => padL + ((x + 3) / 6) * (plotW - padL - padR);
  const toY = (y: number) => padT + ((3.1 - y) / 3.6) * (plotH - padT - padB);

  const geluCurve: string[] = [];
  const reluCurve: string[] = [];
  for (let i = 0; i <= 80; i++) {
    const x = -3 + (i / 80) * 6;
    geluCurve.push(`${toX(x).toFixed(1)},${toY(gelu(x)).toFixed(1)}`);
    reluCurve.push(`${toX(x).toFixed(1)},${toY(Math.max(0, x)).toFixed(1)}`);
  }

  const activeNeuronIdx = hoveredNeuron ?? 0;
  const activePre = mlp.expandedPreGelu[activeNeuronIdx];
  const activePost = mlp.expandedPostGelu[activeNeuronIdx];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col lg:flex-row lg:items-end justify-between gap-4">
        <div className="max-w-2xl">
          <h3 className="text-lg font-bold text-slate-900">
            03. Residual Stream, Layer Normalization &amp; MLP (GELU)
          </h3>
          <p className="text-sm text-slate-600 mt-1 leading-relaxed">
            While self-attention moves information <em>between</em> tokens, the Multilayer Perceptron (MLP) processes each token <em>independently</em>—expanding it 4× (<span className="font-mono text-slate-800">768 → 3072</span>), applying the smooth non-linear <strong>GELU</strong> gate, and projecting back into the residual highway.
          </p>
        </div>

        <div className="font-serif italic text-sm text-slate-800 bg-slate-50 border border-slate-200 rounded-lg px-3.5 py-2 shrink-0">
          GELU(x) = x · Φ(x) ≈ 0.5x (1 + tanh[√(2/π)(x + 0.0447x³)])
        </div>
      </div>

      {/* 2-Column Interactive Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: Residual & LayerNorm Pipeline */}
        <div className="lg:col-span-6 space-y-3 bg-slate-50 border border-slate-200 rounded-xl p-4">
          <div className="flex items-center justify-between text-xs">
            <span className="font-semibold text-slate-800">
              Token <span className="font-mono text-indigo-700">“{tok.text}”</span> inside Block {selectedLayer + 1}
            </span>
            <span className="font-mono text-slate-500 tabular-nums">
              μ = {mlp.layerNormMean.toFixed(2)} · σ = {mlp.layerNormStd.toFixed(2)}
            </span>
          </div>

          {/* Step A: Residual Addition */}
          <div className="bg-white border border-slate-200 rounded-lg p-3 space-y-1.5">
            <div className="flex items-center justify-between text-xs">
              <span className="font-semibold text-slate-900">1. Residual Add (x_prev + Attention Out)</span>
              <span className="font-mono text-slate-400">768d</span>
            </div>
            <VectorStrip values={mlp.residualPreMlp} palette="diverging" heightClass="h-6" showNumbers />
            <p className="text-[11px] text-slate-500">
              Skip connections let gradients and earlier token meanings bypass sub-layers intact.
            </p>
          </div>

          {/* Step B: LayerNorm */}
          <div className="bg-white border border-slate-200 rounded-lg p-3 space-y-1.5">
            <div className="flex items-center justify-between text-xs">
              <span className="font-semibold text-slate-900">2. Layer Normalization</span>
              <span className="font-mono text-slate-500">(x − μ) / σ</span>
            </div>
            <VectorStrip values={mlp.layerNormOut} palette="sky" heightClass="h-6" showNumbers />
            <p className="text-[11px] text-slate-500">
              Re-centers values to zero mean and unit variance so deep layers stay numerically stable.
            </p>
          </div>

          {/* Step C: MLP Expansion & Projection */}
          <div className="bg-white border border-slate-200 rounded-lg p-3 space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span className="font-semibold text-indigo-900">3. MLP Projection Out (3072 → 768)</span>
              <span className="font-mono text-indigo-600">W_proj · GELU(h)</span>
            </div>
            <VectorStrip values={mlp.projectedOut} palette="indigo" heightClass="h-6" showNumbers />
          </div>

          {/* Step D: Final Block Output */}
          <div className="bg-white border border-indigo-200 rounded-lg p-3 space-y-1.5">
            <div className="flex items-center justify-between text-xs">
              <span className="font-bold text-slate-900">4. Block {selectedLayer + 1} Output (Residual Add #2)</span>
              <span className="font-mono text-slate-500">→ Block {selectedLayer < 11 ? selectedLayer + 2 : 'Final Norm'}</span>
            </div>
            <VectorStrip values={mlp.finalBlockOutput} palette="diverging" heightClass="h-7" showNumbers />
          </div>
        </div>

        {/* Right: Interactive GELU Activation Curve */}
        <div className="lg:col-span-6 bg-slate-50 border border-slate-200 rounded-xl p-4 flex flex-col justify-between space-y-4">
          <div className="flex items-start justify-between gap-3">
            <div>
              <h4 className="text-sm font-bold text-slate-900">
                Live GELU Non-Linearity (12 Sample Hidden Neurons)
              </h4>
              <p className="text-xs text-slate-600 mt-0.5">
                Hover any point to compare its linear pre-activation against its smooth GELU output.
              </p>
            </div>
            <div className="text-right font-mono text-xs bg-white border border-slate-200 rounded-lg px-2.5 py-1.5 tabular-nums shrink-0">
              <div className="text-slate-500 text-[10px]">Neuron #{activeNeuronIdx + 1}</div>
              <div className="text-slate-900 font-semibold">
                {activePre.toFixed(2)} → <span className="text-indigo-700">{activePost.toFixed(2)}</span>
              </div>
            </div>
          </div>

          <svg
            viewBox={`0 0 ${plotW} ${plotH}`}
            className="w-full h-56 overflow-visible"
            aria-label="Interactive GELU activation function plot with live neuron values"
          >
            {/* Grid lines */}
            {[-2, -1, 0, 1, 2].map((gx) => (
              <g key={`gx-${gx}`}>
                <line
                  x1={toX(gx)}
                  y1={padT}
                  x2={toX(gx)}
                  y2={plotH - padB}
                  stroke={gx === 0 ? '#94a3b8' : '#e2e8f0'}
                  strokeWidth={gx === 0 ? 1.2 : 1}
                />
                <text
                  x={toX(gx)}
                  y={plotH - padB + 15}
                  textAnchor="middle"
                  className="text-[10px] font-mono fill-slate-500"
                >
                  {gx}
                </text>
              </g>
            ))}
            {[0, 1, 2, 3].map((gy) => (
              <g key={`gy-${gy}`}>
                <line
                  x1={padL}
                  y1={toY(gy)}
                  x2={plotW - padR}
                  y2={toY(gy)}
                  stroke={gy === 0 ? '#94a3b8' : '#e2e8f0'}
                  strokeWidth={gy === 0 ? 1.2 : 1}
                />
                <text
                  x={padL - 8}
                  y={toY(gy) + 3}
                  textAnchor="end"
                  className="text-[10px] font-mono fill-slate-500"
                >
                  {gy}
                </text>
              </g>
            ))}

            {/* Reference ReLU dashed line */}
            <polyline
              fill="none"
              stroke="#94a3b8"
              strokeWidth="1.5"
              strokeDasharray="4 4"
              points={reluCurve.join(' ')}
            />

            {/* Main GELU curve */}
            <polyline
              fill="none"
              stroke="#4f46e5"
              strokeWidth="2.5"
              strokeLinecap="round"
              points={geluCurve.join(' ')}
            />

            {/* Live neuron activation points */}
            {mlp.expandedPreGelu.map((xVal, idx) => {
              const yVal = mlp.expandedPostGelu[idx];
              const isHovered = idx === activeNeuronIdx;
              const cx = toX(xVal);
              const cy = toY(yVal);
              return (
                <g
                  key={idx}
                  onMouseEnter={() => setHoveredNeuron(idx)}
                  onClick={() => setHoveredNeuron(idx)}
                  className="cursor-pointer"
                >
                  {isHovered && (
                    <>
                      <line x1={cx} y1={toY(0)} x2={cx} y2={cy} stroke="#4f46e5" strokeWidth="1" strokeDasharray="2 2" />
                      <line x1={toX(0)} y1={cy} x2={cx} y2={cy} stroke="#4f46e5" strokeWidth="1" strokeDasharray="2 2" />
                    </>
                  )}
                  <circle
                    cx={cx}
                    cy={cy}
                    r={isHovered ? 6.5 : 4.5}
                    fill={isHovered ? '#4f46e5' : '#ffffff'}
                    stroke="#4f46e5"
                    strokeWidth={isHovered ? 2.5 : 2}
                    className="transition-all duration-150"
                  />
                </g>
              );
            })}

            {/* Axis labels */}
            <text x={plotW / 2 + 12} y={plotH - 3} textAnchor="middle" className="text-[10px] font-medium fill-slate-500">
              Pre-activation (x)
            </text>
            <text
              x={12}
              y={plotH / 2 - 8}
              textAnchor="middle"
              transform={`rotate(-90 12 ${plotH / 2 - 8})`}
              className="text-[10px] font-medium fill-slate-500"
            >
              GELU(x)
            </text>
          </svg>

          <div className="flex flex-wrap items-center justify-between gap-2 text-[11px] text-slate-600 pt-1 border-t border-slate-200/70">
            <div className="flex items-center gap-4">
              <span className="inline-flex items-center gap-1.5">
                <span className="w-3.5 h-0.5 bg-indigo-600 inline-block" />
                <span>GELU(x) — Smooth gate</span>
              </span>
              <span className="inline-flex items-center gap-1.5">
                <span className="w-3.5 h-0.5 border-t border-dashed border-slate-400 inline-block" />
                <span>ReLU(x) — Hard zero</span>
              </span>
            </div>
            <span className="text-slate-500">Slightly negative dip preserves small gradients</span>
          </div>
        </div>
      </div>
    </div>
  );
};
