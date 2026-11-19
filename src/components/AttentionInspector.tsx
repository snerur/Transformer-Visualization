import React, { useState } from 'react';
import { HEAD_ROLES, TransformerRun } from '../engine/transformerEngine';
import { VectorStrip } from './VectorStrip';

interface AttentionInspectorProps {
  run: TransformerRun;
  selectedTokenIdx: number;
  onSelectToken: (idx: number) => void;
  selectedLayer: number;
  onSelectLayer: (layer: number) => void;
  selectedHead: number;
  onSelectHead: (head: number) => void;
}

export const AttentionInspector: React.FC<AttentionInspectorProps> = ({
  run,
  selectedTokenIdx,
  onSelectToken,
  selectedLayer,
  onSelectLayer,
  selectedHead,
  onSelectHead,
}) => {
  const safeQ = Math.min(selectedTokenIdx, run.tokens.length - 1);
  const [selectedKeyIdx, setSelectedKeyIdx] = useState<number>(Math.max(0, safeQ - 1));

  const layer = run.layers[selectedLayer] ?? run.layers[0];
  const head = layer.heads[selectedHead] ?? layer.heads[0];
  const safeK = Math.min(selectedKeyIdx, run.tokens.length - 1);

  const isCausalMasked = safeK > safeQ;
  const rawDot = head.dotScores[safeQ][safeK];
  const scaledScore = head.scaledScores[safeQ][safeK];
  const attnWeight = head.attentionWeights[safeQ][safeK];

  // Coordinates for Token-to-Token Attention Arc Diagram
  const svgWidth = 640;
  const n = run.tokens.length;
  const padX = 48;
  const stepX = n > 1 ? (svgWidth - padX * 2) / (n - 1) : 0;
  const tokenX = (idx: number) => (n === 1 ? svgWidth / 2 : padX + idx * stepX);
  const baseY = 128;

  return (
    <div className="space-y-6">
      {/* Header & Layer/Head summary */}
      <div className="flex flex-col lg:flex-row lg:items-end justify-between gap-4">
        <div className="max-w-2xl">
          <h3 className="text-lg font-bold text-slate-900">
            02. Multi-Head Masked Self-Attention (Q · K · V)
          </h3>
          <p className="text-sm text-slate-600 mt-1 leading-relaxed">
            Each token emits a <span className="font-semibold text-sky-700">Query</span> (what it seeks), a <span className="font-semibold text-teal-700">Key</span> (what it offers), and a <span className="font-semibold text-amber-700">Value</span> (its payload). Because GPT-2 predicts the future, a <strong className="text-slate-800">Causal Mask</strong> hides all tokens to the right (<span className="font-mono text-slate-800">k &gt; q</span>).
          </p>
        </div>

        {/* Formula box */}
        <div className="font-serif italic text-sm text-slate-800 bg-slate-50 border border-slate-200 rounded-lg px-3.5 py-2 shrink-0">
          Attention(Q, K, V) = softmax( (Q · K<sup>T</sup>) / √d<sub>k</sub> + Mask ) · V
        </div>
      </div>

      {/* Interactive Layer Selector Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-1 border-b border-slate-100">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-xs font-semibold text-slate-500">Transformer Block</span>
          <div className="flex items-center gap-1 flex-wrap">
            {Array.from({ length: 12 }, (_, idx) => (
              <button
                key={idx}
                type="button"
                onClick={() => onSelectLayer(idx)}
                className={`w-7 h-7 text-xs font-mono tabular-nums rounded-md transition-colors focus-visible:outline-2 focus-visible:outline-sky-600 ${
                  selectedLayer === idx
                    ? 'bg-slate-900 text-white font-semibold'
                    : 'bg-slate-100 text-slate-600 hover:bg-slate-200 hover:text-slate-900'
                }`}
              >
                {idx + 1}
              </button>
            ))}
          </div>
        </div>

        <div className="text-xs text-slate-600">
          Active Head <span className="font-mono font-semibold text-slate-900">{selectedHead + 1}</span>
          <span className="mx-1.5 text-slate-300" aria-hidden="true">·</span>
          <span className="font-semibold text-teal-800">{head.roleName}</span>
          <span className="hidden sm:inline text-slate-500"> — {head.roleSummary}</span>
        </div>
      </div>

      {/* Main 2-column interactive diagram + Pairwise Math Inspector */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: Attention Arc Diagram */}
        <div className="lg:col-span-7 bg-slate-50 border border-slate-200 rounded-xl p-4 space-y-3">
          <div className="flex items-center justify-between text-xs">
            <span className="font-semibold text-slate-800">
              Attention Arcs from Query <span className="font-mono text-sky-700">“{run.tokens[safeQ].text}”</span>
            </span>
            <span className="text-slate-500">Click any target token to inspect its Q·K calculation</span>
          </div>

          <div className="overflow-x-auto">
            <svg
              viewBox={`0 0 ${svgWidth} 164`}
              className="w-full min-w-[460px] h-44"
              aria-label="Token-to-token attention arcs"
            >
              {/* Draw causal arcs from safeQ to all j <= safeQ */}
              {run.tokens.map((kTok, j) => {
                const x1 = tokenX(safeQ);
                const x2 = tokenX(j);
                const weight = head.attentionWeights[safeQ][j];
                const masked = j > safeQ;

                if (masked) {
                  // Subtle dashed blocked indicator for future tokens
                  const midX = (x1 + x2) / 2;
                  const ctrlY = baseY - 26 - Math.min(52, (j - safeQ) * 14);
                  return (
                    <path
                      key={`arc-masked-${j}`}
                      d={`M ${x1} ${baseY - 12} Q ${midX} ${ctrlY} ${x2} ${baseY - 12}`}
                      fill="none"
                      stroke="#cbd5e1"
                      strokeWidth="1"
                      strokeDasharray="3 4"
                      opacity="0.65"
                    />
                  );
                }

                const isSelf = j === safeQ;
                const midX = (x1 + x2) / 2;
                const arcHeight = isSelf ? 42 : 34 + Math.min(72, (safeQ - j) * 16);
                const strokeW = Math.max(1.2, weight * 7.5);
                const isInspectedKey = j === safeK;

                return (
                  <g key={`arc-${j}`}>
                    {isSelf ? (
                      <path
                        d={`M ${x1 - 8} ${baseY - 12} C ${x1 - 24} ${baseY - arcHeight - 16}, ${x1 + 24} ${baseY - arcHeight - 16}, ${x1 + 8} ${baseY - 12}`}
                        fill="none"
                        stroke={isInspectedKey ? '#0f172a' : '#0284c7'}
                        strokeWidth={strokeW}
                        strokeOpacity={Math.max(0.22, weight)}
                      />
                    ) : (
                      <path
                        d={`M ${x1} ${baseY - 12} Q ${midX} ${baseY - arcHeight} ${x2} ${baseY - 12}`}
                        fill="none"
                        stroke={isInspectedKey ? '#0f172a' : '#0284c7'}
                        strokeWidth={strokeW}
                        strokeOpacity={Math.max(0.2, weight + 0.12)}
                        className="transition-all duration-150"
                      />
                    )}

                    {/* Weight label for significant arcs or selected key */}
                    {(weight >= 0.12 || isInspectedKey) && (
                      <text
                        x={isSelf ? x1 : midX}
                        y={isSelf ? baseY - arcHeight - 16 : baseY - arcHeight * 0.56 - 6}
                        textAnchor="middle"
                        className={`text-[10px] font-mono tabular-nums ${
                          isInspectedKey ? 'fill-slate-900 font-bold' : 'fill-sky-800 font-medium'
                        }`}
                      >
                        {Math.round(weight * 100)}%
                      </text>
                    )}
                  </g>
                );
              })}

              {/* Token nodes along bottom */}
              {run.tokens.map((tok, idx) => {
                const cx = tokenX(idx);
                const isQ = idx === safeQ;
                const isK = idx === safeK;
                const masked = idx > safeQ;

                return (
                  <g
                    key={`node-${idx}`}
                    onClick={() => setSelectedKeyIdx(idx)}
                    onDoubleClick={() => onSelectToken(idx)}
                    className="cursor-pointer"
                  >
                    <circle
                      cx={cx}
                      cy={baseY}
                      r={isQ ? 7 : isK ? 6.5 : 5}
                      fill={isQ ? '#0284c7' : masked ? '#e2e8f0' : isK ? '#0d9488' : '#ffffff'}
                      stroke={isQ ? '#0284c7' : isK ? '#0f172a' : masked ? '#cbd5e1' : '#64748b'}
                      strokeWidth={isQ || isK ? 2 : 1.5}
                    />
                    <text
                      x={cx}
                      y={baseY + 21}
                      textAnchor="middle"
                      className={`text-[11px] font-mono ${
                        isQ
                          ? 'fill-sky-800 font-bold'
                          : isK
                          ? 'fill-teal-800 font-semibold'
                          : masked
                          ? 'fill-slate-400'
                          : 'fill-slate-700'
                      }`}
                    >
                      {tok.text.length > 9 ? `${tok.text.slice(0, 8)}…` : tok.text}
                    </text>
                    {masked && (
                      <text
                        x={cx}
                        y={baseY + 33}
                        textAnchor="middle"
                        className="text-[9px] font-mono fill-slate-400"
                      >
                        masked
                      </text>
                    )}
                  </g>
                );
              })}
            </svg>
          </div>

          <div className="flex items-center justify-between text-[11px] text-slate-500 pt-1 border-t border-slate-200/70">
            <span>Solid blue arcs = visible past context (sum to 100%)</span>
            <span>Dashed grey arcs = future tokens blocked by Causal Mask</span>
          </div>
        </div>

        {/* Right: Step-by-Step Q · K · V Pairwise Computation */}
        <div className="lg:col-span-5 bg-slate-50 border border-slate-200 rounded-xl p-4 space-y-3.5">
          <div className="flex items-center justify-between gap-2">
            <h4 className="text-sm font-bold text-slate-900">
              Pairwise Score Breakdown
            </h4>
            <span className="font-mono text-xs text-slate-600">
              Q[{safeQ}] · K[{safeK}]
            </span>
          </div>

          {/* Query & Key vectors */}
          <div className="space-y-2.5">
            <div className="space-y-1">
              <div className="flex justify-between text-xs">
                <span className="font-semibold text-sky-700">
                  Query: “{run.tokens[safeQ].text}”
                </span>
                <span className="font-mono text-slate-400">pos {safeQ}</span>
              </div>
              <VectorStrip values={head.queries[safeQ]} palette="sky" heightClass="h-6" showNumbers />
            </div>

            <div className="space-y-1">
              <div className="flex justify-between text-xs">
                <span className="font-semibold text-teal-700">
                  Key: “{run.tokens[safeK].text}”
                </span>
                <span className="font-mono text-slate-400">pos {safeK}</span>
              </div>
              <VectorStrip values={head.keys[safeK]} palette="teal" heightClass="h-6" showNumbers />
            </div>
          </div>

          {/* Numeric calculation pipeline */}
          <div className="grid grid-cols-2 gap-2 pt-1 text-xs">
            <div className="bg-white border border-slate-200 rounded-lg p-2.5">
              <div className="text-slate-500">1. Dot Product (Q·K)</div>
              <div className="font-mono font-bold text-slate-900 text-sm mt-0.5 tabular-nums">
                {rawDot >= 0 ? `+${rawDot.toFixed(2)}` : rawDot.toFixed(2)}
              </div>
            </div>
            <div className="bg-white border border-slate-200 rounded-lg p-2.5">
              <div className="text-slate-500">2. Scaled (÷ √64)</div>
              <div className="font-mono font-bold text-slate-900 text-sm mt-0.5 tabular-nums">
                {scaledScore >= 0 ? `+${scaledScore.toFixed(2)}` : scaledScore.toFixed(2)}
              </div>
            </div>
            <div className="bg-white border border-slate-200 rounded-lg p-2.5">
              <div className="text-slate-500">3. Causal Mask</div>
              <div
                className={`font-mono font-bold text-sm mt-0.5 ${
                  isCausalMasked ? 'text-rose-700' : 'text-emerald-700'
                }`}
              >
                {isCausalMasked ? '−∞ (Blocked)' : 'Pass (k ≤ q)'}
              </div>
            </div>
            <div className="bg-white border border-slate-200 rounded-lg p-2.5">
              <div className="text-slate-500">4. Softmax Weight</div>
              <div className="font-mono font-bold text-sky-700 text-sm mt-0.5 tabular-nums">
                {(attnWeight * 100).toFixed(1)}%
              </div>
            </div>
          </div>

          {/* Resulting Value Output */}
          <div className="pt-2 border-t border-slate-200 space-y-1">
            <div className="flex justify-between text-xs">
              <span className="font-semibold text-amber-800">
                Weighted Value Output (∑ α · V)
              </span>
              <span className="font-mono text-slate-500">Head {selectedHead + 1} out</span>
            </div>
            <VectorStrip values={head.weightedValues[safeQ]} palette="amber" heightClass="h-6" showNumbers />
          </div>
        </div>
      </div>

      {/* All 12 Heads Mini-Heatmap Gallery */}
      <div className="space-y-2.5">
        <div className="flex items-center justify-between">
          <h4 className="text-sm font-bold text-slate-900">
            Compare All 12 Attention Heads in Block {selectedLayer + 1}
          </h4>
          <span className="text-xs text-slate-500">Click any head to inspect its specialization</span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-2.5">
          {layer.heads.map((h) => {
            const isSelected = h.headIndex === selectedHead;
            return (
              <button
                key={h.headIndex}
                type="button"
                onClick={() => onSelectHead(h.headIndex)}
                className={`text-left p-2.5 rounded-lg border transition-all focus-visible:outline-2 focus-visible:outline-sky-600 ${
                  isSelected
                    ? 'bg-sky-50/70 border-sky-600 ring-1 ring-sky-600/20'
                    : 'bg-white border-slate-200 hover:border-slate-300'
                }`}
              >
                <div className="flex items-center justify-between text-[11px] mb-1.5">
                  <span className="font-mono font-bold text-slate-900 tabular-nums">
                    H{h.headIndex + 1}
                  </span>
                  <span className="text-slate-500 truncate ml-1.5">
                    {HEAD_ROLES[h.headIndex].name}
                  </span>
                </div>

                {/* Tiny lower-triangular heatmap */}
                <div
                  className="grid gap-[1.5px] w-full aspect-square bg-slate-100 p-1 rounded"
                  style={{ gridTemplateColumns: `repeat(${n}, minmax(0, 1fr))` }}
                  aria-hidden="true"
                >
                  {h.attentionWeights.map((row, rIdx) =>
                    row.map((w, cIdx) => (
                      <div
                        key={`${rIdx}-${cIdx}`}
                        style={{
                          backgroundColor:
                            cIdx > rIdx
                              ? '#f1f5f9'
                              : `rgba(2, 132, 199, ${0.1 + w * 0.9})`,
                        }}
                        className="rounded-[1px]"
                      />
                    ))
                  )}
                </div>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
};
