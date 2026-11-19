import React, { useState } from 'react';
import { ArrowRight, ChevronRight } from 'lucide-react';
import {
  SamplingMode,
  StageId,
  TransformerRun,
} from '../engine/transformerEngine';
import { VectorStrip } from './VectorStrip';

interface PipelineCanvasProps {
  run: TransformerRun;
  activeStage: StageId;
  onSelectStage: (stage: StageId) => void;
  selectedTokenIdx: number;
  onSelectToken: (idx: number) => void;
  selectedLayer: number;
  onSelectLayer: (layer: number) => void;
  selectedHead: number;
  onSelectHead: (head: number) => void;
  temperature: number;
  onTemperatureChange: (t: number) => void;
  samplingMode: SamplingMode;
  onSamplingModeChange: (m: SamplingMode) => void;
  topK: number;
  topP: number;
  onAppendCandidate: (tokenRaw: string) => void;
}

export const PipelineCanvas: React.FC<PipelineCanvasProps> = ({
  run,
  activeStage,
  onSelectStage,
  selectedTokenIdx,
  onSelectToken,
  selectedLayer,
  onSelectLayer,
  selectedHead,
  onSelectHead,
  temperature,
  onTemperatureChange,
  samplingMode,
  onSamplingModeChange,
  topK,
  topP,
  onAppendCandidate,
}) => {
  const [hoveredCell, setHoveredCell] = useState<{ q: number; k: number } | null>(null);

  const layer = run.layers[selectedLayer] ?? run.layers[0];
  const head = layer.heads[selectedHead] ?? layer.heads[0];
  const safeTokenIdx = Math.min(selectedTokenIdx, run.tokens.length - 1);
  const activeMlp = layer.mlpStates[safeTokenIdx] ?? layer.mlpStates[0];

  const topCandidates = run.candidates.slice(0, 7);

  // Build SVG path points for mini GELU preview
  const geluPoints: string[] = [];
  for (let i = 0; i <= 32; i++) {
    const xVal = -3 + (i / 32) * 6;
    const c = Math.sqrt(2 / Math.PI);
    const yVal = 0.5 * xVal * (1 + Math.tanh(c * (xVal + 0.044715 * xVal ** 3)));
    const sx = 8 + (i / 32) * 144;
    const sy = 56 - ((yVal + 0.4) / 3.6) * 48;
    geluPoints.push(`${sx.toFixed(1)},${sy.toFixed(1)}`);
  }

  return (
    <section id="pipeline" aria-label="Transformer Architecture Pipeline" className="space-y-3">
      {/* Architecture header strip (unboxed quiet metadata) */}
      <div className="flex flex-wrap items-baseline justify-between gap-3">
        <div>
          <h2 className="text-lg sm:text-xl font-bold text-slate-900 tracking-tight">
            Interactive Forward Pass
          </h2>
          <p className="text-sm text-slate-600 mt-0.5">
            Click any stage to inspect its math below, or hover cells in the attention matrix to trace token relationships.
          </p>
        </div>
        <div className="text-xs text-slate-500 font-mono tabular-nums flex items-center gap-2 flex-wrap">
          <span>GPT-2 Small</span>
          <span aria-hidden="true">·</span>
          <span>12 Layers</span>
          <span aria-hidden="true">·</span>
          <span>12 Heads</span>
          <span aria-hidden="true">·</span>
          <span>768 Dim</span>
          <span aria-hidden="true">·</span>
          <span>50,257 Vocab</span>
        </div>
      </div>

      {/* Main 4-stage visual pipeline grid */}
      <div className="grid grid-cols-1 xl:grid-cols-12 gap-4 items-stretch">
        {/* STAGE 1: EMBEDDING */}
        <div
          role="button"
          tabIndex={0}
          onClick={() => onSelectStage('embedding')}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault();
              onSelectStage('embedding');
            }
          }}
          className={`xl:col-span-3 bg-white rounded-xl p-4 border transition-all cursor-pointer flex flex-col justify-between focus-visible:outline-2 focus-visible:outline-sky-600 ${
            activeStage === 'embedding'
              ? 'border-sky-600 ring-2 ring-sky-600/15'
              : 'border-slate-200 hover:border-slate-300'
          }`}
        >
          <div className="space-y-3">
            <div className="flex items-center justify-between gap-2">
              <div className="flex items-center gap-2">
                <span className="font-mono text-xs font-semibold text-sky-700 tabular-nums">01</span>
                <h3 className="text-sm font-bold text-slate-900">Embedding</h3>
              </div>
              <span className="text-[11px] font-mono text-slate-400 tabular-nums">
                {run.tokens.length} × 768
              </span>
            </div>

            <p className="text-xs text-slate-600 leading-relaxed">
              Each token maps to a numeric vector and adds its sequence position (<span className="font-mono text-slate-800">E_tok + E_pos</span>).
            </p>

            {/* Token embedding rows */}
            <div className="space-y-2 pt-1">
              {run.tokens.map((tok) => {
                const isSelected = tok.index === safeTokenIdx;
                return (
                  <div
                    key={tok.index}
                    onClick={(e) => {
                      e.stopPropagation();
                      onSelectToken(tok.index);
                      onSelectStage('embedding');
                    }}
                    className={`p-2 rounded-lg border transition-colors ${
                      isSelected
                        ? 'bg-sky-50/70 border-sky-300'
                        : 'bg-slate-50/70 border-slate-100 hover:border-slate-200'
                    }`}
                  >
                    <div className="flex items-center justify-between text-xs mb-1.5">
                      <span className={`font-mono font-semibold truncate ${isSelected ? 'text-sky-900' : 'text-slate-800'}`}>
                        {tok.text}
                      </span>
                      <span className="font-mono text-[10px] text-slate-400 tabular-nums shrink-0 ml-2">
                        #{tok.tokenId} · p{tok.index}
                      </span>
                    </div>
                    <VectorStrip
                      values={tok.combinedEmbedding}
                      palette="diverging"
                      heightClass="h-3"
                      label={`Combined embedding for ${tok.text}`}
                    />
                  </div>
                );
              })}
            </div>
          </div>

          <div className="pt-3 mt-3 border-t border-slate-100 flex items-center justify-between text-xs font-medium text-sky-700">
            <span>Inspect embeddings</span>
            <ChevronRight className="w-3.5 h-3.5" aria-hidden="true" />
          </div>
        </div>

        {/* STAGE 2: MULTI-HEAD MASKED SELF-ATTENTION */}
        <div
          role="button"
          tabIndex={0}
          onClick={() => onSelectStage('attention')}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault();
              onSelectStage('attention');
            }
          }}
          className={`xl:col-span-4 bg-white rounded-xl p-4 border transition-all cursor-pointer flex flex-col justify-between focus-visible:outline-2 focus-visible:outline-sky-600 ${
            activeStage === 'attention'
              ? 'border-sky-600 ring-2 ring-sky-600/15'
              : 'border-slate-200 hover:border-slate-300'
          }`}
        >
          <div className="space-y-3">
            <div className="flex items-center justify-between gap-2">
              <div className="flex items-center gap-2">
                <span className="font-mono text-xs font-semibold text-teal-700 tabular-nums">02</span>
                <h3 className="text-sm font-bold text-slate-900">Masked Self-Attention</h3>
              </div>

              {/* Compact Layer & Head Controls */}
              <div
                className="flex items-center gap-1.5"
                onClick={(e) => e.stopPropagation()}
              >
                <label className="sr-only" htmlFor="pipeline-layer-select">Layer</label>
                <select
                  id="pipeline-layer-select"
                  value={selectedLayer}
                  onChange={(e) => onSelectLayer(Number(e.target.value))}
                  className="text-xs font-mono bg-slate-100 hover:bg-slate-200/80 text-slate-800 rounded-md px-2 py-1 border border-slate-200 focus:outline-none focus:border-sky-600 cursor-pointer"
                >
                  {Array.from({ length: 12 }, (_, i) => (
                    <option key={i} value={i}>
                      Block {i + 1}
                    </option>
                  ))}
                </select>
                <label className="sr-only" htmlFor="pipeline-head-select">Attention Head</label>
                <select
                  id="pipeline-head-select"
                  value={selectedHead}
                  onChange={(e) => onSelectHead(Number(e.target.value))}
                  className="text-xs font-mono bg-slate-100 hover:bg-slate-200/80 text-slate-800 rounded-md px-2 py-1 border border-slate-200 focus:outline-none focus:border-sky-600 cursor-pointer"
                >
                  {Array.from({ length: 12 }, (_, i) => (
                    <option key={i} value={i}>
                      Head {i + 1}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {/* Q / K / V projection preview for active token */}
            <div className="grid grid-cols-3 gap-2 pt-0.5">
              <div className="space-y-1">
                <div className="flex items-center justify-between text-[11px]">
                  <span className="font-semibold text-sky-700">Query (Q)</span>
                  <span className="font-mono text-slate-400">64d</span>
                </div>
                <VectorStrip values={head.queries[safeTokenIdx]} palette="sky" heightClass="h-3" />
              </div>
              <div className="space-y-1">
                <div className="flex items-center justify-between text-[11px]">
                  <span className="font-semibold text-teal-700">Key (K)</span>
                  <span className="font-mono text-slate-400">64d</span>
                </div>
                <VectorStrip values={head.keys[safeTokenIdx]} palette="teal" heightClass="h-3" />
              </div>
              <div className="space-y-1">
                <div className="flex items-center justify-between text-[11px]">
                  <span className="font-semibold text-amber-700">Value (V)</span>
                  <span className="font-mono text-slate-400">64d</span>
                </div>
                <VectorStrip values={head.values[safeTokenIdx]} palette="amber" heightClass="h-3" />
              </div>
            </div>

            {/* Interactive Causal Attention Heatmap Matrix */}
            <div className="pt-1">
              <div className="flex items-center justify-between text-[11px] text-slate-500 mb-1.5">
                <span>
                  Pattern: <strong className="text-slate-800 font-semibold">{head.roleName}</strong>
                </span>
                <span className="font-mono tabular-nums">
                  {hoveredCell
                    ? hoveredCell.k > hoveredCell.q
                      ? `Masked (−∞ → 0%)`
                      : `${run.tokens[hoveredCell.q]?.text} → ${run.tokens[hoveredCell.k]?.text}: ${(head.attentionWeights[hoveredCell.q][hoveredCell.k] * 100).toFixed(1)}%`
                    : 'Hover matrix cells'}
                </span>
              </div>

              <div className="overflow-x-auto pb-1">
                <div
                  className="grid gap-1 items-center"
                  style={{
                    gridTemplateColumns: `64px repeat(${run.tokens.length}, minmax(22px, 1fr))`,
                  }}
                >
                  {/* Column header labels (Keys) */}
                  <div className="text-[10px] font-mono text-slate-400 text-right pr-1">Q \ K</div>
                  {run.tokens.map((kTok) => (
                    <div
                      key={`col-${kTok.index}`}
                      className={`text-[10px] font-mono truncate text-center ${
                        hoveredCell?.k === kTok.index ? 'text-teal-700 font-semibold' : 'text-slate-500'
                      }`}
                      title={`Key: ${kTok.text}`}
                    >
                      {kTok.text.slice(0, 4)}
                    </div>
                  ))}

                  {/* Rows (Queries) */}
                  {run.tokens.map((qTok, qIdx) => (
                    <React.Fragment key={`row-${qIdx}`}>
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          onSelectToken(qIdx);
                        }}
                        className={`text-[11px] font-mono truncate text-right pr-1.5 py-0.5 rounded transition-colors ${
                          qIdx === safeTokenIdx
                            ? 'text-sky-700 font-semibold bg-sky-50'
                            : 'text-slate-600 hover:text-slate-900'
                        }`}
                        title={`Focus Query token: ${qTok.text}`}
                      >
                        {qTok.text}
                      </button>

                      {run.tokens.map((kTok, kIdx) => {
                        const isMasked = kIdx > qIdx;
                        const weight = head.attentionWeights[qIdx][kIdx];
                        const isHighlighted =
                          hoveredCell?.q === qIdx && hoveredCell?.k === kIdx;
                        const isActiveRow = qIdx === safeTokenIdx;

                        return (
                          <div
                            key={`cell-${qIdx}-${kIdx}`}
                            onMouseEnter={() => setHoveredCell({ q: qIdx, k: kIdx })}
                            onMouseLeave={() => setHoveredCell(null)}
                            onClick={(e) => {
                              e.stopPropagation();
                              onSelectToken(qIdx);
                              onSelectStage('attention');
                            }}
                            style={{
                              backgroundColor: isMasked
                                ? '#f8fafc'
                                : `rgba(2, 132, 199, ${0.08 + weight * 0.88})`,
                            }}
                            className={`h-6 rounded-[4px] flex items-center justify-center text-[10px] font-mono tabular-nums transition-transform duration-100 select-none ${
                              isMasked
                                ? 'border border-dashed border-slate-200 text-slate-300'
                                : weight > 0.45
                                ? 'text-white font-semibold'
                                : 'text-slate-800'
                            } ${
                              isHighlighted
                                ? 'ring-2 ring-slate-900 scale-105 z-10'
                                : isActiveRow && !isMasked
                                ? 'ring-1 ring-sky-400/60'
                                : ''
                            }`}
                          >
                            {isMasked ? '×' : Math.round(weight * 100)}
                          </div>
                        );
                      })}
                    </React.Fragment>
                  ))}
                </div>
              </div>
            </div>
          </div>

          <div className="pt-3 mt-3 border-t border-slate-100 flex items-center justify-between text-xs font-medium text-teal-700">
            <span>Inspect Q·K·V &amp; all 12 heads</span>
            <ChevronRight className="w-3.5 h-3.5" aria-hidden="true" />
          </div>
        </div>

        {/* STAGE 3: MLP & RESIDUAL STREAM */}
        <div
          role="button"
          tabIndex={0}
          onClick={() => onSelectStage('mlp')}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault();
              onSelectStage('mlp');
            }
          }}
          className={`xl:col-span-2 bg-white rounded-xl p-4 border transition-all cursor-pointer flex flex-col justify-between focus-visible:outline-2 focus-visible:outline-sky-600 ${
            activeStage === 'mlp'
              ? 'border-sky-600 ring-2 ring-sky-600/15'
              : 'border-slate-200 hover:border-slate-300'
          }`}
        >
          <div className="space-y-3">
            <div className="flex items-center justify-between gap-2">
              <div className="flex items-center gap-2">
                <span className="font-mono text-xs font-semibold text-indigo-700 tabular-nums">03</span>
                <h3 className="text-sm font-bold text-slate-900">MLP &amp; Norm</h3>
              </div>
            </div>

            <p className="text-xs text-slate-600 leading-relaxed">
              Refines each token independently via expansion, GELU, and residual addition.
            </p>

            {/* Vertical flow schematic */}
            <div className="space-y-2 pt-1">
              <div className="px-2.5 py-2 rounded-lg bg-slate-50 border border-slate-200 text-xs">
                <div className="flex items-center justify-between font-medium text-slate-800">
                  <span>Add &amp; LayerNorm</span>
                  <span className="font-mono text-[10px] text-slate-500">x + Attn</span>
                </div>
                <div className="mt-1.5">
                  <VectorStrip values={activeMlp.layerNormOut} palette="diverging" heightClass="h-2.5" />
                </div>
              </div>

              <div className="flex justify-center text-slate-300" aria-hidden="true">
                <ArrowRight className="w-3.5 h-3.5 rotate-90" />
              </div>

              {/* Mini GELU curve preview */}
              <div className="px-2.5 py-2 rounded-lg bg-indigo-50/50 border border-indigo-200/80 space-y-1.5">
                <div className="flex items-center justify-between text-[11px]">
                  <span className="font-semibold text-indigo-900">Expand + GELU</span>
                  <span className="font-mono text-[10px] text-indigo-700 tabular-nums">768→3072</span>
                </div>
                <svg viewBox="0 0 160 64" className="w-full h-14 overflow-visible" aria-label="GELU activation function preview">
                  <line x1="8" y1="50" x2="152" y2="50" stroke="#cbd5e1" strokeWidth="1" />
                  <line x1="80" y1="6" x2="80" y2="58" stroke="#cbd5e1" strokeWidth="1" />
                  <polyline
                    fill="none"
                    stroke="#4f46e5"
                    strokeWidth="2"
                    strokeLinecap="round"
                    points={geluPoints.join(' ')}
                  />
                  {activeMlp.expandedPreGelu.slice(0, 6).map((xVal, i) => {
                    const yVal = activeMlp.expandedPostGelu[i];
                    const cx = 8 + ((xVal + 3) / 6) * 144;
                    const cy = 56 - ((yVal + 0.4) / 3.6) * 48;
                    return (
                      <circle
                        key={i}
                        cx={cx}
                        cy={cy}
                        r="3"
                        fill="#4f46e5"
                        stroke="#ffffff"
                        strokeWidth="1.2"
                      />
                    );
                  })}
                </svg>
              </div>

              <div className="flex justify-center text-slate-300" aria-hidden="true">
                <ArrowRight className="w-3.5 h-3.5 rotate-90" />
              </div>

              <div className="px-2.5 py-2 rounded-lg bg-slate-50 border border-slate-200 text-xs">
                <div className="flex items-center justify-between font-medium text-slate-800">
                  <span>Project + Residual</span>
                  <span className="font-mono text-[10px] text-slate-500">×12 Blocks</span>
                </div>
                <div className="mt-1.5">
                  <VectorStrip values={activeMlp.finalBlockOutput} palette="indigo" heightClass="h-2.5" />
                </div>
              </div>
            </div>
          </div>

          <div className="pt-3 mt-3 border-t border-slate-100 flex items-center justify-between text-xs font-medium text-indigo-700">
            <span>Inspect GELU &amp; Norm</span>
            <ChevronRight className="w-3.5 h-3.5" aria-hidden="true" />
          </div>
        </div>

        {/* STAGE 4: OUTPUT PROBABILITIES & SAMPLING */}
        <div
          role="button"
          tabIndex={0}
          onClick={() => onSelectStage('sampling')}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault();
              onSelectStage('sampling');
            }
          }}
          className={`xl:col-span-3 bg-white rounded-xl p-4 border transition-all cursor-pointer flex flex-col justify-between focus-visible:outline-2 focus-visible:outline-sky-600 ${
            activeStage === 'sampling'
              ? 'border-sky-600 ring-2 ring-sky-600/15'
              : 'border-slate-200 hover:border-slate-300'
          }`}
        >
          <div className="space-y-3">
            <div className="flex items-center justify-between gap-2">
              <div className="flex items-center gap-2">
                <span className="font-mono text-xs font-semibold text-emerald-700 tabular-nums">04</span>
                <h3 className="text-sm font-bold text-slate-900">Next-Token Probabilities</h3>
              </div>
            </div>

            {/* Inline quick temperature & strategy controls */}
            <div
              className="p-2.5 rounded-lg bg-slate-50 border border-slate-200 space-y-2"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center justify-between text-xs">
                <label htmlFor="quick-temp" className="font-medium text-slate-700">
                  Temperature
                </label>
                <span className="font-mono font-semibold text-slate-900 tabular-nums">
                  {temperature.toFixed(2)}
                </span>
              </div>
              <input
                id="quick-temp"
                type="range"
                min={0.05}
                max={2.0}
                step={0.05}
                value={temperature}
                onChange={(e) => onTemperatureChange(Number(e.target.value))}
                className="w-full lab-slider"
              />
              <div className="flex items-center justify-between gap-2 pt-0.5">
                <div className="flex items-center gap-1 bg-slate-200/70 p-0.5 rounded-md">
                  <button
                    type="button"
                    onClick={() => onSamplingModeChange('top-k')}
                    className={`px-2 py-0.5 text-[11px] font-medium rounded transition-colors whitespace-nowrap ${
                      samplingMode === 'top-k'
                        ? 'bg-white text-slate-900 shadow-xs'
                        : 'text-slate-600 hover:text-slate-900'
                    }`}
                  >
                    Top-k ({topK})
                  </button>
                  <button
                    type="button"
                    onClick={() => onSamplingModeChange('top-p')}
                    className={`px-2 py-0.5 text-[11px] font-medium rounded transition-colors whitespace-nowrap ${
                      samplingMode === 'top-p'
                        ? 'bg-white text-slate-900 shadow-xs'
                        : 'text-slate-600 hover:text-slate-900'
                    }`}
                  >
                    Top-p ({topP.toFixed(2)})
                  </button>
                </div>
                <span className="text-[11px] text-slate-500">Click token to append</span>
              </div>
            </div>

            {/* Probability bars */}
            <div className="space-y-1.5" onClick={(e) => e.stopPropagation()}>
              {topCandidates.map((cand) => {
                const pct = cand.included
                  ? cand.filteredProbability * 100
                  : cand.baseProbability * 100;
                return (
                  <button
                    key={cand.token}
                    type="button"
                    onClick={() => onAppendCandidate(cand.token)}
                    disabled={run.tokens.length >= 12}
                    title={`Append "${cand.token.trim()}" to prompt (logit ${cand.rawLogit.toFixed(2)})`}
                    className={`w-full text-left px-2 py-1.5 rounded-md border transition-colors group disabled:cursor-not-allowed ${
                      cand.included
                        ? 'bg-white border-slate-200 hover:border-emerald-400 hover:bg-emerald-50/40'
                        : 'bg-slate-50/60 border-slate-100 opacity-55 hover:opacity-80'
                    }`}
                  >
                    <div className="flex items-center justify-between text-xs mb-1">
                      <span className="font-mono font-medium text-slate-900 truncate">
                        {cand.token.trim()}
                      </span>
                      <div className="flex items-center gap-2 shrink-0 ml-2">
                        <span className="text-[10px] font-medium text-slate-500">
                          {cand.included ? 'Eligible' : 'Filtered'}
                        </span>
                        <span
                          className={`font-mono text-xs font-semibold tabular-nums ${
                            cand.included ? 'text-emerald-700' : 'text-slate-400 line-through'
                          }`}
                        >
                          {pct.toFixed(1)}%
                        </span>
                      </div>
                    </div>
                    <div className="h-1.5 w-full bg-slate-100 rounded-full overflow-hidden">
                      <div
                        className={`h-full rounded-full transition-all duration-150 ${
                          cand.included ? 'bg-emerald-600' : 'bg-slate-300'
                        }`}
                        style={{ width: `${Math.max(2, Math.min(100, pct))}%` }}
                      />
                    </div>
                  </button>
                );
              })}
            </div>
          </div>

          <div className="pt-3 mt-3 border-t border-slate-100 flex items-center justify-between text-xs font-medium text-emerald-700">
            <span>Inspect softmax &amp; sampling</span>
            <ChevronRight className="w-3.5 h-3.5" aria-hidden="true" />
          </div>
        </div>
      </div>
    </section>
  );
};
