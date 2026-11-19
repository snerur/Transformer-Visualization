import React from 'react';
import { TransformerRun } from '../engine/transformerEngine';
import { VectorStrip } from './VectorStrip';

interface EmbeddingInspectorProps {
  run: TransformerRun;
  selectedTokenIdx: number;
  onSelectToken: (idx: number) => void;
}

export const EmbeddingInspector: React.FC<EmbeddingInspectorProps> = ({
  run,
  selectedTokenIdx,
  onSelectToken,
}) => {
  const safeIdx = Math.min(selectedTokenIdx, run.tokens.length - 1);
  const tok = run.tokens[safeIdx];

  // Build sinusoidal wave curves for 3 sample frequencies across sequence positions
  const maxPos = Math.max(8, run.tokens.length);
  const waves = [
    { dim: 0, label: 'dim 0 (fast sin)', color: '#0284c7', freq: 1 },
    { dim: 2, label: 'dim 2 (mid sin)', color: '#0d9488', freq: 2.2 },
    { dim: 4, label: 'dim 4 (slow sin)', color: '#4f46e5', freq: 4.5 },
  ];

  return (
    <div className="space-y-6">
      {/* Header prose */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div className="max-w-2xl">
          <h3 className="text-lg font-bold text-slate-900">
            01. Tokenization, Vector Lookup &amp; Positional Encoding
          </h3>
          <p className="text-sm text-slate-600 mt-1 leading-relaxed">
            Transformers cannot read raw text directly. The input is split into subword tokens, mapped to learned semantic vectors (<span className="font-mono text-slate-800">768d</span>), and summed element-wise with positional signals so the model knows word order.
          </p>
        </div>
        <div className="font-mono text-xs text-slate-600 bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 tabular-nums shrink-0">
          X[{tok.index}] = E_token(#{tok.tokenId}) + E_pos({tok.index})
        </div>
      </div>

      {/* Main 2-column inspection grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: Element-wise vector addition breakdown */}
        <div className="lg:col-span-7 space-y-4">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-700">
              Inspecting token <span className="font-mono text-sky-700">“{tok.text}”</span> (Position {tok.index}, Vocab ID #{tok.tokenId})
            </span>
            <span className="text-xs text-slate-500 font-mono">Showing dims 0–7 of 768</span>
          </div>

          <div className="space-y-3 bg-slate-50 border border-slate-200 rounded-xl p-4">
            {/* Token Embedding */}
            <div className="space-y-1.5">
              <div className="flex items-center justify-between text-xs">
                <span className="font-semibold text-slate-800">Token Embedding (Meaning)</span>
                <span className="font-mono text-slate-500">W_E[{tok.tokenId}]</span>
              </div>
              <VectorStrip
                values={tok.tokenEmbedding}
                palette="sky"
                heightClass="h-8"
                showNumbers
                label="Token embedding values"
              />
            </div>

            <div className="flex items-center justify-center text-xs font-mono font-bold text-slate-400">
              + element-wise addition
            </div>

            {/* Positional Encoding */}
            <div className="space-y-1.5">
              <div className="flex items-center justify-between text-xs">
                <span className="font-semibold text-slate-800">Positional Encoding (Order)</span>
                <span className="font-mono text-slate-500">PE(pos={tok.index})</span>
              </div>
              <VectorStrip
                values={tok.positionalEncoding}
                palette="teal"
                heightClass="h-8"
                showNumbers
                label="Positional encoding values"
              />
            </div>

            <div className="border-t border-slate-200 pt-3 space-y-1.5">
              <div className="flex items-center justify-between text-xs">
                <span className="font-bold text-slate-900">Combined Input Embedding (Sent to Block 1)</span>
                <span className="font-mono text-slate-600">X[{tok.index}]</span>
              </div>
              <VectorStrip
                values={tok.combinedEmbedding}
                palette="diverging"
                heightClass="h-9"
                showNumbers
                label="Combined embedding values"
              />
            </div>
          </div>

          <p className="text-xs text-slate-500 leading-relaxed">
            Try clicking different tokens above or repeating the same word in your prompt (e.g., <span className="font-mono text-slate-700">“the … the”</span>) to see how identical Token Embeddings become distinct once Positional Encodings are added.
          </p>
        </div>

        {/* Right: Positional Encoding Wave Visualizer */}
        <div className="lg:col-span-5 bg-slate-50 border border-slate-200 rounded-xl p-4 flex flex-col justify-between space-y-4">
          <div>
            <h4 className="text-sm font-bold text-slate-900">
              How Position Waves Encode Order
            </h4>
            <p className="text-xs text-slate-600 mt-1">
              Each embedding dimension oscillates at a different wavelength, creating a unique coordinate signature for every token position.
            </p>
          </div>

          <svg viewBox="0 0 320 155" className="w-full h-40 overflow-visible" aria-label="Positional encoding sinusoidal waves">
            {/* Horizontal zero line */}
            <line x1="24" y1="75" x2="306" y2="75" stroke="#cbd5e1" strokeWidth="1" strokeDasharray="3 3" />
            {/* Axes */}
            <line x1="24" y1="20" x2="24" y2="132" stroke="#94a3b8" strokeWidth="1" />
            <line x1="24" y1="132" x2="306" y2="132" stroke="#94a3b8" strokeWidth="1" />

            {/* Wave paths */}
            {waves.map((w) => {
              const pts: string[] = [];
              for (let s = 0; s <= 60; s++) {
                const pos = (s / 60) * (maxPos - 1);
                const x = 24 + (s / 60) * 282;
                const y = 75 - Math.sin(pos / w.freq) * 44;
                pts.push(`${x.toFixed(1)},${y.toFixed(1)}`);
              }
              return (
                <polyline
                  key={w.dim}
                  fill="none"
                  stroke={w.color}
                  strokeWidth="2"
                  strokeOpacity="0.82"
                  points={pts.join(' ')}
                />
              );
            })}

            {/* Vertical marker for selected token position */}
            {(() => {
              const xPos = 24 + (safeIdx / Math.max(1, maxPos - 1)) * 282;
              return (
                <g>
                  <line x1={xPos} y1="18" x2={xPos} y2="132" stroke="#0f172a" strokeWidth="1.5" strokeDasharray="2 2" />
                  {waves.map((w) => {
                    const y = 75 - Math.sin(safeIdx / w.freq) * 44;
                    return (
                      <circle
                        key={`pt-${w.dim}`}
                        cx={xPos}
                        cy={y}
                        r="4"
                        fill={w.color}
                        stroke="#ffffff"
                        strokeWidth="1.5"
                      />
                    );
                  })}
                  <text
                    x={xPos}
                    y="12"
                    textAnchor="middle"
                    className="fill-slate-900 text-[10px] font-mono font-semibold"
                  >
                    pos {safeIdx}
                  </text>
                </g>
              );
            })()}

            {/* Clickable token position ticks */}
            {run.tokens.map((t) => {
              const tx = 24 + (t.index / Math.max(1, maxPos - 1)) * 282;
              return (
                <g
                  key={`tick-${t.index}`}
                  onClick={() => onSelectToken(t.index)}
                  className="cursor-pointer"
                >
                  <line x1={tx} y1="132" x2={tx} y2="137" stroke="#64748b" strokeWidth="1" />
                  <text
                    x={tx}
                    y="148"
                    textAnchor="middle"
                    className={`text-[10px] font-mono ${
                      t.index === safeIdx ? 'fill-sky-700 font-bold' : 'fill-slate-500'
                    }`}
                  >
                    {t.index}
                  </text>
                </g>
              );
            })}
          </svg>

          {/* Clean unboxed legend */}
          <div className="flex items-center flex-wrap gap-x-4 gap-y-1 text-[11px] text-slate-600 font-mono">
            {waves.map((w) => (
              <span key={w.dim} className="inline-flex items-center gap-1.5">
                <span className="w-3 h-0.5 inline-block rounded-full" style={{ backgroundColor: w.color }} />
                <span>{w.label}</span>
              </span>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};
