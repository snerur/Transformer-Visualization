import React, { useEffect, useState } from 'react';
import { Dices, Sparkles } from 'lucide-react';
import {
  runSamplingSimulation,
  SamplingMode,
  TransformerRun,
} from '../engine/transformerEngine';

interface SamplingInspectorProps {
  run: TransformerRun;
  temperature: number;
  onTemperatureChange: (t: number) => void;
  samplingMode: SamplingMode;
  onSamplingModeChange: (m: SamplingMode) => void;
  topK: number;
  onTopKChange: (k: number) => void;
  topP: number;
  onTopPChange: (p: number) => void;
  onGenerateToken: () => void;
  onAppendCandidate: (tokenRaw: string) => void;
  canGenerate: boolean;
}

export const SamplingInspector: React.FC<SamplingInspectorProps> = ({
  run,
  temperature,
  onTemperatureChange,
  samplingMode,
  onSamplingModeChange,
  topK,
  onTopKChange,
  topP,
  onTopPChange,
  onGenerateToken,
  onAppendCandidate,
  canGenerate,
}) => {
  const [simCounts, setSimCounts] = useState<Record<string, number> | null>(null);

  // Reset empirical draws whenever distribution parameters change
  useEffect(() => {
    setSimCounts(null);
  }, [run.prompt, temperature, samplingMode, topK, topP]);

  const handleRun100Draws = () => {
    setSimCounts(runSamplingSimulation(run.candidates, 100));
  };

  const eligibleCount = run.candidates.filter((c) => c.included).length;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col lg:flex-row lg:items-end justify-between gap-4">
        <div className="max-w-2xl">
          <h3 className="text-lg font-bold text-slate-900">
            04. Logits, Temperature Scaling &amp; Decoding Strategy
          </h3>
          <p className="text-sm text-slate-600 mt-1 leading-relaxed">
            After the final block, the last token’s vector is projected across all <span className="font-mono text-slate-800">50,257</span> vocabulary words to produce raw <strong>logits</strong>. Temperature reshapes confidence before Softmax converts scores into probabilities, and Top-k or Top-p trims unlikely tail words.
          </p>
        </div>

        <div className="font-serif italic text-sm text-slate-800 bg-slate-50 border border-slate-200 rounded-lg px-3.5 py-2 shrink-0">
          P(w<sub>i</sub>) = exp(z<sub>i</sub> / T) / ∑<sub>j</sub> exp(z<sub>j</sub> / T)
        </div>
      </div>

      {/* Interactive Controls Deck */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 bg-slate-50 border border-slate-200 rounded-xl p-4">
        {/* Temperature Control */}
        <div className="lg:col-span-4 space-y-2">
          <div className="flex items-center justify-between text-xs">
            <label htmlFor="inspector-temp" className="font-semibold text-slate-800">
              Temperature (T)
            </label>
            <span className="font-mono font-bold text-slate-900 tabular-nums">
              {temperature.toFixed(2)} · {temperature <= 0.35 ? 'Deterministic' : temperature <= 1.05 ? 'Balanced' : 'Creative / Flat'}
            </span>
          </div>
          <input
            id="inspector-temp"
            type="range"
            min={0.05}
            max={2.0}
            step={0.05}
            value={temperature}
            onChange={(e) => onTemperatureChange(Number(e.target.value))}
            className="w-full lab-slider"
          />
          <div className="flex justify-between text-[11px] font-mono text-slate-500">
            <span>0.05 (Greedy)</span>
            <span>1.00 (Default)</span>
            <span>2.00 (Diverse)</span>
          </div>
        </div>

        {/* Sampling Mode + Cutoff Slider */}
        <div className="lg:col-span-5 space-y-2">
          <div className="flex items-center justify-between gap-2">
            <div className="flex items-center gap-1 p-0.5 bg-slate-200/80 rounded-lg">
              <button
                type="button"
                onClick={() => onSamplingModeChange('top-k')}
                className={`px-3 py-1 text-xs font-semibold rounded-md transition-colors whitespace-nowrap ${
                  samplingMode === 'top-k'
                    ? 'bg-white text-slate-900 shadow-xs'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                Top-k Sampling
              </button>
              <button
                type="button"
                onClick={() => onSamplingModeChange('top-p')}
                className={`px-3 py-1 text-xs font-semibold rounded-md transition-colors whitespace-nowrap ${
                  samplingMode === 'top-p'
                    ? 'bg-white text-slate-900 shadow-xs'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                Top-p (Nucleus)
              </button>
            </div>

            <span className="font-mono text-xs font-bold text-slate-900 tabular-nums">
              {samplingMode === 'top-k' ? `k = ${topK} tokens` : `p = ${topP.toFixed(2)} (${eligibleCount} kept)`}
            </span>
          </div>

          {samplingMode === 'top-k' ? (
            <>
              <input
                aria-label="Top-k value"
                type="range"
                min={1}
                max={10}
                step={1}
                value={topK}
                onChange={(e) => onTopKChange(Number(e.target.value))}
                className="w-full lab-slider"
              />
              <div className="flex justify-between text-[11px] font-mono text-slate-500">
                <span>k=1 (Argmax)</span>
                <span>k=5</span>
                <span>k=10 (All shown)</span>
              </div>
            </>
          ) : (
            <>
              <input
                aria-label="Top-p nucleus probability threshold"
                type="range"
                min={0.15}
                max={1.0}
                step={0.05}
                value={topP}
                onChange={(e) => onTopPChange(Number(e.target.value))}
                className="w-full lab-slider"
              />
              <div className="flex justify-between text-[11px] font-mono text-slate-500">
                <span>p=0.15 (Tight)</span>
                <span>p=0.90 (Standard)</span>
                <span>p=1.00 (Full)</span>
              </div>
            </>
          )}
        </div>

        {/* Action buttons */}
        <div className="lg:col-span-3 flex flex-col justify-center gap-2">
          <button
            type="button"
            onClick={handleRun100Draws}
            className="w-full inline-flex items-center justify-center gap-2 px-3.5 py-2 text-xs font-semibold text-slate-800 bg-white border border-slate-300 rounded-lg hover:bg-slate-100 transition-colors whitespace-nowrap focus-visible:outline-2 focus-visible:outline-sky-600"
          >
            <Dices className="w-3.5 h-3.5" aria-hidden="true" />
            <span>Run 100 Random Draws</span>
          </button>
          <button
            type="button"
            onClick={onGenerateToken}
            disabled={!canGenerate}
            className="w-full inline-flex items-center justify-center gap-2 px-3.5 py-2 text-xs font-semibold text-white bg-slate-900 rounded-lg hover:bg-slate-800 disabled:opacity-45 disabled:cursor-not-allowed transition-colors whitespace-nowrap focus-visible:outline-2 focus-visible:outline-sky-600"
          >
            <Sparkles className="w-3.5 h-3.5" aria-hidden="true" />
            <span>Sample &amp; Append Token</span>
          </button>
        </div>
      </div>

      {/* Detailed Numeric Transformation Table */}
      <div className="overflow-x-auto border border-slate-200 rounded-xl bg-white">
        <table className="w-full text-left border-collapse text-xs">
          <thead>
            <tr className="border-b border-slate-200 bg-slate-50 text-slate-600 font-semibold">
              <th className="py-2.5 px-3">Rank</th>
              <th className="py-2.5 px-3">Candidate Token</th>
              <th className="py-2.5 px-3 text-right">Raw Logit (z)</th>
              <th className="py-2.5 px-3 text-right">Scaled (z / T)</th>
              <th className="py-2.5 px-3 text-right">Softmax %</th>
              <th className="py-2.5 px-3 text-right">Cumulative</th>
              <th className="py-2.5 px-3">Filter Status</th>
              <th className="py-2.5 px-3 w-44">Final Sampling Chance</th>
              {simCounts && <th className="py-2.5 px-3 text-right">100 Draws</th>}
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {run.candidates.map((cand, idx) => {
              const finalPct = cand.filteredProbability * 100;
              const draws = simCounts ? simCounts[cand.token] || 0 : 0;
              return (
                <tr
                  key={cand.token}
                  className={`transition-colors ${
                    cand.included ? 'hover:bg-slate-50/80' : 'bg-slate-50/50 text-slate-400'
                  }`}
                >
                  <td className="py-2.5 px-3 font-mono tabular-nums text-slate-500">
                    {idx + 1}
                  </td>
                  <td className="py-2.5 px-3">
                    <button
                      type="button"
                      onClick={() => onAppendCandidate(cand.token)}
                      disabled={!canGenerate}
                      className="font-mono font-semibold text-slate-900 hover:text-sky-700 hover:underline underline-offset-2 disabled:no-underline disabled:cursor-not-allowed"
                    >
                      “{cand.token.trim()}”
                    </button>
                    <span className="ml-2 font-mono text-[10px] text-slate-400 tabular-nums">
                      #{cand.tokenId}
                    </span>
                  </td>
                  <td className="py-2.5 px-3 text-right font-mono tabular-nums">
                    {cand.rawLogit.toFixed(2)}
                  </td>
                  <td className="py-2.5 px-3 text-right font-mono tabular-nums">
                    {cand.scaledLogit.toFixed(2)}
                  </td>
                  <td className="py-2.5 px-3 text-right font-mono tabular-nums">
                    {(cand.baseProbability * 100).toFixed(1)}%
                  </td>
                  <td className="py-2.5 px-3 text-right font-mono tabular-nums text-slate-500">
                    {(cand.cumulativeProb * 100).toFixed(1)}%
                  </td>
                  <td className="py-2.5 px-3 font-medium whitespace-nowrap">
                    {cand.included ? (
                      <span className="text-emerald-700">✓ Eligible</span>
                    ) : (
                      <span className="text-slate-400">× Excluded</span>
                    )}
                  </td>
                  <td className="py-2.5 px-3">
                    <div className="flex items-center gap-2">
                      <div className="flex-1 h-2 bg-slate-100 rounded-full overflow-hidden">
                        <div
                          className={`h-full rounded-full transition-all duration-150 ${
                            cand.included ? 'bg-emerald-600' : 'bg-slate-200'
                          }`}
                          style={{ width: `${Math.min(100, finalPct)}%` }}
                        />
                      </div>
                      <span
                        className={`font-mono font-semibold tabular-nums w-11 text-right ${
                          cand.included ? 'text-slate-900' : 'text-slate-400'
                        }`}
                      >
                        {finalPct.toFixed(1)}%
                      </span>
                    </div>
                  </td>
                  {simCounts && (
                    <td className="py-2.5 px-3 text-right font-mono font-bold tabular-nums text-sky-700">
                      {draws} / 100
                    </td>
                  )}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};
