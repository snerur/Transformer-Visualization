/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useEffect, useMemo, useState } from 'react';
import { Undo2 } from 'lucide-react';
import {
  PRESET_PROMPTS,
  runTransformerForwardPass,
  sampleNextToken,
  SamplingMode,
  StageId,
} from './engine/transformerEngine';
import { Header } from './components/Header';
import { PromptBar } from './components/PromptBar';
import { PipelineCanvas } from './components/PipelineCanvas';
import { EmbeddingInspector } from './components/EmbeddingInspector';
import { AttentionInspector } from './components/AttentionInspector';
import { MlpInspector } from './components/MlpInspector';
import { SamplingInspector } from './components/SamplingInspector';
import { CitationSection } from './components/CitationSection';

const STAGE_ORDER: StageId[] = ['embedding', 'attention', 'mlp', 'sampling'];

const STAGE_TABS: { id: StageId; num: string; label: string }[] = [
  { id: 'embedding', num: '01', label: 'Embeddings' },
  { id: 'attention', num: '02', label: 'Self-Attention' },
  { id: 'mlp', num: '03', label: 'MLP & Residuals' },
  { id: 'sampling', num: '04', label: 'Softmax & Sampling' },
];

export default function App() {
  const [prompt, setPrompt] = useState<string>(PRESET_PROMPTS[0].text);
  const [promptHistory, setPromptHistory] = useState<string[]>([]);
  const [lastGeneratedToken, setLastGeneratedToken] = useState<string | null>(null);

  const [activeStage, setActiveStage] = useState<StageId>('attention');
  const [isPlaying, setIsPlaying] = useState<boolean>(false);

  const [selectedTokenIdx, setSelectedTokenIdx] = useState<number>(4);
  const [selectedLayer, setSelectedLayer] = useState<number>(0);
  const [selectedHead, setSelectedHead] = useState<number>(1);

  const [temperature, setTemperature] = useState<number>(0.8);
  const [samplingMode, setSamplingMode] = useState<SamplingMode>('top-p');
  const [topK, setTopK] = useState<number>(5);
  const [topP, setTopP] = useState<number>(0.9);

  // Run full forward pass whenever prompt or decoding hyperparameters change
  const run = useMemo(
    () => runTransformerForwardPass(prompt, temperature, samplingMode, topK, topP),
    [prompt, temperature, samplingMode, topK, topP]
  );

  // Keep selectedTokenIdx within valid bounds when prompt length changes
  const safeTokenIdx = Math.min(selectedTokenIdx, Math.max(0, run.tokens.length - 1));

  // Guided walkthrough autoplay timer
  useEffect(() => {
    if (!isPlaying) return;
    const timer = window.setInterval(() => {
      setActiveStage((prev) => {
        const idx = STAGE_ORDER.indexOf(prev);
        return STAGE_ORDER[(idx + 1) % STAGE_ORDER.length];
      });
    }, 2800);
    return () => window.clearInterval(timer);
  }, [isPlaying]);

  const handleSelectStage = (stage: StageId) => {
    setIsPlaying(false);
    setActiveStage(stage);
  };

  const handleStepBack = () => {
    setIsPlaying(false);
    setActiveStage((prev) => {
      const idx = STAGE_ORDER.indexOf(prev);
      return STAGE_ORDER[(idx - 1 + STAGE_ORDER.length) % STAGE_ORDER.length];
    });
  };

  const handleStepForward = () => {
    setIsPlaying(false);
    setActiveStage((prev) => {
      const idx = STAGE_ORDER.indexOf(prev);
      return STAGE_ORDER[(idx + 1) % STAGE_ORDER.length];
    });
  };

  const handlePromptChange = (val: string) => {
    setPrompt(val);
    setLastGeneratedToken(null);
  };

  const handleSelectPreset = (text: string) => {
    setPromptHistory([]);
    setPrompt(text);
    setLastGeneratedToken(null);
    // Focus the last token of the preset so causal attention shows full sentence context
    const wordCount = text.trim().split(/\s+/).length;
    setSelectedTokenIdx(Math.max(0, wordCount - 1));
  };

  const handleResetPrompt = () => {
    setPromptHistory([]);
    setPrompt(PRESET_PROMPTS[0].text);
    setLastGeneratedToken(null);
    setSelectedTokenIdx(4);
    setTemperature(0.8);
    setSamplingMode('top-p');
    setTopK(5);
    setTopP(0.9);
  };

  const appendToken = (tokenRaw: string) => {
    if (run.tokens.length >= 12) return;
    setPromptHistory((prev) => [...prev.slice(-8), prompt]);
    const nextPrompt = `${prompt.trimEnd()}${tokenRaw}`;
    setPrompt(nextPrompt);
    setLastGeneratedToken(tokenRaw);
    setSelectedTokenIdx(Math.min(11, run.tokens.length));
  };

  const handleGenerateToken = () => {
    if (run.tokens.length >= 12) return;
    const sampled = sampleNextToken(run.candidates);
    appendToken(sampled.token);
  };

  const handleUndoGenerate = () => {
    if (promptHistory.length === 0) return;
    const previous = promptHistory[promptHistory.length - 1];
    setPromptHistory((prev) => prev.slice(0, -1));
    setPrompt(previous);
    setLastGeneratedToken(null);
  };

  const canGenerate = run.tokens.length < 12;

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 text-slate-900">
      <Header
        onSelectStage={handleSelectStage}
        onGenerateToken={handleGenerateToken}
        canGenerate={canGenerate}
      />

      <main className="flex-1 max-w-[1400px] w-full mx-auto px-4 sm:px-6 py-6 sm:py-8 space-y-8">
        {/* Compact Editorial Intro */}
        <section className="flex flex-col lg:flex-row lg:items-end justify-between gap-4">
          <div className="max-w-3xl space-y-1.5">
            <div className="text-xs text-slate-500 flex items-center gap-2 flex-wrap">
              <span>Interactive Architecture Guide</span>
              <span aria-hidden="true">·</span>
              <span>Inspired by Cho et al., ACM CHI 2026</span>
              <span aria-hidden="true">·</span>
              <span>Decoder-Only Transformer (GPT-2)</span>
            </div>
            <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-slate-900">
              How a Transformer predicts the next word
            </h1>
            <p className="text-sm sm:text-base text-slate-600 leading-relaxed">
              Edit the sentence, switch attention layers and heads, or adjust temperature to watch raw text transform into contextual vectors and next-token probabilities in real time.
            </p>
          </div>

          {lastGeneratedToken && promptHistory.length > 0 && (
            <div className="flex items-center gap-3 bg-emerald-50 border border-emerald-200 rounded-lg px-3.5 py-2 text-xs text-emerald-900 shrink-0">
              <span>
                Appended <strong className="font-mono">“{lastGeneratedToken.trim()}”</strong>
              </span>
              <button
                type="button"
                onClick={handleUndoGenerate}
                className="inline-flex items-center gap-1 font-semibold text-emerald-800 hover:text-emerald-950 underline underline-offset-2 whitespace-nowrap"
              >
                <Undo2 className="w-3.5 h-3.5" aria-hidden="true" />
                <span>Undo</span>
              </button>
            </div>
          )}
        </section>

        {/* Prompt Input, Token Focus, and Walkthrough Bar */}
        <PromptBar
          prompt={prompt}
          onPromptChange={handlePromptChange}
          onSelectPreset={handleSelectPreset}
          onResetPrompt={handleResetPrompt}
          tokens={run.tokens}
          selectedTokenIdx={safeTokenIdx}
          onSelectToken={setSelectedTokenIdx}
          activeStage={activeStage}
          isPlaying={isPlaying}
          onTogglePlay={() => setIsPlaying((p) => !p)}
          onStepBack={handleStepBack}
          onStepForward={handleStepForward}
          lastGeneratedToken={lastGeneratedToken}
        />

        {/* Macro Focal Stage: End-to-End 4-Stage Pipeline */}
        <PipelineCanvas
          run={run}
          activeStage={activeStage}
          onSelectStage={handleSelectStage}
          selectedTokenIdx={safeTokenIdx}
          onSelectToken={setSelectedTokenIdx}
          selectedLayer={selectedLayer}
          onSelectLayer={setSelectedLayer}
          selectedHead={selectedHead}
          onSelectHead={setSelectedHead}
          temperature={temperature}
          onTemperatureChange={setTemperature}
          samplingMode={samplingMode}
          onSamplingModeChange={setSamplingMode}
          topK={topK}
          topP={topP}
          onAppendCandidate={appendToken}
        />

        {/* Micro Deep-Dive Inspector Deck */}
        <section
          id="inspector"
          aria-label="Detailed stage mathematics inspector"
          className="bg-white border border-slate-200 rounded-xl overflow-hidden"
        >
          {/* Stage Tabs Header */}
          <div className="border-b border-slate-200 bg-slate-50/70 px-4 sm:px-6 pt-3 flex items-center justify-between gap-4 overflow-x-auto">
            <div className="flex items-center gap-1" role="tablist" aria-label="Architecture stages">
              {STAGE_TABS.map((tab) => {
                const isActive = activeStage === tab.id;
                return (
                  <button
                    key={tab.id}
                    type="button"
                    role="tab"
                    aria-selected={isActive}
                    onClick={() => handleSelectStage(tab.id)}
                    className={`px-3.5 py-2.5 text-xs sm:text-sm font-semibold border-b-2 transition-colors whitespace-nowrap shrink-0 focus-visible:outline-2 focus-visible:outline-sky-600 ${
                      isActive
                        ? 'border-sky-600 text-slate-900 bg-white rounded-t-lg'
                        : 'border-transparent text-slate-500 hover:text-slate-800'
                    }`}
                  >
                    <span className="font-mono text-xs text-slate-400 mr-1.5 tabular-nums">
                      {tab.num}
                    </span>
                    <span>{tab.label}</span>
                  </button>
                );
              })}
            </div>

            <div className="hidden md:block text-xs text-slate-500 pb-2 whitespace-nowrap shrink-0">
              Focused Token:{' '}
              <span className="font-mono font-semibold text-slate-900">
                “{run.tokens[safeTokenIdx]?.text}”
              </span>{' '}
              <span className="text-slate-400">·</span> Block {selectedLayer + 1} · Head {selectedHead + 1}
            </div>
          </div>

          {/* Stage Inspector Content */}
          <div className="p-5 sm:p-6">
            {activeStage === 'embedding' && (
              <EmbeddingInspector
                run={run}
                selectedTokenIdx={safeTokenIdx}
                onSelectToken={setSelectedTokenIdx}
              />
            )}
            {activeStage === 'attention' && (
              <AttentionInspector
                run={run}
                selectedTokenIdx={safeTokenIdx}
                onSelectToken={setSelectedTokenIdx}
                selectedLayer={selectedLayer}
                onSelectLayer={setSelectedLayer}
                selectedHead={selectedHead}
                onSelectHead={setSelectedHead}
              />
            )}
            {activeStage === 'mlp' && (
              <MlpInspector
                run={run}
                selectedTokenIdx={safeTokenIdx}
                selectedLayer={selectedLayer}
              />
            )}
            {activeStage === 'sampling' && (
              <SamplingInspector
                run={run}
                temperature={temperature}
                onTemperatureChange={setTemperature}
                samplingMode={samplingMode}
                onSamplingModeChange={setSamplingMode}
                topK={topK}
                onTopKChange={setTopK}
                topP={topP}
                onTopPChange={setTopP}
                onGenerateToken={handleGenerateToken}
                onAppendCandidate={appendToken}
                canGenerate={canGenerate}
              />
            )}
          </div>
        </section>

        {/* Source Paper Acknowledgment */}
        <CitationSection />
      </main>

      {/* Quiet Footer */}
      <footer className="border-t border-slate-200 bg-white mt-8">
        <div className="max-w-[1400px] mx-auto px-4 sm:px-6 py-5 flex flex-col sm:flex-row items-center justify-between gap-2 text-xs text-slate-500">
          <span>
            Interactive Transformer Architecture Explainer · Educational visualization based on Cho et al. (CHI 2026)
          </span>
          <a
            href="https://dl.acm.org/doi/pdf/10.1145/3772318.3791725"
            target="_blank"
            rel="noopener noreferrer"
            className="hover:text-slate-900 underline underline-offset-2"
          >
            DOI: 10.1145/3772318.3791725
          </a>
        </div>
      </footer>
    </div>
  );
}
