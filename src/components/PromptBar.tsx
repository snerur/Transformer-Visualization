import React from 'react';
import {
  Play,
  Pause,
  SkipBack,
  SkipForward,
  RotateCcw,
} from 'lucide-react';
import {
  PRESET_PROMPTS,
  StageId,
  TokenData,
} from '../engine/transformerEngine';

interface PromptBarProps {
  prompt: string;
  onPromptChange: (val: string) => void;
  onSelectPreset: (text: string) => void;
  onResetPrompt: () => void;
  tokens: TokenData[];
  selectedTokenIdx: number;
  onSelectToken: (idx: number) => void;
  activeStage: StageId;
  isPlaying: boolean;
  onTogglePlay: () => void;
  onStepBack: () => void;
  onStepForward: () => void;
  lastGeneratedToken: string | null;
}

const STAGE_META: Record<StageId, { step: number; label: string }> = {
  embedding: { step: 1, label: 'Tokenization & Embeddings' },
  attention: { step: 2, label: 'Multi-Head Masked Self-Attention' },
  mlp: { step: 3, label: 'Residual Stream & MLP (GELU)' },
  sampling: { step: 4, label: 'Logits, Softmax & Decoding' },
};

export const PromptBar: React.FC<PromptBarProps> = ({
  prompt,
  onPromptChange,
  onSelectPreset,
  onResetPrompt,
  tokens,
  selectedTokenIdx,
  onSelectToken,
  activeStage,
  isPlaying,
  onTogglePlay,
  onStepBack,
  onStepForward,
  lastGeneratedToken,
}) => {
  const stageInfo = STAGE_META[activeStage];

  return (
    <section
      aria-label="Prompt and walkthrough controls"
      className="bg-white border border-slate-200 rounded-xl p-4 sm:p-5 space-y-4"
    >
      {/* Row 1: Prompt input + Example presets */}
      <div className="flex flex-col lg:flex-row lg:items-center gap-3">
        <div className="flex-1 flex items-center gap-2">
          <label htmlFor="prompt-input" className="text-xs font-semibold text-slate-500 whitespace-nowrap shrink-0">
            Input Text
          </label>
          <div className="relative flex-1">
            <input
              id="prompt-input"
              type="text"
              value={prompt}
              onChange={(e) => onPromptChange(e.target.value)}
              maxLength={110}
              placeholder="Type any sentence to trace through the Transformer..."
              className="w-full h-10 px-3.5 pr-10 text-sm sm:text-base font-medium text-slate-900 bg-slate-50 border border-slate-200 rounded-lg focus:bg-white focus:border-sky-600 focus:outline-none focus:ring-2 focus:ring-sky-600/15 transition-colors"
            />
            <button
              type="button"
              onClick={onResetPrompt}
              title="Reset to default prompt"
              aria-label="Reset prompt"
              className="absolute right-2 top-1/2 -translate-y-1/2 p-1.5 text-slate-400 hover:text-slate-700 rounded-md transition-colors focus-visible:outline-2 focus-visible:outline-sky-600"
            >
              <RotateCcw className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Example selector segmented buttons */}
        <div className="flex items-center gap-1 p-1 bg-slate-100 rounded-lg overflow-x-auto">
          {PRESET_PROMPTS.map((preset) => {
            const isActive = prompt.trim().toLowerCase() === preset.text.toLowerCase();
            return (
              <button
                key={preset.id}
                type="button"
                onClick={() => onSelectPreset(preset.text)}
                className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 focus-visible:outline-2 focus-visible:outline-sky-600 ${
                  isActive
                    ? 'bg-white text-slate-900 shadow-xs'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                {preset.label}
              </button>
            );
          })}
        </div>
      </div>

      {/* Hairline divider */}
      <div className="border-t border-slate-100" />

      {/* Row 2: Clickable Token Selector + Guided Walkthrough Playback */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
        <div className="flex items-center gap-2.5 flex-wrap">
          <span className="text-xs font-semibold text-slate-500 whitespace-nowrap shrink-0">
            Focus Token
          </span>
          <div className="flex items-center gap-1.5 flex-wrap" role="tablist" aria-label="Select a token to inspect">
            {tokens.map((tok) => {
              const isSelected = tok.index === selectedTokenIdx;
              const isJustGenerated =
                lastGeneratedToken !== null &&
                tok.index === tokens.length - 1 &&
                tok.raw === lastGeneratedToken;
              return (
                <button
                  key={`${tok.index}-${tok.tokenId}`}
                  type="button"
                  role="tab"
                  aria-selected={isSelected}
                  onClick={() => onSelectToken(tok.index)}
                  className={`px-2.5 py-1.5 text-xs font-mono rounded-md border transition-all whitespace-nowrap shrink-0 focus-visible:outline-2 focus-visible:outline-sky-600 ${
                    isSelected
                      ? 'bg-sky-600 text-white border-sky-600 font-semibold'
                      : isJustGenerated
                      ? 'bg-emerald-50 text-emerald-900 border-emerald-300 hover:border-emerald-400'
                      : 'bg-white text-slate-800 border-slate-200 hover:border-slate-400'
                  }`}
                >
                  <span className={`mr-1.5 text-[10px] tabular-nums ${isSelected ? 'text-sky-100' : 'text-slate-400'}`}>
                    {tok.index}
                  </span>
                  <span>{tok.text}</span>
                </button>
              );
            })}
          </div>
          {tokens.length >= 12 && (
            <span className="text-xs text-amber-700 font-medium">
              Max 12 tokens displayed
            </span>
          )}
        </div>

        {/* Guided step-by-step walkthrough controls */}
        <div className="flex items-center gap-3 shrink-0">
          <div className="text-xs text-slate-600 whitespace-nowrap">
            <span className="font-semibold text-slate-900 tabular-nums">
              Stage {stageInfo.step}/4
            </span>
            <span className="mx-1.5 text-slate-300" aria-hidden="true">·</span>
            <span>{stageInfo.label}</span>
          </div>

          <div className="flex items-center gap-1 p-1 bg-slate-100 rounded-lg">
            <button
              type="button"
              onClick={onStepBack}
              aria-label="Previous stage"
              title="Previous stage"
              className="p-1.5 text-slate-700 hover:text-slate-900 hover:bg-white rounded-md transition-colors focus-visible:outline-2 focus-visible:outline-sky-600"
            >
              <SkipBack className="w-3.5 h-3.5" />
            </button>
            <button
              type="button"
              onClick={onTogglePlay}
              className="inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-semibold text-slate-900 bg-white rounded-md shadow-xs hover:bg-slate-50 transition-colors whitespace-nowrap shrink-0 focus-visible:outline-2 focus-visible:outline-sky-600"
            >
              {isPlaying ? (
                <>
                  <Pause className="w-3.5 h-3.5" aria-hidden="true" />
                  <span>Pause</span>
                </>
              ) : (
                <>
                  <Play className="w-3.5 h-3.5" aria-hidden="true" />
                  <span>Play Flow</span>
                </>
              )}
            </button>
            <button
              type="button"
              onClick={onStepForward}
              aria-label="Next stage"
              title="Next stage"
              className="p-1.5 text-slate-700 hover:text-slate-900 hover:bg-white rounded-md transition-colors focus-visible:outline-2 focus-visible:outline-sky-600"
            >
              <SkipForward className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </div>
    </section>
  );
};
