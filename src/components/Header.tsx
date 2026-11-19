import React from 'react';
import { Sparkles } from 'lucide-react';
import { StageId } from '../engine/transformerEngine';

interface HeaderProps {
  onSelectStage: (stage: StageId) => void;
  onGenerateToken: () => void;
  canGenerate: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  onSelectStage,
  onGenerateToken,
  canGenerate,
}) => {
  const scrollToSection = (id: string, stage?: StageId) => {
    if (stage) {
      onSelectStage(stage);
    }
    const el = document.getElementById(id);
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  };

  return (
    <header className="sticky top-0 z-30 bg-white/95 backdrop-blur-sm border-b border-slate-200">
      <div className="max-w-[1400px] mx-auto px-4 sm:px-6 h-14 flex items-center justify-between gap-8">
        {/* Zone 1: Single text element wordmark */}
        <a
          href="#pipeline"
          onClick={(e) => {
            e.preventDefault();
            scrollToSection('pipeline');
          }}
          className="text-base sm:text-lg font-bold tracking-tight text-slate-900 whitespace-nowrap shrink-0 focus-visible:outline-2 focus-visible:outline-sky-600 focus-visible:outline-offset-2 rounded"
        >
          Transformer Explainer
        </a>

        {/* Zone 2: Clean single-line navigation links */}
        <nav
          aria-label="Primary"
          className="hidden md:flex items-center gap-6 text-sm font-medium text-slate-600"
        >
          <a
            href="#pipeline"
            onClick={(e) => {
              e.preventDefault();
              scrollToSection('pipeline');
            }}
            className="hover:text-slate-900 hover:underline underline-offset-4 transition-colors whitespace-nowrap shrink-0"
          >
            Pipeline
          </a>
          <a
            href="#inspector"
            onClick={(e) => {
              e.preventDefault();
              scrollToSection('inspector', 'embedding');
            }}
            className="hover:text-slate-900 hover:underline underline-offset-4 transition-colors whitespace-nowrap shrink-0"
          >
            Embeddings
          </a>
          <a
            href="#inspector"
            onClick={(e) => {
              e.preventDefault();
              scrollToSection('inspector', 'attention');
            }}
            className="hover:text-slate-900 hover:underline underline-offset-4 transition-colors whitespace-nowrap shrink-0"
          >
            Self-Attention
          </a>
          <a
            href="#inspector"
            onClick={(e) => {
              e.preventDefault();
              scrollToSection('inspector', 'mlp');
            }}
            className="hover:text-slate-900 hover:underline underline-offset-4 transition-colors whitespace-nowrap shrink-0"
          >
            MLP &amp; Residuals
          </a>
          <a
            href="#source"
            onClick={(e) => {
              e.preventDefault();
              scrollToSection('source');
            }}
            className="hover:text-slate-900 hover:underline underline-offset-4 transition-colors whitespace-nowrap shrink-0"
          >
            Source Paper
          </a>
        </nav>

        {/* Zone 3: Single primary action */}
        <div className="flex items-center shrink-0">
          <button
            type="button"
            onClick={onGenerateToken}
            disabled={!canGenerate}
            className="inline-flex items-center gap-2 px-4 py-2 text-xs sm:text-sm font-semibold text-white bg-slate-900 rounded-lg hover:bg-slate-800 active:scale-[0.98] disabled:opacity-45 disabled:cursor-not-allowed transition-all whitespace-nowrap shrink-0 focus-visible:outline-2 focus-visible:outline-sky-600 focus-visible:outline-offset-2"
          >
            <Sparkles className="w-3.5 h-3.5" aria-hidden="true" />
            <span>Generate Token</span>
          </button>
        </div>
      </div>
    </header>
  );
};
