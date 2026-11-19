import React, { useState } from 'react';
import { Check, Copy, ExternalLink } from 'lucide-react';

const BIBTEX = `@inproceedings{cho2026transformerexplainer,
  title     = {Transformer Explainer: Learning LLM Transformers with Interactive Visual Explanation and Experimentation},
  author    = {Cho, Aeree and Kim, Grace C. and Karpekov, Alexander and Lee, Seongmin and Helbling, Alec and Hoover, Benjamin and Wang, Zijie J. and Kahng, Minsuk and Chau, Duen Horng},
  booktitle = {Proceedings of the 2026 CHI Conference on Human Factors in Computing Systems (CHI '26)},
  year      = {2026},
  publisher = {Association for Computing Machinery},
  doi       = {10.1145/3772318.3791725},
  url       = {https://dl.acm.org/doi/10.1145/3772318.3791725}
}`;

export const CitationSection: React.FC = () => {
  const [copied, setCopied] = useState(false);

  const handleCopyBibtex = async () => {
    try {
      await navigator.clipboard.writeText(BIBTEX);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Fallback copy if clipboard API is restricted inside iframe
      const textarea = document.createElement('textarea');
      textarea.value = BIBTEX;
      textarea.style.position = 'fixed';
      textarea.style.opacity = '0';
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand('copy');
      document.body.removeChild(textarea);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <section
      id="source"
      aria-label="Source acknowledgment and academic citation"
      className="bg-white border border-slate-200 rounded-xl p-6 sm:p-8 space-y-5"
    >
      <div className="flex flex-col md:flex-row md:items-start justify-between gap-4">
        <div className="space-y-2 max-w-3xl">
          <div className="text-xs font-semibold text-slate-500">
            Source Acknowledgment · ACM CHI 2026
          </div>
          <h2 className="text-xl font-bold text-slate-900 tracking-tight">
            Based on Transformer Explainer
          </h2>
          <p className="text-sm text-slate-600 leading-relaxed">
            This interactive visualization is inspired by the pedagogical architecture and multi-level explanation design presented in{' '}
            <em className="font-serif text-slate-900">
              Transformer Explainer: Learning LLM Transformers with Interactive Visual Explanation and Experimentation
            </em>{' '}
            by Aeree Cho, Grace C. Kim, Alexander Karpekov, Seongmin Lee, Alec Helbling, Benjamin Hoover, Zijie J. Wang, Minsuk Kahng, and Duen Horng (Polo) Chau (Georgia Institute of Technology, IBM Research, and Oregon State University).
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2.5 shrink-0">
          <a
            href="https://dl.acm.org/doi/pdf/10.1145/3772318.3791725"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-semibold text-slate-900 bg-slate-100 hover:bg-slate-200 rounded-lg transition-colors whitespace-nowrap"
          >
            <span>ACM Paper PDF</span>
            <ExternalLink className="w-3.5 h-3.5" aria-hidden="true" />
          </a>
          <a
            href="https://arxiv.org/abs/2408.04619"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-semibold text-slate-700 bg-white border border-slate-200 hover:border-slate-300 rounded-lg transition-colors whitespace-nowrap"
          >
            <span>arXiv:2408.04619</span>
            <ExternalLink className="w-3.5 h-3.5" aria-hidden="true" />
          </a>
        </div>
      </div>

      {/* BibTeX block */}
      <div className="bg-slate-50 border border-slate-200 rounded-lg p-4 space-y-2">
        <div className="flex items-center justify-between gap-2">
          <span className="text-xs font-semibold text-slate-700">
            BibTeX Citation (DOI: 10.1145/3772318.3791725)
          </span>
          <button
            type="button"
            onClick={handleCopyBibtex}
            className="inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium text-slate-700 bg-white border border-slate-200 hover:border-slate-300 rounded-md transition-colors whitespace-nowrap focus-visible:outline-2 focus-visible:outline-sky-600"
          >
            {copied ? (
              <>
                <Check className="w-3.5 h-3.5 text-emerald-600" aria-hidden="true" />
                <span className="text-emerald-700">Copied</span>
              </>
            ) : (
              <>
                <Copy className="w-3.5 h-3.5" aria-hidden="true" />
                <span>Copy BibTeX</span>
              </>
            )}
          </button>
        </div>
        <pre className="text-xs font-mono text-slate-700 overflow-x-auto leading-relaxed">
          {BIBTEX}
        </pre>
      </div>
    </section>
  );
};
