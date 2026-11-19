# Transformer Architecture Explainer

An interactive, visual web application that demystifies how decoder-only Transformer large language models (modeled after GPT-2 Small: 12 layers, 12 attention heads, 768 embedding dimensions) process text and predict the next token.

Inspired by **Transformer Explainer** (*Cho et al., CHI 2026*), this tool bridges high-level architectural intuition with live low-level mathematical operations directly in the browser—no GPU, backend server, or coding experience required.

---

## Key Interactive Features

1. **End-to-End Architecture Pipeline**:
   - Follow any custom or preset text prompt from raw characters through **Tokenization & Embeddings**, **Multi-Head Masked Self-Attention**, **Residual Connections & MLP (GELU)**, and **Next-Token Probability Decoding**.
   - Use the **Guided Walkthrough** (`Play`, `Pause`, `Step Back`, `Step Forward`, `Reset`) to trace data flow one stage at a time.
2. **Tokenization & Positional Encodings**:
   - Inspect subword token splits, vocabulary IDs, token embedding vectors, sinusoidal positional encoding waves, and element-wise vector addition ($E_{\text{token}} + E_{\text{pos}}$).
3. **Multi-Head Masked Self-Attention ($Q, K, V$)**:
   - Explore all **12 Transformer layers** and **12 attention heads** with a live mini-head comparison grid.
   - Hover over any cell in the causal attention matrix or token-to-token arc diagram to inspect raw dot products ($Q \cdot K^T$), $\sqrt{d_k}$ scaling, upper-triangular **causal masking ($-\infty \to 0$)**, softmax weights, and weighted Value ($V$) aggregation.
4. **MLP Feed-Forward Network & Residual Pathways**:
   - Inspect how each token's representation expands ($768 \to 3072$), passes through the non-linear **GELU** activation curve with live plotted activations, projects back ($3072 \to 768$), and stabilizes via **Layer Normalization** and **Residual Skip Connections**.
5. **Interactive Decoding Hyperparameters & Autoregressive Generation**:
   - Adjust **Temperature ($T$)**, switch between **Top-$k$** and **Top-$p$ (Nucleus)** sampling, inspect live logit-to-probability transformations, run a **100-draw Monte Carlo sampling test**, and click **Generate Token** to append the sampled token and watch the Transformer recalculate autoregressively.

---

## Tech Stack

- **Frontend**: React 19, TypeScript, Vite
- **Styling**: Tailwind CSS v4
- **Icons**: Lucide React
- **Math & Visualization**: Custom deterministic in-browser Transformer forward-pass engine (`src/engine/transformerEngine.ts`) with interactive SVG diagrams, matrices, and activation curves.

---

## Prerequisites

- [Node.js](https://nodejs.org/) **v18.0.0 or higher** (v20+ recommended)
- **npm** v9+ (bundled with Node.js)

---

## Installation & Local Setup

1. **Clone the repository** (or download the project directory):
   ```bash
   git clone <repository-url>
   cd transformer-architecture-explainer
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Start the development server**:
   ```bash
   npm run dev
   ```
   By default, Vite serves the application locally on port `3000` (or the URL displayed in your terminal, e.g. `http://localhost:3000`).

4. **Build for production**:
   ```bash
   npm run build
   ```
   The compiled static bundle will be output to the `dist/` directory.

5. **Preview the production build locally**:
   ```bash
   npm run preview
   ```

6. **Type-check / Lint**:
   ```bash
   npm run lint
   ```

---

## Project Structure

```text
├── index.html                          # HTML entry point & typography imports
├── package.json                        # Scripts and dependencies
├── README.md                           # Documentation, setup guide, and source attribution
├── src/
│   ├── main.tsx                        # React DOM entry point
│   ├── index.css                       # Tailwind CSS v4 setup and custom slider/motion rules
│   ├── App.tsx                         # Main interactive workspace and state orchestration
│   ├── engine/
│   │   └── transformerEngine.ts        # In-browser tokenization, embeddings, QKV attention, MLP/GELU, and sampling math
│   └── components/
│       ├── Header.tsx                  # Top navigation bar
│       ├── PromptBar.tsx               # Prompt input, presets, token selector, and guided walkthrough controls
│       ├── PipelineCanvas.tsx          # Macro end-to-end interactive Transformer architecture diagram
│       ├── EmbeddingInspector.tsx      # Micro view: BPE tokens, embeddings, and positional encoding addition
│       ├── AttentionInspector.tsx      # Micro view: Q/K/V math, causal mask heatmap, arcs, and 12-head selector
│       ├── MlpInspector.tsx            # Micro view: LayerNorm, residual skip flow, and live GELU curve
│       ├── SamplingInspector.tsx       # Micro view: Logits, temperature scaling, Top-k/Top-p cutoff, and 100-draw simulator
│       └── CitationSection.tsx         # Formal academic source acknowledgment and BibTeX citation
```

---

## Source Acknowledgment

This interactive visualization is inspired by and based on the concepts, pedagogical framing, and multi-level visual explanation framework introduced in **Transformer Explainer**:

> **Transformer Explainer: Learning LLM Transformers with Interactive Visual Explanation and Experimentation**  
> Aeree Cho, Grace C. Kim, Alexander Karpekov, Seongmin Lee, Alec Helbling, Benjamin Hoover, Zijie J. Wang, Minsuk Kahng, and Duen Horng (Polo) Chau.  
> *Proceedings of the 2026 CHI Conference on Human Factors in Computing Systems (CHI '26)*, Association for Computing Machinery (ACM).  
> **ACM Digital Library**: [https://dl.acm.org/doi/10.1145/3772318.3791725](https://dl.acm.org/doi/10.1145/3772318.3791725)  
> **PDF**: [https://dl.acm.org/doi/pdf/10.1145/3772318.3791725](https://dl.acm.org/doi/pdf/10.1145/3772318.3791725)  
> **arXiv Preprint**: [https://arxiv.org/abs/2408.04619](https://arxiv.org/abs/2408.04619)

### BibTeX Citation

```bibtex
@inproceedings{cho2026transformerexplainer,
  title     = {Transformer Explainer: Learning LLM Transformers with Interactive Visual Explanation and Experimentation},
  author    = {Cho, Aeree and Kim, Grace C. and Karpekov, Alexander and Lee, Seongmin and Helbling, Alec and Hoover, Benjamin and Wang, Zijie J. and Kahng, Minsuk and Chau, Duen Horng},
  booktitle = {Proceedings of the 2026 CHI Conference on Human Factors in Computing Systems (CHI '26)},
  year      = {2026},
  publisher = {Association for Computing Machinery},
  doi       = {10.1145/3772318.3791725},
  url       = {https://dl.acm.org/doi/10.1145/3772318.3791725}
}
```

### Additional Architectural Foundation

- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). *Attention Is All You Need*. Advances in Neural Information Processing Systems (NeurIPS 2017).
- Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). *Language Models are Unsupervised Multitask Learners* (GPT-2).

---

## License

This project is provided for educational and research visualization purposes.
