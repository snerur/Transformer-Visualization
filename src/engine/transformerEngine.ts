export type StageId = 'embedding' | 'attention' | 'mlp' | 'sampling';
export type SamplingMode = 'top-k' | 'top-p';

export interface PresetPrompt {
  id: string;
  label: string;
  text: string;
  topic: string;
}

export const PRESET_PROMPTS: PresetPrompt[] = [
  {
    id: 'dataviz',
    label: 'Data visualization',
    text: 'Data visualization empowers users to',
    topic: 'Paper benchmark example',
  },
  {
    id: 'science',
    label: 'Scientific discovery',
    text: 'Modern telescopes allow astronomers to',
    topic: 'Science & astronomy',
  },
  {
    id: 'fox',
    label: 'Classic pangram',
    text: 'The quick brown fox jumps over',
    topic: 'Syntactic structure',
  },
  {
    id: 'coffee',
    label: 'Everyday sentence',
    text: 'Warm coffee in the morning helps',
    topic: 'Natural continuation',
  },
];

export const HEAD_ROLES: { index: number; name: string; summary: string }[] = [
  { index: 0, name: 'Previous Token', summary: 'Focuses strongly on the immediately preceding token to track local bigram syntax.' },
  { index: 1, name: 'Semantic Core', summary: 'Connects the current token to high-meaning nouns and verbs earlier in the sentence.' },
  { index: 2, name: 'Sentence Anchor', summary: 'Attends heavily to the opening token as a stable reference point (attention sink).' },
  { index: 3, name: 'Verb–Object Link', summary: 'Links action verbs and prepositions to their grammatical subjects and targets.' },
  { index: 4, name: 'Local Window', summary: 'Aggregates context evenly across the last 2–3 neighboring tokens.' },
  { index: 5, name: 'Long-Range Context', summary: 'Bridges distant tokens across the sequence to preserve topic coherence.' },
  { index: 6, name: 'Self Focus', summary: 'Reinforces the token’s own identity while lightly blending adjacent cues.' },
  { index: 7, name: 'Modifier Binding', summary: 'Associates adjectives and qualifiers with the nouns they describe.' },
  { index: 8, name: 'Phrase Boundary', summary: 'Tracks clause transitions, articles, and phrase-leading function words.' },
  { index: 9, name: 'Induction Pattern', summary: 'Searches earlier context for structural cues that predict the upcoming continuation.' },
  { index: 10, name: 'Broad Context', summary: 'Distributes attention smoothly across all visible prior tokens.' },
  { index: 11, name: 'Next-Token Prep', summary: 'Concentrates salient sentence context onto the final token for decoding.' },
];

export interface TokenData {
  index: number;
  text: string;
  raw: string;
  tokenId: number;
  tokenEmbedding: number[];
  positionalEncoding: number[];
  combinedEmbedding: number[];
}

export interface HeadComputation {
  headIndex: number;
  roleName: string;
  roleSummary: string;
  queries: number[][];
  keys: number[][];
  values: number[][];
  dotScores: number[][];
  scaledScores: number[][];
  attentionWeights: number[][];
  weightedValues: number[][];
}

export interface MlpTokenState {
  tokenIndex: number;
  inputAfterAttention: number[];
  residualPreMlp: number[];
  layerNormMean: number;
  layerNormStd: number;
  layerNormOut: number[];
  expandedPreGelu: number[];
  expandedPostGelu: number[];
  projectedOut: number[];
  finalBlockOutput: number[];
}

export interface LayerComputation {
  layerIndex: number;
  heads: HeadComputation[];
  mlpStates: MlpTokenState[];
}

export interface CandidateToken {
  token: string;
  tokenId: number;
  rawLogit: number;
  scaledLogit: number;
  baseProbability: number;
  cumulativeProb: number;
  included: boolean;
  filteredProbability: number;
}

export interface TransformerRun {
  prompt: string;
  tokens: TokenData[];
  layers: LayerComputation[];
  candidates: CandidateToken[];
  config: {
    nLayers: number;
    nHeads: number;
    dModel: number;
    dHead: number;
    dMlp: number;
    vocabSize: number;
  };
}

// Deterministic hash for repeatable vector generation
function hashString(str: string): number {
  let h = 2166136261;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function seededRandom(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function round3(n: number): number {
  return Math.round(n * 1000) / 1000;
}

export function gelu(x: number): number {
  const c = Math.sqrt(2 / Math.PI);
  return 0.5 * x * (1 + Math.tanh(c * (x + 0.044715 * x * x * x)));
}

// Realistic GPT-2 style known token IDs for common words
const KNOWN_TOKEN_IDS: Record<string, number> = {
  Data: 6601,
  ' visualization': 32704,
  ' empowers': 44960,
  ' users': 2985,
  ' to': 284,
  The: 464,
  ' quick': 2068,
  ' brown': 7586,
  ' fox': 21831,
  ' jumps': 18045,
  ' over': 625,
  ' the': 262,
  Modern: 17150,
  ' telescopes': 37742,
  ' allow': 1249,
  ' astronomers': 28512,
  Warm: 32944,
  ' coffee': 6891,
  ' in': 287,
  ' morning': 3329,
  ' helps': 5419,
  ' explore': 7301,
  ' understand': 1833,
  ' discover': 7073,
  ' analyze': 16620,
  ' interpret': 10205,
  ' see': 766,
  ' identify': 5911,
  ' communicate': 10996,
  ' make': 787,
  ' visualize': 26425,
  ' observe': 7012,
  ' study': 2050,
  ' detect': 4886,
  ' measure': 3868,
  ' map': 3975,
  ' capture': 5152,
  ' peer': 14251,
  ' track': 2610,
  ' lazy': 16931,
  ' tall': 7331,
  ' sleeping': 10790,
  ' mossy': 41219,
  ' wooden': 10681,
  ' fence': 13990,
  ' me': 502,
  ' people': 661,
  ' start': 923,
  ' focus': 2962,
  ' sharpen': 29412,
  ' wake': 7765,
  ' calm': 8542,
};

const FUNCTION_WORDS = new Set([
  'the', 'a', 'an', 'to', 'in', 'on', 'of', 'for', 'with', 'over', 'and', 'or', 'is', 'are', 'was', 'be', 'by', 'at', 'from', 'that', 'this',
]);

export function tokenizePrompt(input: string): { raw: string; text: string; tokenId: number }[] {
  const cleaned = input.trim().slice(0, 120);
  if (!cleaned) {
    return [{ raw: 'Data', text: 'Data', tokenId: 6601 }];
  }

  // Split into words and punctuation while preserving leading spaces (GPT-2 BPE style)
  const matches = cleaned.match(/\s*[^\s.,!?;:]+|[.,!?;:]/g) || [];
  const pieces: { raw: string; text: string; tokenId: number }[] = [];

  for (let i = 0; i < matches.length && pieces.length < 12; i++) {
    const raw = matches[i];
    const trimmed = raw.trim();
    // Simulate BPE subword split for long technical words (> 12 chars) if not in common list
    if (trimmed.length > 13 && !KNOWN_TOKEN_IDS[raw] && /^[A-Za-z]+$/.test(trimmed)) {
      const splitAt = Math.ceil(trimmed.length * 0.6);
      const partA = raw.slice(0, raw.length - trimmed.length + splitAt);
      const partB = trimmed.slice(splitAt);
      pieces.push({
        raw: partA,
        text: partA.trim(),
        tokenId: KNOWN_TOKEN_IDS[partA] ?? (1000 + (hashString(partA) % 48000)),
      });
      if (pieces.length < 12) {
        pieces.push({
          raw: partB,
          text: partB,
          tokenId: KNOWN_TOKEN_IDS[partB] ?? (1000 + (hashString(partB) % 48000)),
        });
      }
    } else {
      pieces.push({
        raw,
        text: trimmed,
        tokenId: KNOWN_TOKEN_IDS[raw] ?? (1000 + (hashString(raw) % 48000)),
      });
    }
  }

  return pieces;
}

const SLICE_DIM = 8; // Inspectable 8-element slice representing d_model=768 / d_head=64
const MLP_SLICE = 12; // Inspectable 12-neuron slice representing d_mlp=3072

function generateTokenEmbedding(tokenId: number, text: string): number[] {
  const rand = seededRandom(tokenId * 31 + hashString(text.toLowerCase()));
  const vec: number[] = [];
  for (let d = 0; d < SLICE_DIM; d++) {
    vec.push(round3((rand() * 2 - 1) * 0.85));
  }
  return vec;
}

function generatePositionalEncoding(pos: number): number[] {
  const vec: number[] = [];
  for (let d = 0; d < SLICE_DIM; d++) {
    const freq = Math.pow(10000, (2 * Math.floor(d / 2)) / 64);
    const val = d % 2 === 0 ? Math.sin(pos / freq) : Math.cos(pos / freq);
    vec.push(round3(val * 0.45));
  }
  return vec;
}

function projectVector(vec: number[], seed: number, scale = 0.7): number[] {
  const rand = seededRandom(seed);
  const out: number[] = [];
  for (let i = 0; i < vec.length; i++) {
    let sum = 0;
    for (let j = 0; j < vec.length; j++) {
      sum += vec[j] * ((rand() * 2 - 1) * scale);
    }
    out.push(round3(Math.tanh(sum) * 1.35));
  }
  return out;
}

function dot(a: number[], b: number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function layerNorm(vec: number[]): { mean: number; std: number; out: number[] } {
  const mean = vec.reduce((a, b) => a + b, 0) / vec.length;
  const variance = vec.reduce((a, b) => a + (b - mean) * (b - mean), 0) / vec.length;
  const std = Math.sqrt(variance + 1e-5);
  return {
    mean: round3(mean),
    std: round3(std),
    out: vec.map((v) => round3((v - mean) / std)),
  };
}

// Contextual next-token candidate dictionary for realistic interactive continuations
interface CandidateSpec {
  token: string;
  logit: number;
}

const CONTEXT_CANDIDATES: { match: RegExp; candidates: CandidateSpec[] }[] = [
  {
    match: /data visualization empowers users to$/i,
    candidates: [
      { token: ' explore', logit: 8.92 },
      { token: ' understand', logit: 8.54 },
      { token: ' discover', logit: 7.98 },
      { token: ' analyze', logit: 7.45 },
      { token: ' interpret', logit: 7.02 },
      { token: ' identify', logit: 6.74 },
      { token: ' see', logit: 6.42 },
      { token: ' communicate', logit: 6.08 },
      { token: ' make', logit: 5.71 },
      { token: ' gain', logit: 5.35 },
    ],
  },
  {
    match: /empowers users to (explore|understand|discover|analyze|interpret|see)$/i,
    candidates: [
      { token: ' complex', logit: 8.75 },
      { token: ' patterns', logit: 8.42 },
      { token: ' hidden', logit: 7.86 },
      { token: ' large', logit: 7.41 },
      { token: ' relationships', logit: 7.05 },
      { token: ' trends', logit: 6.82 },
      { token: ' datasets', logit: 6.44 },
      { token: ' insights', logit: 6.09 },
      { token: ' information', logit: 5.78 },
      { token: ' models', logit: 5.42 },
    ],
  },
  {
    match: /(complex|hidden|large)$/i,
    candidates: [
      { token: ' patterns', logit: 8.81 },
      { token: ' datasets', logit: 8.47 },
      { token: ' relationships', logit: 8.02 },
      { token: ' structures', logit: 7.54 },
      { token: ' systems', logit: 7.18 },
      { token: ' data', logit: 6.84 },
      { token: ' ideas', logit: 6.32 },
      { token: ' phenomena', logit: 5.95 },
      { token: ' networks', logit: 5.58 },
      { token: ' trends', logit: 5.24 },
    ],
  },
  {
    match: /modern telescopes allow astronomers to$/i,
    candidates: [
      { token: ' observe', logit: 9.15 },
      { token: ' study', logit: 8.64 },
      { token: ' detect', logit: 8.12 },
      { token: ' peer', logit: 7.48 },
      { token: ' measure', logit: 7.15 },
      { token: ' map', logit: 6.82 },
      { token: ' capture', logit: 6.45 },
      { token: ' track', logit: 6.04 },
      { token: ' examine', logit: 5.76 },
      { token: ' explore', logit: 5.49 },
    ],
  },
  {
    match: /allow astronomers to (observe|study|detect|measure|map|capture)$/i,
    candidates: [
      { token: ' distant', logit: 9.04 },
      { token: ' faint', logit: 8.31 },
      { token: ' galaxies', logit: 7.96 },
      { token: ' exoplanets', logit: 7.72 },
      { token: ' ancient', logit: 7.14 },
      { token: ' cosmic', logit: 6.88 },
      { token: ' stars', logit: 6.52 },
      { token: ' nebulae', logit: 6.18 },
      { token: ' light', logit: 5.83 },
      { token: ' planets', logit: 5.46 },
    ],
  },
  {
    match: /quick brown fox jumps over$/i,
    candidates: [
      { token: ' the', logit: 10.12 },
      { token: ' a', logit: 7.85 },
      { token: ' every', logit: 6.32 },
      { token: ' tall', logit: 5.94 },
      { token: ' fallen', logit: 5.61 },
      { token: ' two', logit: 5.38 },
      { token: ' mossy', logit: 5.12 },
      { token: ' wooden', logit: 4.89 },
      { token: ' quiet', logit: 4.64 },
      { token: ' low', logit: 4.41 },
    ],
  },
  {
    match: /jumps over the$/i,
    candidates: [
      { token: ' lazy', logit: 10.45 },
      { token: ' sleeping', logit: 7.92 },
      { token: ' wooden', logit: 7.15 },
      { token: ' old', logit: 6.84 },
      { token: ' garden', logit: 6.42 },
      { token: ' mossy', logit: 6.08 },
      { token: ' fallen', logit: 5.79 },
      { token: ' quiet', logit: 5.41 },
      { token: ' tall', logit: 5.12 },
      { token: ' stone', logit: 4.88 },
    ],
  },
  {
    match: /over the lazy$/i,
    candidates: [
      { token: ' dog', logit: 10.82 },
      { token: ' hound', logit: 7.42 },
      { token: ' cat', logit: 6.94 },
      { token: ' bear', logit: 6.15 },
      { token: ' badger', logit: 5.78 },
      { token: ' wolf', logit: 5.44 },
      { token: ' deer', logit: 5.12 },
      { token: ' sheep', logit: 4.88 },
      { token: ' log', logit: 4.55 },
      { token: ' turtle', logit: 4.21 },
    ],
  },
  {
    match: /warm coffee in the morning helps$/i,
    candidates: [
      { token: ' me', logit: 8.85 },
      { token: ' people', logit: 8.22 },
      { token: ' sharpen', logit: 7.76 },
      { token: ' kickstart', logit: 7.48 },
      { token: ' wake', logit: 7.12 },
      { token: ' improve', logit: 6.78 },
      { token: ' calm', logit: 6.39 },
      { token: ' boost', logit: 6.14 },
      { token: ' bring', logit: 5.72 },
      { token: ' set', logit: 5.38 },
    ],
  },
  {
    match: /helps me$/i,
    candidates: [
      { token: ' focus', logit: 9.08 },
      { token: ' wake', logit: 8.56 },
      { token: ' start', logit: 8.14 },
      { token: ' think', logit: 7.62 },
      { token: ' stay', logit: 7.25 },
      { token: ' relax', logit: 6.88 },
      { token: ' feel', logit: 6.51 },
      { token: ' plan', logit: 6.12 },
      { token: ' write', logit: 5.84 },
      { token: ' breathe', logit: 5.41 },
    ],
  },
];

function getContextualCandidates(prompt: string): CandidateSpec[] {
  const normalized = prompt.trim();
  for (const rule of CONTEXT_CANDIDATES) {
    if (rule.match.test(normalized)) {
      return rule.candidates;
    }
  }

  // Grammar-aware fallback for arbitrary user prompts
  const lastWord = normalized.split(/\s+/).pop()?.toLowerCase().replace(/[^a-z]/g, '') || '';
  if (lastWord === 'to') {
    return [
      { token: ' learn', logit: 8.42 },
      { token: ' build', logit: 8.15 },
      { token: ' understand', logit: 7.88 },
      { token: ' create', logit: 7.51 },
      { token: ' explore', logit: 7.12 },
      { token: ' find', logit: 6.84 },
      { token: ' make', logit: 6.56 },
      { token: ' see', logit: 6.21 },
      { token: ' share', logit: 5.89 },
      { token: ' improve', logit: 5.54 },
    ];
  }
  if (['the', 'a', 'an', 'this', 'that', 'my', 'our'].includes(lastWord)) {
    return [
      { token: ' model', logit: 8.38 },
      { token: ' system', logit: 8.04 },
      { token: ' world', logit: 7.76 },
      { token: ' next', logit: 7.42 },
      { token: ' idea', logit: 7.08 },
      { token: ' process', logit: 6.75 },
      { token: ' pattern', logit: 6.44 },
      { token: ' future', logit: 6.12 },
      { token: ' answer', logit: 5.81 },
      { token: ' data', logit: 5.48 },
    ];
  }
  if (['is', 'was', 'are', 'were', 'be'].includes(lastWord)) {
    return [
      { token: ' designed', logit: 8.32 },
      { token: ' built', logit: 8.01 },
      { token: ' able', logit: 7.72 },
      { token: ' clear', logit: 7.38 },
      { token: ' powered', logit: 7.04 },
      { token: ' simple', logit: 6.71 },
      { token: ' shaped', logit: 6.39 },
      { token: ' changing', logit: 6.05 },
      { token: ' ready', logit: 5.74 },
      { token: ' remarkably', logit: 5.38 },
    ];
  }

  // Deterministic general continuation tailored by prompt hash
  const pool = [
    ' clearly', ' effectively', ' quickly', ' seamlessly', ' naturally',
    ' together', ' today', ' across', ' through', ' within',
    ' patterns', ' insights', ' context', ' meaning', ' systems',
  ];
  const rand = seededRandom(hashString(normalized));
  const picked = [...pool]
    .sort(() => rand() - 0.5)
    .slice(0, 10)
    .map((token, idx) => ({
      token,
      logit: round3(8.6 - idx * 0.38 + (rand() - 0.5) * 0.15),
    }))
    .sort((a, b) => b.logit - a.logit);

  return picked;
}

export function applyDecodingStrategy(
  rawCandidates: CandidateSpec[],
  temperature: number,
  samplingMode: SamplingMode,
  topK: number,
  topP: number
): CandidateToken[] {
  const safeTemp = Math.max(0.05, Math.min(2.0, temperature));
  const scaled = rawCandidates.map((c) => c.logit / safeTemp);
  const maxScaled = Math.max(...scaled);
  const exps = scaled.map((s) => Math.exp(s - maxScaled));
  const sumExp = exps.reduce((a, b) => a + b, 0);

  let cumulative = 0;
  const items: CandidateToken[] = rawCandidates.map((c, idx) => {
    const baseProbability = exps[idx] / sumExp;
    const prevCumulative = cumulative;
    cumulative += baseProbability;

    let included = false;
    if (samplingMode === 'top-k') {
      included = idx < topK;
    } else {
      // Top-p (Nucleus): include tokens until cumulative probability exceeds topP (always keep at least 1)
      included = idx === 0 || prevCumulative < topP;
    }

    return {
      token: c.token,
      tokenId: KNOWN_TOKEN_IDS[c.token] ?? (2000 + (hashString(c.token) % 45000)),
      rawLogit: round3(c.logit),
      scaledLogit: round3(scaled[idx]),
      baseProbability,
      cumulativeProb: Math.min(1, cumulative),
      included,
      filteredProbability: 0,
    };
  });

  const includedSum = items
    .filter((i) => i.included)
    .reduce((acc, i) => acc + i.baseProbability, 0);

  return items.map((item) => ({
    ...item,
    filteredProbability: item.included && includedSum > 0 ? item.baseProbability / includedSum : 0,
  }));
}

export function sampleNextToken(candidates: CandidateToken[]): CandidateToken {
  const eligible = candidates.filter((c) => c.included && c.filteredProbability > 0);
  if (eligible.length === 0) return candidates[0];
  const r = Math.random();
  let acc = 0;
  for (const c of eligible) {
    acc += c.filteredProbability;
    if (r <= acc) return c;
  }
  return eligible[eligible.length - 1];
}

export function runSamplingSimulation(
  candidates: CandidateToken[],
  draws = 100
): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const c of candidates) {
    counts[c.token] = 0;
  }
  for (let i = 0; i < draws; i++) {
    const picked = sampleNextToken(candidates);
    counts[picked.token] = (counts[picked.token] || 0) + 1;
  }
  return counts;
}

export function runTransformerForwardPass(
  prompt: string,
  temperature: number,
  samplingMode: SamplingMode,
  topK: number,
  topP: number
): TransformerRun {
  const rawTokens = tokenizePrompt(prompt);
  const tokens: TokenData[] = rawTokens.map((t, index) => {
    const tokenEmbedding = generateTokenEmbedding(t.tokenId, t.text);
    const positionalEncoding = generatePositionalEncoding(index);
    const combinedEmbedding = tokenEmbedding.map((v, d) => round3(v + positionalEncoding[d]));
    return {
      index,
      text: t.text,
      raw: t.raw,
      tokenId: t.tokenId,
      tokenEmbedding,
      positionalEncoding,
      combinedEmbedding,
    };
  });

  const nTokens = tokens.length;
  const nLayers = 12;
  const nHeads = 12;
  const layers: LayerComputation[] = [];

  let currentVectors = tokens.map((t) => [...t.combinedEmbedding]);

  for (let layerIdx = 0; layerIdx < nLayers; layerIdx++) {
    const heads: HeadComputation[] = [];

    for (let headIdx = 0; headIdx < nHeads; headIdx++) {
      const baseSeed = (layerIdx + 1) * 997 + (headIdx + 1) * 131;
      const queries = currentVectors.map((v) => projectVector(v, baseSeed + 11));
      const keys = currentVectors.map((v) => projectVector(v, baseSeed + 23));
      const values = currentVectors.map((v) => projectVector(v, baseSeed + 37));

      const dotScores: number[][] = [];
      const scaledScores: number[][] = [];
      const attentionWeights: number[][] = [];

      const scaleFactor = Math.sqrt(64); // Authentic GPT-2 sqrt(d_k) = sqrt(64) = 8

      for (let i = 0; i < nTokens; i++) {
        const dotRow: number[] = [];
        const scaledRow: number[] = [];
        const logitsForRow: number[] = [];

        for (let j = 0; j < nTokens; j++) {
          // Real Q·K dot product amplified slightly to reflect 64-dim energy
          const rawDot = dot(queries[i], keys[j]) * 2.4;

          // Head-specific interpretable structural pattern (reflecting real GPT-2 head specializations)
          let structuralBias = 0;
          const isContentWordJ = !FUNCTION_WORDS.has(tokens[j].text.toLowerCase());
          const distance = i - j;

          switch (headIdx % 12) {
            case 0: // Previous token
              structuralBias = distance === 1 ? 2.8 : distance === 0 ? 0.5 : -0.6;
              break;
            case 1: // Semantic core
              structuralBias = isContentWordJ ? 2.1 : -0.5;
              break;
            case 2: // Sentence anchor
              structuralBias = j === 0 ? 2.9 : -0.3;
              break;
            case 3: // Verb-object / early subject
              structuralBias = j === Math.max(0, i - 2) || isContentWordJ ? 1.8 : -0.4;
              break;
            case 4: // Local window
              structuralBias = distance <= 2 ? 1.9 - distance * 0.4 : -0.9;
              break;
            case 5: // Long-range context
              structuralBias = distance >= 3 ? 2.2 : 0.2;
              break;
            case 6: // Self focus
              structuralBias = i === j ? 2.7 : -0.4;
              break;
            case 7: // Modifier binding
              structuralBias = distance === 1 && isContentWordJ ? 2.4 : isContentWordJ ? 0.9 : -0.3;
              break;
            case 8: // Phrase boundary
              structuralBias = !isContentWordJ ? 1.7 : 0.3;
              break;
            case 9: // Induction / alternating
              structuralBias = (i + j + layerIdx) % 2 === 0 ? 1.4 : -0.2;
              break;
            case 10: // Broad context
              structuralBias = 0.2;
              break;
            case 11: // Next-token prep (sharper in deeper layers)
              structuralBias = isContentWordJ ? 1.4 + layerIdx * 0.08 : 0.1;
              break;
          }

          const combinedDot = round3(rawDot + structuralBias * 2.1);
          const scaled = round3(combinedDot / (scaleFactor / 3.2));
          dotRow.push(combinedDot);
          scaledRow.push(scaled);
          logitsForRow.push(j <= i ? scaled : -Infinity);
        }

        // Softmax over causal positions j <= i
        const validLogits = logitsForRow.filter((v) => v !== -Infinity);
        const maxL = Math.max(...validLogits);
        const expRow = logitsForRow.map((v) => (v === -Infinity ? 0 : Math.exp(v - maxL)));
        const sumE = expRow.reduce((a, b) => a + b, 0);
        const weightRow = expRow.map((e) => (sumE > 0 ? round3(e / sumE) : 0));

        dotScores.push(dotRow);
        scaledScores.push(scaledRow);
        attentionWeights.push(weightRow);
      }

      // Weighted sum of Values: Z_i = sum_j (alpha_ij * V_j)
      const weightedValues: number[][] = [];
      for (let i = 0; i < nTokens; i++) {
        const z = new Array(SLICE_DIM).fill(0);
        for (let j = 0; j <= i; j++) {
          const w = attentionWeights[i][j];
          for (let d = 0; d < SLICE_DIM; d++) {
            z[d] += w * values[j][d];
          }
        }
        weightedValues.push(z.map(round3));
      }

      heads.push({
        headIndex: headIdx,
        roleName: HEAD_ROLES[headIdx].name,
        roleSummary: HEAD_ROLES[headIdx].summary,
        queries,
        keys,
        values,
        dotScores,
        scaledScores,
        attentionWeights,
        weightedValues,
      });
    }

    // Aggregate multi-head outputs + Residual + LayerNorm + MLP (Expansion -> GELU -> Projection) + Residual
    const mlpStates: MlpTokenState[] = [];
    const nextVectors: number[][] = [];

    for (let i = 0; i < nTokens; i++) {
      // Average across heads for representative multi-head projection
      const attnOut = new Array(SLICE_DIM).fill(0);
      for (let h = 0; h < nHeads; h++) {
        for (let d = 0; d < SLICE_DIM; d++) {
          attnOut[d] += heads[h].weightedValues[i][d] / nHeads;
        }
      }
      const roundedAttnOut = attnOut.map(round3);

      // Residual Add #1
      const residualPreMlp = currentVectors[i].map((v, d) => round3(v + roundedAttnOut[d] * 0.45));

      // LayerNorm before MLP
      const ln = layerNorm(residualPreMlp);

      // MLP Expansion (768 -> 3072, inspectable 12-neuron slice)
      const expandRand = seededRandom((layerIdx + 1) * 503 + 17);
      const expandedPreGelu: number[] = [];
      for (let m = 0; m < MLP_SLICE; m++) {
        let sum = (expandRand() - 0.5) * 0.6;
        for (let d = 0; d < SLICE_DIM; d++) {
          sum += ln.out[d] * ((expandRand() * 2 - 1) * 0.85);
        }
        expandedPreGelu.push(round3(Math.max(-3.0, Math.min(3.0, sum))));
      }

      // GELU activation
      const expandedPostGelu = expandedPreGelu.map((x) => round3(gelu(x)));

      // MLP Projection back (3072 -> 768)
      const projRand = seededRandom((layerIdx + 1) * 709 + 29);
      const projectedOut: number[] = [];
      for (let d = 0; d < SLICE_DIM; d++) {
        let sum = 0;
        for (let m = 0; m < MLP_SLICE; m++) {
          sum += expandedPostGelu[m] * ((projRand() * 2 - 1) * 0.35);
        }
        projectedOut.push(round3(sum));
      }

      // Residual Add #2
      const finalBlockOutput = residualPreMlp.map((v, d) => round3(v + projectedOut[d] * 0.4));

      mlpStates.push({
        tokenIndex: i,
        inputAfterAttention: roundedAttnOut,
        residualPreMlp,
        layerNormMean: ln.mean,
        layerNormStd: ln.std,
        layerNormOut: ln.out,
        expandedPreGelu,
        expandedPostGelu,
        projectedOut,
        finalBlockOutput,
      });

      nextVectors.push(finalBlockOutput);
    }

    layers.push({
      layerIndex: layerIdx,
      heads,
      mlpStates,
    });

    currentVectors = nextVectors;
  }

  const rawCandidates = getContextualCandidates(prompt);
  const candidates = applyDecodingStrategy(rawCandidates, temperature, samplingMode, topK, topP);

  return {
    prompt,
    tokens,
    layers,
    candidates,
    config: {
      nLayers,
      nHeads,
      dModel: 768,
      dHead: 64,
      dMlp: 3072,
      vocabSize: 50257,
    },
  };
}
