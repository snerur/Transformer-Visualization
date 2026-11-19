import React from 'react';

interface VectorStripProps {
  values: number[];
  palette?: 'sky' | 'teal' | 'amber' | 'indigo' | 'diverging';
  heightClass?: string;
  showNumbers?: boolean;
  label?: string;
}

function getCellBackground(val: number, palette: VectorStripProps['palette']): string {
  const clamped = Math.max(-1.6, Math.min(1.6, val));
  const norm = Math.min(1, Math.abs(clamped) / 1.4);

  if (palette === 'sky') {
    return clamped >= 0
      ? `rgba(2, 132, 199, ${0.15 + norm * 0.78})`
      : `rgba(148, 163, 184, ${0.15 + norm * 0.45})`;
  }
  if (palette === 'teal') {
    return clamped >= 0
      ? `rgba(13, 148, 136, ${0.15 + norm * 0.78})`
      : `rgba(148, 163, 184, ${0.15 + norm * 0.45})`;
  }
  if (palette === 'amber') {
    return clamped >= 0
      ? `rgba(217, 119, 6, ${0.16 + norm * 0.78})`
      : `rgba(148, 163, 184, ${0.15 + norm * 0.45})`;
  }
  if (palette === 'indigo') {
    return clamped >= 0
      ? `rgba(79, 70, 229, ${0.15 + norm * 0.78})`
      : `rgba(203, 213, 225, ${0.22 + norm * 0.48})`;
  }
  // Diverging blue (+) vs warm coral (-)
  if (clamped >= 0) {
    return `rgba(2, 132, 199, ${0.12 + norm * 0.76})`;
  }
  return `rgba(225, 29, 72, ${0.12 + norm * 0.68})`;
}

export const VectorStrip: React.FC<VectorStripProps> = ({
  values,
  palette = 'diverging',
  heightClass = 'h-5',
  showNumbers = false,
  label,
}) => {
  return (
    <div className="w-full" aria-label={label}>
      <div
        className={`grid gap-[2px] w-full ${heightClass}`}
        style={{ gridTemplateColumns: `repeat(${values.length}, minmax(0, 1fr))` }}
      >
        {values.map((v, idx) => (
          <div
            key={idx}
            title={`dim ${idx}: ${v >= 0 ? '+' : ''}${v.toFixed(3)}`}
            style={{ backgroundColor: getCellBackground(v, palette) }}
            className="rounded-[2px] flex items-center justify-center transition-colors duration-150"
          >
            {showNumbers && (
              <span
                className={`font-mono text-[10px] tabular-nums leading-none ${
                  Math.abs(v) > 0.72 ? 'text-white font-medium' : 'text-slate-800'
                }`}
              >
                {v >= 0 ? `+${v.toFixed(1)}` : v.toFixed(1)}
              </span>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};
