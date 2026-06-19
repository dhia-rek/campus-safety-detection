import React, { useMemo } from "react";

const W = 1000, H = 240, PAD = 16;
const LINES = [
  { key: "audio_z",  color: "#f0a93b", label: "sound"  },
  { key: "verbal_z", color: "#ff5d7a", label: "verbal" },
  { key: "visual_z", color: "#58a6ff", label: "visual" },
];

export default function ScoreChart({ series, threshold, frame, nFrames }) {
  const { paths, y, n } = useMemo(() => {
    const cols = LINES.map((l) => series[l.key]).filter(Boolean);
    const n = cols[0]?.length || 0;
    let min = Math.min(threshold, 0), max = Math.max(threshold, 1);
    for (const c of cols) for (const v of c) { if (v < min) min = v; if (v > max) max = v; }
    const span = max - min || 1;
    const y = (v) => PAD + (1 - (v - min) / span) * (H - 2 * PAD);
    const x = (i) => (n <= 1 ? 0 : (i / (n - 1)) * W);
    const paths = LINES.map((l) => {
      const c = series[l.key];
      if (!c) return { ...l, d: "" };
      const d = c.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
      return { ...l, d };
    });
    return { paths, y, n };
  }, [series, threshold]);

  if (!n) return <div className="chart empty">No score data</div>;

  const px = nFrames > 1 ? (frame / (nFrames - 1)) * W : 0;

  return (
    <div className="chart">
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" className="chart-svg">
        {/* threshold line */}
        <line x1="0" x2={W} y1={y(threshold)} y2={y(threshold)} className="thr-line" />
        {paths.map((p) => (
          <path key={p.key} d={p.d} fill="none" stroke={p.color} strokeWidth="2" />
        ))}
        {/* playhead */}
        <line x1={px} x2={px} y1="0" y2={H} className="playhead" />
      </svg>
      <div className="legend">
        {LINES.map((l) => (
          <span key={l.key} className="leg"><i style={{ background: l.color }} />{l.label}</span>
        ))}
        <span className="leg"><i className="dash" />threshold</span>
      </div>
    </div>
  );
}
