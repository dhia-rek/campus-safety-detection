import React, { useMemo, useRef, useEffect } from "react";
import { MessageSquare } from "lucide-react";

// Live speech transcript. The segment covering the current playback time is
// highlighted; flagged bad-words are shown in red.
export default function Transcript({ segments, time }) {
  const activeIdx = useMemo(
    () => segments.findIndex((s) => time >= s.start && time < s.end),
    [segments, time]
  );
  const listRef = useRef(null);

  useEffect(() => {
    const el = listRef.current?.querySelector(".seg.active");
    if (el) el.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [activeIdx]);

  if (!segments.length) {
    return (
      <div className="transcript empty">
        <MessageSquare size={14} /> No transcript — run <code>src.pipeline.transcribe</code> on a .wav to enable live speech.
      </div>
    );
  }

  return (
    <div className="transcript" ref={listRef}>
      <h3><MessageSquare size={13} /> Live transcript</h3>
      <div className="seg-list">
        {segments.map((s, i) => (
          <div key={i} className={"seg " + (i === activeIdx ? "active" : "")}>
            <span className="ts">{s.start.toFixed(1)}s</span>
            <span className="txt">{highlight(s.text, s.bad_words || [])}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function highlight(text, bad) {
  if (!bad.length) return text;
  // Split on the flagged terms (case-insensitive), wrap matches in <mark>.
  const esc = bad.map((b) => b.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  const re = new RegExp(`(${esc.join("|")})`, "ig");
  return text.split(re).map((part, i) =>
    bad.some((b) => b.toLowerCase() === part.toLowerCase())
      ? <mark key={i}>{part}</mark>
      : <span key={i}>{part}</span>
  );
}
