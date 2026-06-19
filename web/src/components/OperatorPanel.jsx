import React, { useEffect, useRef, useState } from "react";
import {
  Eye, Volume2, MessageSquareText, Brain,
  ShieldAlert, ShieldCheck, RefreshCw,
} from "lucide-react";
import { getDecision } from "../api.js";

// The right-rail "brain": live threat status, live per-agent signal meters, and
// the coordinator's verdict (auto-runs when an alert begins, plus manual re-run).

const SIGNALS = [
  { key: "visual", Icon: Eye,               label: "Vision", varc: "--blue"  },
  { key: "sound",  Icon: Volume2,           label: "Sound",  varc: "--amber" },
  { key: "speech", Icon: MessageSquareText, label: "Speech", varc: "--terra" },
];

function Meter({ Icon, label, value, threshold, varc, available }) {
  const color = `var(${varc})`;
  if (!available) {
    // No data source for this modality (e.g. video-only footage has no audio).
    return (
      <div className="sig off">
        <span className="sig-ic"><Icon size={15} /></span>
        <span className="sig-label">{label}</span>
        <div className="sig-bar" />
        <span className="sig-val">—</span>
      </div>
    );
  }
  const pct = Math.max(0, Math.min(1, value / 4)) * 100;
  const hot = value >= threshold;
  return (
    <div className={"sig " + (hot ? "hot" : "")}>
      <span className="sig-ic" style={{ color }}><Icon size={15} /></span>
      <span className="sig-label">{label}</span>
      <div className="sig-bar">
        <span className="sig-fill" style={{ width: pct + "%", background: color }} />
      </div>
      <span className="sig-val">{value.toFixed(2)}</span>
    </div>
  );
}

export default function OperatorPanel({ scene, frame, threshold, signals, avail = {}, alert }) {
  const [decision, setDecision] = useState(null);
  const [loading, setLoading]   = useState(false);
  const prevAlert = useRef(false);

  const run = async () => {
    if (!scene) return;
    setLoading(true);
    try { setDecision(await getDecision(scene, frame, threshold)); }
    catch (e) { setDecision({ error: String(e) }); }
    finally { setLoading(false); }
  };

  // Auto-run on the rising edge of an alert (don't hammer it every frame).
  useEffect(() => {
    if (alert && !prevAlert.current) run();
    prevAlert.current = alert;
  }, [alert]); // eslint-disable-line react-hooks/exhaustive-deps

  // Reset when the scene changes.
  useEffect(() => { setDecision(null); prevAlert.current = false; }, [scene]);

  const dec = decision?.decision;

  return (
    <div className="operator">
      {/* THREAT STATUS */}
      <div className={"threat " + (alert ? "is-alert" : "is-ok")}>
        {alert ? <ShieldAlert size={22} /> : <ShieldCheck size={22} />}
        <div className="threat-txt">
          <b>{alert ? "ABNORMAL" : "NORMAL"}</b>
          <span className="threat-sub">frame {frame} · fused {signals.fused.toFixed(2)}</span>
        </div>
        {dec && <span className={"sev sev-" + dec.severity}>{dec.severity}</span>}
      </div>

      {/* LIVE AGENT SIGNALS */}
      <div className="panel-block">
        <h3>Agent signals</h3>
        {SIGNALS.map((s) => (
          <Meter key={s.key} Icon={s.Icon} label={s.label}
                 value={signals[s.key]} threshold={threshold} varc={s.varc}
                 available={avail[s.key] !== false} />
        ))}
        <div className="thr-note">threshold {threshold.toFixed(2)} · “—” = no input source</div>
      </div>

      {/* COORDINATOR VERDICT */}
      <div className="panel-block">
        <div className="coord-head">
          <h3><Brain size={15} /> Coordinator</h3>
          <button onClick={run} disabled={!scene || loading}>
            <RefreshCw size={13} className={loading ? "spin" : ""} /> {loading ? "…" : "run"}
          </button>
        </div>

        {decision?.error && <div className="banner error">{decision.error}</div>}
        {!decision && !loading && (
          <p className="muted small">Runs automatically on an alert — or click <b>run</b>.</p>
        )}
        {loading && <p className="muted small">Analyzing frame {frame}…</p>}

        {dec && (
          <div className="verdict">
            <div className="verdict-row">
              <span className={"incident " + (dec.is_incident ? "yes" : "no")}>
                {dec.is_incident ? "INCIDENT" : "no incident"}
              </span>
              <span className="src">via {dec.source}</span>
            </div>
            {dec.modalities?.length > 0 && (
              <div className="verdict-mods">{dec.modalities.join(" · ")}</div>
            )}
            {dec.report && <p className="report">{dec.report}</p>}
            {dec.action && <p className="action"><b>Action.</b> {dec.action}</p>}
          </div>
        )}
      </div>
    </div>
  );
}
