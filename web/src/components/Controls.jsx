import React from "react";
import { Clapperboard, Play, Pause, Bell, Video, Clock } from "lucide-react";

// Group scenes by type so the picker differentiates a fight from a calm/campus clip.
function groupScenes(scenes) {
  const cat = (n) =>
    n.startsWith("fight") ? "Violence (fights)"
    : n.startsWith("calm") ? "Calm / control"
    : /^\d/.test(n)        ? "Campus CCTV (ShanghaiTech)"
    : "Other";
  const order = ["Violence (fights)", "Calm / control", "Campus CCTV (ShanghaiTech)", "Other"];
  const groups = {};
  for (const s of scenes) (groups[cat(s.scene)] ||= []).push(s);
  return order.filter((g) => groups[g]).map((g) => [g, groups[g]]);
}

export default function Controls({
  scenes, scene, setScene,
  threshold, setThreshold, fps, setFps,
  playing, setPlaying, frame, setFrame, nFrames,
}) {
  return (
    <div className="controls">
      <label>
        <span className="lbl"><Clapperboard size={13} /> Scene</span>
        <select value={scene} onChange={(e) => setScene(e.target.value)}>
          {groupScenes(scenes).map(([label, items]) => (
            <optgroup key={label} label={label}>
              {items.map((s) => (
                <option key={s.scene} value={s.scene}>
                  {s.scene}{s.audio ? " · audio" : ""} ({s.n_frames}f)
                </option>
              ))}
            </optgroup>
          ))}
        </select>
      </label>

      <button className={"play " + (playing ? "on" : "")} onClick={() => setPlaying(!playing)} disabled={!nFrames}>
        {playing ? <><Pause size={15} /> Pause</> : <><Play size={15} /> Play</>}
      </button>

      <label className="range">
        <span className="lbl"><Bell size={13} /> Threshold <b>{threshold.toFixed(2)}</b></span>
        <input type="range" min="0" max="5" step="0.05"
          value={threshold} onChange={(e) => setThreshold(+e.target.value)} />
      </label>

      <label className="range">
        <span className="lbl"><Video size={13} /> FPS <b>{fps}</b></span>
        <input type="range" min="1" max="24" step="1"
          value={fps} onChange={(e) => setFps(+e.target.value)} />
      </label>

      <label className="range grow">
        <span className="lbl"><Clock size={13} /> Frame <b>{frame}</b>/{Math.max(0, nFrames - 1)}</span>
        <input type="range" min="0" max={Math.max(0, nFrames - 1)} step="1"
          value={frame} onChange={(e) => setFrame(+e.target.value)} disabled={!nFrames} />
      </label>
    </div>
  );
}
