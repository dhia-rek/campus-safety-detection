import React, { useEffect, useRef, useState } from "react";
import { ShieldCheck, AlertTriangle, LineChart, Volume2 } from "lucide-react";
import * as api from "./api.js";
import Controls from "./components/Controls.jsx";
import VideoPanel from "./components/VideoPanel.jsx";
import ScoreChart from "./components/ScoreChart.jsx";
import Transcript from "./components/Transcript.jsx";
import OperatorPanel from "./components/OperatorPanel.jsx";
import AgentsGrid from "./components/AgentsGrid.jsx";
import TelegramBadge from "./components/TelegramBadge.jsx";

export default function App() {
  const [scenes, setScenes]       = useState([]);
  const [scene, setScene]         = useState("");
  const [data, setData]           = useState(null);     // /api/scene payload
  const [segments, setSegments]   = useState([]);       // transcript
  const [telegram, setTelegram]   = useState({ token_set: false, subscribers: 0 });

  const [threshold, setThreshold] = useState(1.8);
  const [fps, setFps]             = useState(12);
  const [playing, setPlaying]     = useState(false);
  const [frame, setFrame]         = useState(0);
  const [error, setError]         = useState("");

  const timer = useRef(null);
  const audioRef = useRef(null);

  // Initial load: scenes + telegram status
  useEffect(() => {
    api.getScenes().then((d) => {
      setScenes(d.scenes || []);
      if (d.scenes?.length) setScene(d.scenes[0].scene);
    }).catch((e) => setError(String(e)));
    api.getTelegram().then(setTelegram).catch(() => {});
  }, []);

  // On scene change: load scores + transcript
  useEffect(() => {
    if (!scene) return;
    setPlaying(false);
    setFrame(0);
    setData(null);
    api.getScene(scene, "fused").then(setData).catch((e) => setError(String(e)));
    api.getTranscript(scene).then((d) => setSegments(d.segments || [])).catch(() => setSegments([]));
  }, [scene]);

  const hasAudioClip = !!data?.has_audio_clip;

  // Playback loop (frame timer) — skipped when an audio clip drives the frames.
  useEffect(() => {
    if (!playing || !data || hasAudioClip) return;
    timer.current = setInterval(() => {
      setFrame((f) => {
        if (f + 1 >= data.n_frames) { setPlaying(false); return f; }
        return f + 1;
      });
    }, 1000 / fps);
    return () => clearInterval(timer.current);
  }, [playing, fps, data, hasAudioClip]);

  // Audio clip = the master clock: play/pause it with the playing state.
  useEffect(() => {
    const a = audioRef.current;
    if (!a || !hasAudioClip) return;
    if (playing) a.play().catch(() => {});
    else a.pause();
  }, [playing, hasAudioClip, scene]);

  // While paused, scrubbing the frame seeks the audio to match.
  useEffect(() => {
    const a = audioRef.current;
    if (!a || !hasAudioClip || playing || !a.duration || !data) return;
    a.currentTime = (frame / Math.max(1, data.n_frames - 1)) * a.duration;
  }, [frame, hasAudioClip, playing, data]);

  // Audio time → frame (keeps the video synced to the sound during playback).
  const onAudioTime = () => {
    const a = audioRef.current;
    if (!a || !playing || !data?.n_frames || !a.duration) return;
    setFrame(Math.min(data.n_frames - 1, Math.floor((a.currentTime / a.duration) * data.n_frames)));
  };

  const series  = data?.series || {};
  const at      = (col) => (series[col]?.[frame] ?? 0);
  const fused   = at(data?.alert_col || "fused_score");
  const alert   = fused >= threshold;
  const time    = fps ? frame / fps : 0;

  const signals = {
    visual: at("visual_z"),
    sound:  at("audio_z"),
    speech: at("verbal_z"),
    fused,
  };

  // A modality is "available" only if its stream carries real (non-zero) data —
  // so video-only footage clearly shows Sound/Speech as "no source", not a
  // misleading 0.00 that looks like "analysed and quiet".
  const hasData = (col) => (series[col] || []).some((v) => v !== 0);
  const avail = {
    visual: hasData("visual_z"),
    sound:  hasData("audio_z"),
    speech: hasData("verbal_z"),
  };

  // Play/Pause that restarts from the beginning when at the end.
  const togglePlay = () => {
    setPlaying((pl) => {
      if (!pl && frame >= (data?.n_frames || 1) - 1) setFrame(0);
      return !pl;
    });
  };

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="logo"><ShieldCheck size={26} /></span>
          <div>
            <h1>Campus Safety</h1>
            <p className="sub">Live monitoring · multi-agent threat detection</p>
          </div>
        </div>
        <TelegramBadge telegram={telegram} />
      </header>

      {error && <div className="banner error"><AlertTriangle size={15} /> {error}</div>}
      {!scenes.length && !error && (
        <div className="banner">No scored scenes found. Run the pipeline, then start the API.</div>
      )}

      <Controls
        scenes={scenes} scene={scene} setScene={setScene}
        threshold={threshold} setThreshold={setThreshold}
        fps={fps} setFps={setFps}
        playing={playing} setPlaying={togglePlay}
        frame={frame} setFrame={setFrame}
        nFrames={data?.n_frames || 0}
      />

      <div className="console">
        {/* STAGE — the hero CCTV feed + score timeline */}
        <section className="stage">
          <VideoPanel scene={scene} frame={frame} nFrames={data?.n_frames || 0} alert={alert} />

          {hasAudioClip && (
            <div className="audio-bar">
              <span className="audio-label"><Volume2 size={14} /> Clip audio</span>
              <audio
                ref={audioRef}
                src={api.audioUrl(scene)}
                onTimeUpdate={onAudioTime}
                onEnded={() => setPlaying(false)}
                onPlay={() => setPlaying(true)}
                onPause={() => setPlaying(false)}
                controls
                preload="auto"
              />
            </div>
          )}

          <div className={"stage-status " + (alert ? "is-alert" : "is-ok")}>
            {alert
              ? <><AlertTriangle size={15} /> <b>Abnormal activity</b> · fused {fused.toFixed(2)} ≥ threshold {threshold.toFixed(2)}</>
              : <><ShieldCheck size={15} /> <b>Normal</b> · fused {fused.toFixed(2)} &lt; threshold {threshold.toFixed(2)}</>}
          </div>

          <div className="stage-chart">
            <h3><LineChart size={14} /> Score timeline — 3 modalities vs threshold</h3>
            <ScoreChart series={series} threshold={threshold} frame={frame} nFrames={data?.n_frames || 0} />
          </div>
        </section>

        {/* RAIL — the brain: status, live signals, coordinator verdict */}
        <aside className="rail">
          <OperatorPanel scene={scene} frame={frame} threshold={threshold} signals={signals} avail={avail} alert={alert} />
        </aside>
      </div>

      {/* EVIDENCE — live transcript */}
      <section className="panel evidence">
        <Transcript segments={segments} time={time} />
      </section>

      {/* MODULES — pluggable agents, live from the registry */}
      <AgentsGrid />
    </div>
  );
}
