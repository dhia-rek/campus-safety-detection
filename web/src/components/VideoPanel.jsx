import React, { useEffect, useState } from "react";
import { Radio, ShieldAlert } from "lucide-react";
import { frameUrl, getDetections } from "../api.js";

// The hero CCTV feed with a live YOLO bounding-box overlay. Boxes (detected
// people/objects) turn red when the frame's anomaly score crosses threshold —
// reading as "detected something abnormal here".
export default function VideoPanel({ scene, frame, nFrames, alert }) {
  const [boxes, setBoxes] = useState([]);
  const [dim, setDim]     = useState({ w: 856, h: 480 });

  useEffect(() => {
    if (!scene || !nFrames) { setBoxes([]); return; }
    let active = true;
    getDetections(scene, frame)
      .then((d) => { if (active) setBoxes(d.boxes || []); })
      .catch(() => { if (active) setBoxes([]); });
    return () => { active = false; };
  }, [scene, frame, nFrames]);

  if (!scene || !nFrames) return <div className="video empty">No frames</div>;

  return (
    <div className={"video " + (alert ? "alert" : "")}>
      <img
        src={frameUrl(scene, frame)}
        alt={`frame ${frame}`}
        loading="eager"
        onLoad={(e) => setDim({ w: e.target.naturalWidth || 856, h: e.target.naturalHeight || 480 })}
      />
      {/* viewBox in native pixels + xMidYMid meet => boxes align with object-fit:contain */}
      <svg className="boxes" viewBox={`0 0 ${dim.w} ${dim.h}`} preserveAspectRatio="xMidYMid meet">
        {boxes.map((b, i) => {
          const [x1, y1, x2, y2] = b.bbox;
          return (
            <g key={i} className={"box " + (alert ? "hot" : "")}>
              <rect x={x1} y={y1} width={x2 - x1} height={y2 - y1} rx="2" />
              <text x={x1 + 2} y={y1 - 4}>{b.label} {Math.round(b.conf * 100)}%</text>
            </g>
          );
        })}
      </svg>
      <div className="video-tag"><Radio size={12} /> {scene} · frame {frame} · {boxes.length} detected</div>
      {alert && <div className="video-alert"><ShieldAlert size={14} /> ABNORMAL</div>}
    </div>
  );
}
