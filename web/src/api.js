// Thin wrapper around the FastAPI backend (src/dashboard/api.py).
/** GET a URL and parse JSON; non-2xx responses throw so callers can surface the error. */
const j = (url) => fetch(url).then((r) => {
  if (!r.ok) throw new Error(`${r.status} ${url}`);
  return r.json();
});

export const getAgents      = ()                       => j("/api/agents");
export const getScenes      = ()                       => j("/api/scenes");
export const getScene       = (scene, mode = "fused")  => j(`/api/scene/${scene}?mode=${mode}`);
export const getTranscript  = (scene)                  => j(`/api/transcript/${scene}`);
export const getDetections  = (scene, idx)             => j(`/api/detections/${scene}/${idx}`);
export const getTelegram    = ()                       => j("/api/telegram/status");
export const frameUrl       = (scene, idx)             => `/api/frame/${scene}/${idx}.jpg`;
export const audioUrl       = (scene)                  => `/api/audio/${scene}.wav`;

/** Dry run: perception + coordinator only. Never sends alerts. */
export const getDecision = (scene, frame, threshold) =>
  j(`/api/decision/${scene}?frame=${frame}&threshold=${threshold}`);

/** Same as getDecision but also runs the action agents (Telegram, e-mail, webhook): can send real alerts. */
export const executeDecision = (scene, frame, threshold) =>
  fetch(`/api/decision/${scene}/execute?frame=${frame}&threshold=${threshold}`, { method: "POST" }).then((r) => {
    if (!r.ok) throw new Error(`${r.status} execute`);
    return r.json();
  });

/** Manually push an alert for a frame; resolves with the backend JSON (even on HTTP errors). */
export const sendAlert = (scene, frame, score, modality) =>
  fetch("/api/alert", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ scene, frame, score, modality }),
  }).then((r) => r.json());
