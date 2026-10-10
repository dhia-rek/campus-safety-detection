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

/** Run the agent pipeline on one frame. NOTE: the backend executes real action agents (may send alerts). */
export const getDecision = (scene, frame, threshold) =>
  j(`/api/decision/${scene}?frame=${frame}&threshold=${threshold}`);

/** Manually push an alert for a frame; resolves with the backend JSON (even on HTTP errors). */
export const sendAlert = (scene, frame, score, modality) =>
  fetch("/api/alert", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ scene, frame, score, modality }),
  }).then((r) => r.json());
