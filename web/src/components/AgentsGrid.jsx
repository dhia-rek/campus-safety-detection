import React, { useEffect, useState } from "react";
import {
  Boxes, Eye, Video, Volume2, MessageSquareText, Brain, Bell, Webhook,
} from "lucide-react";
import { getAgents } from "../api.js";

// Dynamic "block agents modules" grid. Renders one card per agent from
// GET /api/agents — so adding a plugin on the backend gives it a UI tile with
// no change here. Cards are grouped by role: perception → coordinator → action.

const ROLES = [
  { key: "perception",  title: "Perception", sub: "sense one modality" },
  { key: "coordinator", title: "Coordinator", sub: "the brain — decides" },
  { key: "action",      title: "Action", sub: "react to the decision" },
];

// Per-agent icon, with a safe fallback for any new/unknown plugin.
const ICONS = {
  vision: Eye, sound: Volume2, speech: MessageSquareText,
  coordinator_llm: Brain, telegram: Bell, webhook: Webhook,
};

export default function AgentsGrid() {
  const [agents, setAgents] = useState([]);
  const [err, setErr] = useState("");

  useEffect(() => {
    getAgents()
      .then((d) => setAgents(d.agents || []))
      .catch((e) => setErr(String(e)));
  }, []);

  return (
    <section className="cell wide">
      <h2><Boxes size={15} /> Agents <span className="muted">— active modules (live from /api/agents)</span></h2>
      {err && <div className="banner error">⚠ {err}</div>}

      <div className="agent-roles">
        {ROLES.map((role) => {
          const items = agents.filter((a) => a.role === role.key);
          return (
            <div key={role.key} className="agent-role">
              <div className="agent-role-head">
                <b>{role.title}</b>
                <span className="muted">{role.sub}</span>
              </div>
              <div className="agent-cards">
                {items.length === 0 && <div className="agent-empty muted">none</div>}
                {items.map((a) => {
                  const Icon = ICONS[a.name] || Boxes;
                  return (
                    <div key={a.name} className={"agent-card " + (a.enabled ? "" : "off")}>
                      <div className="agent-card-top">
                        <span className="agent-ic"><Icon size={16} /></span>
                        <span
                          className={"agent-dot " + (a.enabled ? "on" : "down")}
                          title={a.enabled ? "enabled" : "disabled"}
                        />
                      </div>
                      <div className="agent-label">{a.label}</div>
                      <div className="agent-desc">{a.description}</div>
                      <span className={"agent-tier tier-" + a.tier}>{a.tier}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}
