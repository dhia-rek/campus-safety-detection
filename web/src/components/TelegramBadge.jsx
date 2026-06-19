import React from "react";
import { Send } from "lucide-react";

export default function TelegramBadge({ telegram }) {
  const { token_set, subscribers } = telegram;
  return (
    <div className="tg-badge">
      <span className="tg-title"><Send size={14} /> Telegram</span>
      <span className={"dot " + (token_set ? "ok" : "off")} />
      <span className="tg-info">
        {token_set ? "token set" : "no token"} · {subscribers} subs
      </span>
    </div>
  );
}
