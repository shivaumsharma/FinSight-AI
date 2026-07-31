"use client";

import { useState } from "react";

export default function Tabs({
  tabs,
}: {
  tabs: { label: string; content: React.ReactNode }[];
}) {
  const [active, setActive] = useState(0);
  return (
    <div>
      <div className="flex gap-5 overflow-x-auto border-b border-border pb-0">
        {tabs.map((t, i) => (
          <button
            key={t.label}
            type="button"
            onClick={() => setActive(i)}
            className={`whitespace-nowrap pb-3 font-mono text-xs font-semibold tracking-wide ${
              i === active
                ? "border-b-2 border-accent text-accent"
                : "text-muted hover:text-text"
            }`}
          >
            {t.label.toUpperCase()}
          </button>
        ))}
      </div>
      <div className="pt-4">{tabs[active].content}</div>
    </div>
  );
}
