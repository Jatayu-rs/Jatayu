// app/components/Sidebar.tsx
"use client";
import { useState } from "react";
import { FlaskConical, UserCog, Tractor, Globe, Settings } from "lucide-react";

const PERSONAS = [
  { key: "scientist", label: "Scientist", icon: FlaskConical },
  { key: "officer", label: "District Officer", icon: UserCog },
  { key: "farmer", label: "Farmer", icon: Tractor },
] as const;

export function Sidebar() {
  const [persona, setPersona] = useState<(typeof PERSONAS)[number]["key"]>("scientist");

  return (
    <aside className="flex h-full flex-col justify-between border-r border-ink/10 bg-cream p-6">
      <div>
        <h1 className="font-serif text-2xl">Jatayu / जटायु</h1>
        <p className="mt-1 text-sm text-ink/50">Satellite Analysis Assistant</p>

        <button className="mt-6 w-full rounded-full bg-ink px-4 py-2 text-sm font-medium text-cream">
          + New conversation
        </button>

        <nav className="mt-8 space-y-1">
          {PERSONAS.map(({ key, label, icon: Icon }) => (
            <button
              key={key}
              onClick={() => setPersona(key)}
              className={`flex w-full items-center gap-2 rounded-md px-3 py-2 text-sm ${
                persona === key ? "border-l-2 border-terracotta bg-white font-medium" : "text-ink/70"
              }`}
            >
              <Icon size={16} />
              {label}
            </button>
          ))}
        </nav>
      </div>

      <div className="space-y-1 border-t border-ink/10 pt-4">
        <button className="flex items-center gap-2 text-sm text-ink/70">
          <Globe size={16} /> Language
        </button>
        <button className="flex items-center gap-2 text-sm text-ink/70">
          <Settings size={16} /> Settings
        </button>
      </div>
    </aside>
  );
}