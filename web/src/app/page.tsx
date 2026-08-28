"use client";

import { useState } from "react";
import { motion } from "framer-motion";

const scenarios = [
  {
    eyebrow: "AGRICULTURE",
    title: "Crop Health Analysis",
    description:
      "Assess crop health using NDVI, NDRE, NDMI and complementary satellite evidence.",
    meta: "Spectral analysis",
    icon: "🌾",
  },
  {
    eyebrow: "CHANGE DETECTION",
    title: "Detect What Changed",
    description:
      "Compare satellite imagery across time to identify vegetation loss, construction and land-use change.",
    meta: "Multi-temporal",
    featured: true,
    icon: "◈",
  },
  {
    eyebrow: "MULTIMODAL",
    title: "Optical + SAR Fusion",
    description:
      "Combine Sentinel-1 SAR and Sentinel-2 optical imagery for analysis even under cloud cover.",
    meta: "Cross-modal",
    icon: "◉",
  },
];

function Icon({
  children,
  size = 20,
}: {
  children: React.ReactNode;
  size?: number;
}) {
  return (
    <span
      style={{
        width: size,
        height: size,
        display: "inline-flex",
        alignItems: "center",
        justifyContent: "center",
      }}
    >
      {children}
    </span>
  );
}

export default function Home() {
  const [query, setQuery] = useState("");
  const [active, setActive] = useState("Imagery Workspace");

  const execute = () => {
    if (!query.trim()) return;

    // Replace this later with your real FastAPI call.
    console.log("Jatayu query:", query);
  };

  return (
    <main className="min-h-screen bg-[#f7f3ee] text-[#292522]">
      <div className="flex min-h-screen">
        {/* =========================================================
            SIDEBAR
        ========================================================= */}
        <aside className="hidden w-[260px] shrink-0 border-r border-[#e4ddd5] bg-[#faf7f3] lg:flex lg:flex-col">
          <div className="px-7 pt-8">
            {/* Logo */}
            <div className="mb-10">
              <div className="font-serif text-[32px] font-semibold tracking-[-0.04em] text-[#a84300]">
                Jatayu
              </div>

              <div className="mt-0.5 text-[10px] font-semibold uppercase tracking-[0.18em] text-[#665e57]">
                Physics-Grounded Earth Intelligence
              </div>
            </div>

            {/* New Analysis */}
            <motion.button
              whileHover={{ scale: 1.015 }}
              whileTap={{ scale: 0.985 }}
              onClick={() => setQuery("")}
              className="mb-8 flex w-full items-center justify-center gap-2 rounded-full bg-[#a84300] px-5 py-3.5 text-sm font-semibold text-white shadow-[0_8px_24px_rgba(168,67,0,0.18)] transition hover:bg-[#913900]"
            >
              <span className="text-lg">+</span>
              New Analysis
            </motion.button>

            {/* Navigation */}
            <nav className="space-y-1">
              {[
                ["◈", "Imagery Workspace"],
                ["◷", "Historical Archives"],
                ["⌘", "Intelligence Logs"],
                ["▥", "Strategic Reports"],
                ["⊞", "System Status"],
              ].map(([icon, label]) => (
                <button
                  key={label}
                  onClick={() => setActive(label)}
                  className={`flex w-full items-center gap-3 rounded-r-full px-4 py-3 text-left text-[14px] transition ${
                    active === label
                      ? "border-l-[3px] border-[#a84300] bg-[#f2e3d9] font-semibold text-[#a84300]"
                      : "border-l-[3px] border-transparent text-[#625b55] hover:bg-[#f2ede8]"
                  }`}
                >
                  <span className="w-5 text-[17px]">{icon}</span>
                  {label}
                </button>
              ))}
            </nav>

            {/* Recent */}
            <div className="mt-12">
              <div className="mb-4 px-2 text-[10px] font-semibold uppercase tracking-[0.16em] text-[#746b64]">
                Recent Analyses
              </div>

              <div className="space-y-3 px-2">
                {[
                  "Punjab Crop Health",
                  "Coastal Water Detection",
                  "Urban Change · NCR",
                ].map((item) => (
                  <button
                    key={item}
                    className="flex w-full items-center gap-2 text-left text-[12px] text-[#6d655e] transition hover:text-[#a84300]"
                  >
                    <span className="text-[#918981]">◌</span>
                    {item}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Bottom */}
          <div className="mt-auto border-t border-[#e4ddd5] p-6">
            <div className="mb-5 flex items-center gap-3">
              <div className="flex h-9 w-9 items-center justify-center rounded-full bg-[#66734c] text-xs font-bold text-white">
                JA
              </div>

              <div>
                <div className="text-xs font-semibold">Analyst 04</div>
                <div className="mt-0.5 flex items-center gap-1.5 text-[10px] text-[#6c7460]">
                  <span className="h-1.5 w-1.5 rounded-full bg-[#6b7b47]" />
                  System Ready
                </div>
              </div>
            </div>

            <button className="flex w-full items-center gap-3 text-sm text-[#655d56]">
              ⚙ <span>Settings</span>
            </button>

            <button className="mt-4 flex w-full items-center gap-3 text-sm text-[#655d56]">
              ? <span>Support</span>
            </button>
          </div>
        </aside>

        {/* =========================================================
            MAIN
        ========================================================= */}
        <section className="relative flex min-h-screen flex-1 flex-col overflow-hidden">
          {/* subtle top-right glow */}
          <div className="pointer-events-none absolute right-[-180px] top-[-180px] h-[500px] w-[500px] rounded-full bg-[#ead9cc] opacity-30 blur-3xl" />

          {/* Mobile header */}
          <header className="flex items-center justify-between border-b border-[#e4ddd5] bg-[#faf7f3] px-5 py-4 lg:hidden">
            <div>
              <div className="font-serif text-2xl font-semibold text-[#a84300]">
                Jatayu
              </div>
              <div className="text-[8px] uppercase tracking-[0.15em] text-[#746b64]">
                Earth Intelligence
              </div>
            </div>

            <button
              onClick={() => setQuery("")}
              className="rounded-full bg-[#a84300] px-4 py-2 text-xs font-semibold text-white"
            >
              + New Analysis
            </button>
          </header>

          {/* Hero */}
          <div className="mx-auto flex w-full max-w-[1120px] flex-1 flex-col px-6 pb-16 pt-20 sm:px-10 lg:px-16 lg:pt-28">
            <motion.div
              initial={{ opacity: 0, y: 18 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
              className="flex flex-col items-center text-center"
            >
              {/* Earth icon */}
              <div className="mb-7 flex h-11 w-11 items-center justify-center rounded-full bg-[#e7ddd5] text-xl text-[#8b8179]">
                ◉
              </div>

              <h1 className="max-w-[900px] font-serif text-[48px] font-semibold leading-[0.98] tracking-[-0.045em] text-[#292522] sm:text-[60px] lg:text-[72px]">
                Ask the Earth a question.
              </h1>

              <p className="mt-6 max-w-[650px] text-[15px] leading-7 text-[#756d66] sm:text-[16px]">
                Analyze satellite imagery using natural language,
                physics-grounded models and multimodal geospatial intelligence.
              </p>
            </motion.div>

            {/* =====================================================
                QUERY BOX
            ===================================================== */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.12 }}
              className="mx-auto mt-12 w-full max-w-[900px]"
            >
              <div className="flex items-center rounded-[28px] border border-[#e2dcd5] bg-white p-2 pl-6 shadow-[0_12px_45px_rgba(72,52,37,0.06)] transition focus-within:border-[#caa991] focus-within:shadow-[0_14px_50px_rgba(72,52,37,0.09)]">
                <span className="mr-3 text-xl text-[#8b8179]">⌕</span>

                <input
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") execute();
                  }}
                  placeholder="e.g. Is my crop stressed?"
                  className="min-w-0 flex-1 bg-transparent py-3 text-[14px] text-[#292522] outline-none placeholder:text-[#aaa19a]"
                />

                <motion.button
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.97 }}
                  onClick={execute}
                  className="rounded-full bg-[#a84300] px-7 py-3.5 text-[13px] font-semibold text-white transition hover:bg-[#913900]"
                >
                  Execute&nbsp; →
                </motion.button>
              </div>

              {/* Suggested queries */}
              <div className="mt-4 flex flex-wrap justify-center gap-2">
                {[
                  "Detect water bodies",
                  "Check crop stress",
                  "Find recent changes",
                  "Analyze with SAR",
                ].map((suggestion) => (
                  <button
                    key={suggestion}
                    onClick={() => setQuery(suggestion)}
                    className="rounded-full border border-[#e3dcd5] bg-[#faf7f3] px-4 py-2 text-[11px] text-[#716960] transition hover:border-[#c9ad98] hover:bg-white hover:text-[#a84300]"
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            </motion.div>

            {/* =====================================================
                SCENARIO CARDS
            ===================================================== */}
            <div className="mx-auto mt-16 grid w-full max-w-[1050px] gap-5 md:grid-cols-3">
              {scenarios.map((scenario, index) => (
                <motion.button
                  key={scenario.title}
                  initial={{ opacity: 0, y: 24 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{
                    duration: 0.55,
                    delay: 0.2 + index * 0.08,
                  }}
                  whileHover={{ y: -5 }}
                  onClick={() => setQuery(scenario.title)}
                  className={`group relative min-h-[270px] overflow-hidden rounded-[30px] border text-left transition ${
                    scenario.featured
                      ? "border-[#ddd1c6] bg-[#292522] text-white shadow-[0_18px_50px_rgba(41,37,34,0.16)]"
                      : "border-[#e6dfd8] bg-white text-[#292522] shadow-[0_12px_40px_rgba(72,52,37,0.035)]"
                  }`}
                >
                  {/* Featured image treatment */}
                  {scenario.featured && (
                    <>
                      <div className="absolute inset-x-0 top-0 h-[130px] bg-[radial-gradient(circle_at_35%_40%,#7e756c,transparent_28%),linear-gradient(135deg,#403a35,#171614)]" />

                      <div className="absolute inset-x-0 top-0 h-[150px] bg-[linear-gradient(to_bottom,transparent,#292522)]" />

                      {/* simulated satellite lines */}
                      <div className="absolute left-[-20px] top-[55px] h-[1px] w-[130%] rotate-[-8deg] bg-white/10" />
                      <div className="absolute left-[-20px] top-[85px] h-[1px] w-[130%] rotate-[-8deg] bg-white/10" />
                    </>
                  )}

                  <div className="relative flex h-full flex-col p-7">
                    <div className="flex items-start justify-between">
                      <div
                        className={`flex h-11 w-11 items-center justify-center rounded-full text-lg ${
                          scenario.featured
                            ? "border border-white/20 bg-white/10"
                            : "bg-[#ece8df]"
                        }`}
                      >
                        {scenario.icon}
                      </div>

                      <span
                        className={`rounded-full px-3 py-1.5 text-[8px] font-semibold tracking-[0.16em] ${
                          scenario.featured
                            ? "bg-white/10 text-white/70"
                            : "bg-[#f4eee8] text-[#8a7c70]"
                        }`}
                      >
                        {scenario.eyebrow}
                      </span>
                    </div>

                    <div className="mt-auto">
                      <h2
                        className={`font-serif text-[25px] font-semibold tracking-[-0.025em] ${
                          scenario.featured
                            ? "text-white"
                            : "text-[#292522]"
                        }`}
                      >
                        {scenario.title}
                      </h2>

                      <p
                        className={`mt-3 text-[12px] leading-5 ${
                          scenario.featured
                            ? "text-white/65"
                            : "text-[#756d66]"
                        }`}
                      >
                        {scenario.description}
                      </p>

                      <div
                        className={`mt-6 border-t pt-4 text-[10px] font-medium ${
                          scenario.featured
                            ? "border-white/15 text-white/55"
                            : "border-[#eee7e0] text-[#777069]"
                        }`}
                      >
                        ◈ &nbsp; {scenario.meta}
                      </div>
                    </div>
                  </div>
                </motion.button>
              ))}
            </div>

            {/* Footer statement */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.8 }}
              className="mt-auto pt-16 text-center"
            >
              <p className="text-[10px] uppercase tracking-[0.2em] text-[#aaa099]">
                Natural language → satellite evidence → explainable intelligence
              </p>
            </motion.div>
          </div>
        </section>
      </div>
    </main>
  );
}
