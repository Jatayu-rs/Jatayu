// frontend/jatayu/app/auth/page.tsx
import React from 'react';
import AuthPortalForms from './createAcc';

export default function IndicTwoCardAuthLayout() {
  return (
    <div className="min-h-screen w-full bg-[#1F1412] flex items-center justify-center p-4 md:p-8 font-sans antialiased select-none selection:bg-[#8E2A12]/20 selection:text-[#8E2A12]">
      
      {/* CENTRAL SPLIT-CARD WRAPPER WITH BACKDROP GLASSMORPHISM */}
      <div className="w-full max-w-5xl bg-white/75 backdrop-blur-xl rounded-3xl p-4 md:p-6 shadow-2xl shadow-black/60 flex flex-col md:flex-row gap-6 items-stretch min-h-[80vh] border border-white/20">
        
        {/* CARD 1: FORM INTERFACE SCOPE (LEFT PANEL) */}
        <div className="w-full md:w-1/2 flex items-center justify-center p-2 md:p-4">
          <div className="w-full max-w-md">
            <AuthPortalForms />
          </div>
        </div>

        {/* CARD 2: THEMATIC INDIC DOME CONTAINER PANEL (RIGHT PANEL) */}
        <div className="w-full md:w-1/2 bg-stone-900/5 backdrop-blur-md rounded-2xl border border-white/40 p-6 flex flex-col justify-between items-center relative overflow-hidden group shadow-inner">
          
          {/* THE SEPIA GRADIENT TOP BACKDROP BAR */}
          <div className="w-full flex items-center justify-between z-10 border-b border-stone-950/10 pb-3">
            <span className="text-[10px] tracking-widest text-[#8E2A12] font-extrabold uppercase">🔱 SOVEREIGN INTEL NODE</span>
            <span className="text-[9px] font-bold text-stone-500 tracking-wider">COMMAND READY</span>
          </div>

          {/* THE MATTE DOME STRUCTURED ARCHED ARTWORK CONTAINER */}
          <div className="w-full max-w-xs aspect-3/4 rounded-t-full overflow-hidden relative border-4 border-[#8E2A12]/15 shadow-xl shadow-stone-950/20 my-4 transition-transform duration-500 group-hover:scale-[1.01]">
            <div 
              className="absolute inset-0 w-full h-full bg-cover bg-center bg-no-repeat transition-transform duration-700 group-hover:scale-105"
              style={{ backgroundImage: "url('/jatayu_hero.jpg')" }}
            />
            <div className="absolute inset-0 bg-gradient-to-t from-stone-950/80 via-transparent to-transparent z-10" />
          </div>

          {/* BOTTOM INFRASTRUCTURE COMPLIANCE FOOTER DESCRIPTIONS */}
          <div className="w-full flex flex-col space-y-2 text-center z-10 border-t border-stone-950/10 pt-4">
            <h2 className="text-xl font-black font-serif tracking-tight text-stone-900 leading-tight">
              Ancient Vigilance. Modern Precision.
            </h2>
            <p className="text-stone-600 text-[11px] leading-relaxed max-w-sm mx-auto font-medium">
              Multi-temporal satellite monitoring core pipelines engineered explicitly for border security intelligence and high-stakes terrain observation layouts.
            </p>
            <div className="text-[9px] text-stone-400 font-bold uppercase tracking-widest pt-1">
              Jatayu Ecosystem Data Shield
            </div>
          </div>

        </div>

      </div>

    </div>
  );
}