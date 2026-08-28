// frontend/jatayu/app/auth/createAcc.tsx
'use client';

import React, { useState } from 'react';

export default function AuthPortalForms() {
  const [authMode, setAuthMode] = useState<'signin' | 'signup'>('signin');
  const [loginMethod, setLoginMethod] = useState<'email' | 'phone'>('email');
  
  // Data Capture States
  const [fullName, setFullName] = useState('');
  const [email, setEmail] = useState('');
  const [phone, setPhone] = useState('');
  const [password, setPassword] = useState('');
  const [role, setRole] = useState('Officer');
  const [statusMsg, setStatusMsg] = useState('');

  // Project credential settings explicitly mapped to your dashboard containers
  const PROJECT_REF = "ckmbipofgdtgfsgcrlok";
  const ANON_KEY = "sb_publishable_a8p0zF0NUaIkIHN4mCFXNQ_mwY56";

  // Google OAuth Redirect Handler
  const handleGoogleConnect = () => {
    setStatusMsg("📡 Routing secure authorization stream...");
    const targetUrl = `https://${PROJECT_REF}.supabase.co/auth/v1/authorize?provider=google&redirect_to=${encodeURIComponent(window.location.origin + "/")}`;
    window.location.href = targetUrl;
  };

  return (
    <div className="w-full max-w-md p-6 bg-transparent flex flex-col">
      {/* BRANDING MARKS */}
      <div className="flex flex-col items-center mb-8">
        <div className="flex items-center space-x-2 text-2xl font-bold tracking-tight text-stone-900 font-serif">
          <span className="text-[#8E2A12]">🦅</span>
          <span>Jatayu | <span className="text-[#8E2A12] font-normal">जटायु</span></span>
        </div>
        <p className="text-[10px] font-extrabold tracking-widest text-stone-400 uppercase mt-1">Sovereign Geospatial Command</p>
      </div>

      {/* SEGMENTED CONTROL SLIDER SWITCH */}
      <div className="flex bg-stone-950/5 backdrop-blur-md p-1 rounded-xl border border-stone-950/10 mb-6">
        <button type="button" onClick={() => { setAuthMode('signin'); setStatusMsg(''); }}
                className={`w-1/2 py-2 text-xs font-bold rounded-lg transition-all duration-200 ${authMode === 'signin' ? 'bg-white/90 backdrop-blur-sm text-stone-900 shadow-sm border border-white/40' : 'text-stone-500 hover:text-stone-800'}`}>
          Sign In
        </button>
        <button type="button" onClick={() => { setAuthMode('signup'); setStatusMsg(''); }}
                className={`w-1/2 py-2 text-xs font-bold rounded-lg transition-all duration-200 ${authMode === 'signup' ? 'bg-white/90 backdrop-blur-sm text-stone-900 shadow-sm border border-white/40' : 'text-stone-500 hover:text-stone-800'}`}>
          Create Account
        </button>
      </div>

      {/* METHOD SELECTOR CONTROLS */}
      {authMode === 'signin' && (
        <div className="flex space-x-4 mb-5 border-b border-stone-950/10 pb-3">
          <button type="button" onClick={() => setLoginMethod('email')}
                  className={`text-xs font-bold pb-1 transition-all ${loginMethod === 'email' ? 'border-b-2 border-[#8E2A12] text-[#8E2A12]' : 'text-stone-400'}`}>
            Email Verification
          </button>
          <button type="button" onClick={() => setLoginMethod('phone')}
                  className={`text-xs font-bold pb-1 transition-all ${loginMethod === 'phone' ? 'border-b-2 border-[#8E2A12] text-[#8E2A12]' : 'text-stone-400'}`}>
            Phone OTP
          </button>
        </div>
      )}

      {/* 👑 THE FIX: NATIVE POST FORM STRATEGY BYPASSES ALL CORS FETCH BLOCKS */}
      <form 
        method="POST" 
        action={
          authMode === 'signup' 
            ? `https://${PROJECT_REF}.supabase.co/auth/v1/signup` 
            : `https://${PROJECT_REF}.supabase.co/auth/v1/token?grant_type=password`
        }
        className="flex flex-col space-y-4"
      >
        {/* Hidden headers passed natively inside the form action query pipeline */}
        <input type="hidden" name="apikey" value={ANON_KEY} />
        <input type="hidden" name="redirect_to" value={typeof window !== 'undefined' ? window.location.origin + "/" : "http://localhost:3000/"} />

        {authMode === 'signup' && (
          <>
            <div>
              <label className="block text-[10px] font-bold uppercase text-stone-500 tracking-wider mb-1.5">Full Name</label>
              <input type="text" name="data[full_name]" placeholder="Dr. Devendra Sharma" required value={fullName} onChange={(e) => setFullName(e.target.value)}
                     className="w-full px-4 py-2.5 text-xs bg-white/60 backdrop-blur-sm border border-stone-300/80 rounded-lg focus:outline-none focus:border-[#8E2A12] text-stone-900 placeholder-stone-400" />
            </div>
            <div>
              <label className="block text-[10px] font-bold uppercase text-stone-500 tracking-wider mb-1.5">Profile Classification</label>
              <select name="data[role]" value={role} onChange={(e) => setRole(e.target.value)}
                      className="w-full px-3 py-2.5 text-xs bg-white/60 backdrop-blur-sm border border-stone-300/80 rounded-lg focus:outline-none focus:border-[#8E2A12] text-stone-900">
                <option value="Research Scientist">Research Scientist</option>
                <option value="District Officer">District Officer</option>
                <option value="Agricultural Practitioner">Agricultural Practitioner</option>
              </select>
            </div>
          </>
        )}

        {authMode === 'signin' && loginMethod === 'phone' ? (
          <div>
            <label className="block text-[10px] font-bold uppercase text-stone-500 tracking-wider mb-1.5">Registered Phone Number</label>
            <input type="tel" name="phone" placeholder="+91 XXXXX XXXXX" required value={phone} onChange={(e) => setPhone(e.target.value)}
                   className="w-full px-4 py-2.5 text-xs bg-white/60 backdrop-blur-sm border border-stone-300/80 rounded-lg focus:outline-none focus:border-[#8E2A12] text-stone-900 placeholder-stone-400" />
          </div>
        ) : (
          <div>
            <label className="block text-[10px] font-bold uppercase text-stone-500 tracking-wider mb-1.5">Official Email Address</label>
            <input type="email" name="email" placeholder="officer@gov.in" required value={email} onChange={(e) => setEmail(e.target.value)}
                   className="w-full px-4 py-2.5 text-xs bg-white/60 backdrop-blur-sm border border-stone-300/80 rounded-lg focus:outline-none focus:border-[#8E2A12] text-stone-900 placeholder-stone-400" />
          </div>
        )}

        <div>
          <div className="flex justify-between items-center mb-1.5">
            <label className="text-[10px] font-bold uppercase text-stone-500 tracking-wider">Security Password</label>
            {authMode === 'signin' && <a href="#" className="text-[10px] font-bold text-[#8E2A12] hover:underline">Reset Credentials?</a>}
          </div>
          <input type="password" name="password" placeholder="••••••••" required value={password} onChange={(e) => setPassword(e.target.value)}
                 className="w-full px-4 py-2.5 text-xs bg-white/60 backdrop-blur-sm border border-stone-300/80 rounded-lg focus:outline-none focus:border-[#8E2A12] text-stone-900 placeholder-stone-400" />
        </div>

        <button type="submit"
                className="w-full py-3 bg-[#8E2A12] hover:bg-[#76220E] text-white font-bold text-xs rounded-lg transition-colors shadow-md shadow-[#8E2A12]/10 mt-2">
          {authMode === 'signin' ? 'Access Command Workspace →' : 'Initialize Account Request'}
        </button>
      </form>

      {statusMsg && <div className="mt-4 p-3 bg-stone-900/5 border border-stone-200 rounded-md text-[11px] font-bold text-center text-stone-700">{statusMsg}</div>}

      {/* FEDERATED SINGLE SIGN-ON */}
      <div className="mt-6 pt-5 border-t border-stone-950/10 flex flex-col items-center">
        <span className="text-[9px] font-bold tracking-widest text-stone-400 uppercase mb-3">Federated Single Sign-On</span>
        <button type="button" onClick={handleGoogleConnect}
                className="w-full py-2.5 border border-stone-300 hover:bg-stone-50/50 transition-colors rounded-lg flex items-center justify-center space-x-2 text-xs font-bold text-stone-700">
          <svg className="w-4 h-4" viewBox="0 0 24 24">
            <path fill="#4285F4" d="M23.745 12.27c0-.7-.06-1.4-.19-2.07H12v3.92h6.61c-.3 1.57-1.18 2.9-2.52 3.79v3.13h4.05c2.37-2.17 3.61-5.37 3.61-8.77z"/>
            <path fill="#34A853" d="M12 24c3.24 0 5.97-1.08 7.96-2.91l-4.05-3.13c-1.13.75-2.57 1.21-3.91 1.21-3.01 0-5.56-2.03-6.46-4.77H1.31v3.23C3.29 22.12 7.42 24 12 24z"/>
            <path fill="#FBBC05" d="M5.54 14.4c-.24-.7-.37-1.44-.37-2.2s.13-1.5.37-2.2V6.77H1.31C.48 8.43 0 10.22 0 12s.48 3.57 1.31 5.23l4.23-3.23z"/>
            <path fill="#EA4335" d="M12 4.75c1.77 0 3.35.61 4.6 1.8l3.43-3.43C17.96 1.19 15.24 0 12 0 7.42 0 3.29 1.88 1.31 5.23l4.23 3.23c.9-2.74 3.45-4.77 6.46-4.77z"/>
          </svg>
          <span>Connect with Google</span>
        </button>
      </div>
    </div>
  );
}
