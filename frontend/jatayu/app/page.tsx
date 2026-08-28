// app/page.tsx
"use client";
import { useState } from "react";
import { Sidebar } from "./components/Sidebar";
import { ConversationPane } from "./components/ConversationPane";
import { ContextPanel } from "./components/ContextPanel";
import type { ImageRef } from "../src/lib/local-types";

export default function Home() {
  const [images, setImages] = useState<ImageRef[]>([]);

  return (
    <div className="grid h-screen grid-cols-[260px_1fr_320px]">
      <Sidebar />
      <ConversationPane onImagesChange={setImages} />
      <ContextPanel images={images} />
    </div>
  );
}