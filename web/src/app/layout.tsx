import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Jatayu — Satellite Analysis Assistant",
  description: "Ask the Earth a question.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
