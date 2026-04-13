import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Object Detection Optimizer",
  description: "Frontend for comparing object detection inference runtimes."
};

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
