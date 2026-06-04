import type { Metadata } from 'next';
import { Inter, JetBrains_Mono } from 'next/font/google';
import './globals.css';

const inter = Inter({
  subsets: ['latin'],
  variable: '--font-inter',
  display: 'swap',
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ['latin'],
  variable: '--font-mono',
  display: 'swap',
});

export const metadata: Metadata = {
  title: 'Cityscapes Segmentation Benchmark — SegFormer vs DeepLabV3+',
  description:
    'Interactive benchmarking dashboard comparing SegFormer-B2 (Transformer) against DeepLabV3+ (CNN) for urban scene semantic segmentation on the Cityscapes dataset. Upload images, view side-by-side segmentation masks, and explore detailed accuracy and performance analytics.',
  keywords: [
    'semantic segmentation',
    'cityscapes',
    'segformer',
    'deeplabv3',
    'deep learning',
    'computer vision',
    'benchmark',
  ],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`${inter.variable} ${jetbrainsMono.variable}`}>
      <body className="min-h-screen bg-[#0a0a0f] text-white antialiased">
        {/* Subtle background gradient */}
        <div className="fixed inset-0 -z-10">
          <div className="absolute inset-0 bg-[radial-gradient(ellipse_80%_50%_at_50%_-20%,rgba(120,60,200,0.12),transparent)]" />
          <div className="absolute inset-0 bg-[radial-gradient(ellipse_60%_40%_at_80%_100%,rgba(59,130,246,0.06),transparent)]" />
        </div>
        {children}
      </body>
    </html>
  );
}
