export const metadata = {
  title: 'Skin Cancer Classifier',
  description: 'Vercel frontend calling Hugging Face Inference API',
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body style={{ fontFamily: 'system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif', margin: 0 }}>
        {children}
      </body>
    </html>
  );
}
