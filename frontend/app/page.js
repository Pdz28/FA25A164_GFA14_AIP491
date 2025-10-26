'use client';

import { useState } from 'react';

export default function HomePage() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const onFileChange = (e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    setFile(f);
    setResult(null);
    setError(null);
    const url = URL.createObjectURL(f);
    setPreview(url);
  };

  const onSubmit = async (e) => {
    e.preventDefault();
    if (!file) return;
    setLoading(true);
    setResult(null);
    setError(null);
    try {
      const form = new FormData();
      form.append('file', file);
      const res = await fetch('/api/predict', { method: 'POST', body: form });
      const contentType = res.headers.get('content-type') || '';
      let payload = null;
      if (contentType.includes('application/json')) {
        payload = await res.json();
      } else {
        const text = await res.text();
        try { payload = JSON.parse(text); } catch { payload = { raw: text }; }
      }
      if (!res.ok) {
        throw new Error(payload?.error || payload?.message || 'Inference API error');
      }
      setResult(payload);
    } catch (err) {
      setError(err.message || String(err));
    } finally {
      setLoading(false);
    }
  };

  const pretty = (data) => JSON.stringify(data, null, 2);

  // Try to extract top-1 classification if returned as an array of {label, score}
  const top1 = Array.isArray(result)
    ? result.slice().sort((a,b) => (b.score ?? 0) - (a.score ?? 0))[0]
    : null;

  return (
    <main style={{ maxWidth: 880, margin: '40px auto', padding: '0 16px' }}>
      <h1 style={{ fontSize: 28, marginBottom: 8 }}>Skin Cancer Classifier</h1>
      <p style={{ color: '#555', marginTop: 0 }}>Upload an image; this app will call the Hugging Face Inference API.</p>

      <form onSubmit={onSubmit} style={{ marginTop: 20 }}>
        <input type="file" accept="image/*" onChange={onFileChange} />
        <button type="submit" disabled={!file || loading} style={{ marginLeft: 12 }}>
          {loading ? 'Predicting…' : 'Predict'}
        </button>
      </form>

      {preview && (
        <section style={{ marginTop: 24 }}>
          <h3 style={{ margin: '8px 0' }}>Preview</h3>
          <img src={preview} alt="preview" style={{ maxWidth: '100%', borderRadius: 8, boxShadow: '0 2px 10px rgba(0,0,0,0.1)' }} />
        </section>
      )}

      {error && (
        <section style={{ marginTop: 24, color: '#b00020' }}>
          <strong>Error:</strong> {error}
        </section>
      )}

      {top1 && (
        <section style={{ marginTop: 24 }}>
          <h3>Top-1 Prediction</h3>
          <div style={{ padding: 12, border: '1px solid #eee', borderRadius: 8 }}>
            <div><strong>Label:</strong> {top1.label}</div>
            <div><strong>Score:</strong> {(top1.score * 100).toFixed(2)}%</div>
          </div>
        </section>
      )}

      {result && (
        <section style={{ marginTop: 24 }}>
          <h3>Raw Response</h3>
          <pre style={{ background: '#fafafa', padding: 12, borderRadius: 8, overflowX: 'auto' }}>{pretty(result)}</pre>
        </section>
      )}

      <footer style={{ marginTop: 48, color: '#888' }}>
        <small>Tip: Set HF_API_TOKEN and HF_REPO_ID (or HF_INFERENCE_URL) in your Vercel project settings.</small>
      </footer>
    </main>
  );
}
