export const runtime = 'nodejs';

export async function POST(request) {
  try {
    const form = await request.formData();
    const file = form.get('file');
    if (!file || typeof file === 'string') {
      return Response.json({ error: 'Missing file' }, { status: 400 });
    }

    const arrayBuffer = await file.arrayBuffer();
    const repoId = process.env.HF_REPO_ID;
    const endpoint = process.env.HF_INFERENCE_URL || (repoId ? `https://api-inference.huggingface.co/models/${repoId}` : null);

    if (!process.env.HF_API_TOKEN) {
      return Response.json({ error: 'HF_API_TOKEN is not set on the server' }, { status: 500 });
    }

    if (!endpoint) {
      return Response.json({ error: 'HF_INFERENCE_URL or HF_REPO_ID must be configured' }, { status: 500 });
    }

    const hfRes = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${process.env.HF_API_TOKEN}`,
        'Content-Type': 'application/octet-stream',
        'Accept': 'application/json'
      },
      body: Buffer.from(arrayBuffer)
    });

    // Pass-through response body and content-type
    const text = await hfRes.text();
    const contentType = hfRes.headers.get('content-type') || 'application/json';
    return new Response(text, { status: hfRes.status, headers: { 'Content-Type': contentType } });
  } catch (err) {
    return Response.json({ error: err?.message || String(err) }, { status: 500 });
  }
}

export async function GET() {
  return Response.json({ status: 'ok' });
}
