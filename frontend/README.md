# Vercel Frontend + Hugging Face Inference API

This `frontend/` is a minimal Next.js app designed for deployment on Vercel. It exposes a `/api/predict` route that forwards image bytes to the Hugging Face Inference API and renders results in a simple UI.

## How it works
- UI (`app/page.js`) lets you upload an image and calls `/api/predict`.
- API route (`app/api/predict/route.js`) accepts the image, then:
  - Uses your `HF_API_TOKEN` for authentication
  - Sends raw bytes as `application/octet-stream` to either:
    - `https://api-inference.huggingface.co/models/${HF_REPO_ID}` (public Inference API), or
    - `HF_INFERENCE_URL` (your private Inference Endpoint URL)
- The response (JSON) is rendered back in the UI. If it is an array of `{label, score}`, we also show a top-1 summary.

> Note: The public Inference API only works if your model repo is compatible (e.g., Transformers model/pipeline or a Space). If you only uploaded a `.pth` file without an inference script, use a dedicated Inference Endpoint or keep using your FastAPI backend.

## Local development

Prereqs: Node.js 18+ and pnpm/npm/yarn.

```powershell
# From the repo root
cd frontend
copy .env.example .env.local
# Edit .env.local with your HF token and either HF_REPO_ID or HF_INFERENCE_URL

npm install
npm run dev
# then open http://localhost:3000
```

## Deploy to Vercel

1. Push this repo to GitHub.
2. In Vercel, "Import Project" → select your repo → root directory: `frontend/`.
3. Set Environment Variables on Vercel Project Settings:
   - `HF_API_TOKEN` = your Hugging Face token
   - EITHER `HF_REPO_ID` = `your-username/your-model`
   - OR `HF_INFERENCE_URL` = your Inference Endpoint URL (overrides `HF_REPO_ID`)
4. Deploy.

No extra build config is required; defaults work for Next.js.

## Environment variables

- `HF_API_TOKEN` (required): Token with Inference API access
- `HF_REPO_ID` (optional): Model repo id for public Inference API
- `HF_INFERENCE_URL` (optional): Full URL to a private Inference Endpoint; takes precedence over `HF_REPO_ID`

## FAQ

- Q: My model only has a `.pth`. Will the public API work?
  - A: Not by default. Create a Hugging Face Inference Endpoint for your repo, or package your model as a Transformers pipeline/Space. Then set `HF_INFERENCE_URL` to the endpoint.
- Q: Can I use this with a different task?
  - A: Yes. The proxy just forwards bytes and returns JSON. Adjust the UI rendering for your task-specific response format.
