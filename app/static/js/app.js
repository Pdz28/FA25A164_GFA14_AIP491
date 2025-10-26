const form = document.getElementById('upload-form');
const fileInput = document.getElementById('file-input');
const preview = document.getElementById('uploaded-preview');
const predBox = document.getElementById('predictions');
const modeSelect = document.getElementById('mode-select');
const tokenStageSelect = document.getElementById('token-stage-select');
const enhanceToggle = document.getElementById('enhance-toggle');

const baseImg = document.getElementById('compare-base');
const overlayImg = document.getElementById('compare-overlay');
const overlayDiv = document.getElementById('overlay');
const handle = document.getElementById('handle');
const compareBox = document.getElementById('compare-box');
const slider = document.getElementById('slider');

function setSlider(percent){
  percent = Math.max(0, Math.min(100, percent));
  // Use CSS variable to clip overlay via clip-path and move handle
  compareBox.style.setProperty('--clip', percent + '%');
}

slider.addEventListener('input', () => setSlider(slider.value));
setSlider(50);

fileInput.addEventListener('change', () => {
  const f = fileInput.files?.[0];
  if (f){
    const url = URL.createObjectURL(f);
    preview.src = url;
    baseImg.src = url; // show original in compare box immediately
  }
});

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const file = fileInput.files?.[0];
  if (!file) return alert('Please select an image');

  const data = new FormData();
  data.append('file', file);

  predBox.classList.remove('hidden');
  predBox.textContent = 'Processing...';

  try{
  const mode = modeSelect?.value || 'fusion';
  const stage = tokenStageSelect?.value || 'hr';
  const enhance = !!enhanceToggle?.checked;
  const q = new URLSearchParams({ mode, token_stage: stage, enhance: String(enhance) });
  const res = await fetch(`/predict?${q.toString()}`, { method: 'POST', body: data });
    if (!res.ok) throw new Error('Server error');
    const out = await res.json();

    // Update UI
    preview.src = out.uploaded_url;
    baseImg.src = out.uploaded_url;
    overlayImg.src = out.gradcam_url;
    setSlider(50);

    const probs = out.probs || {};
    const lines = Object.entries(probs)
      .sort((a,b) => b[1] - a[1])
      .map(([k,v]) => `${k}: ${(v*100).toFixed(1)}%`)
      .join('\n');

    predBox.innerHTML = `<b>Prediction:</b> ${out.pred_label}<br/><pre style="margin:6px 0 0">${lines}</pre>`;
  }catch(err){
    console.error(err);
    predBox.textContent = 'Error: ' + err.message;
  }
});
