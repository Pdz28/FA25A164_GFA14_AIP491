const form = document.getElementById('upload-form');
const fileInput = document.getElementById('file-input');
const preview = document.getElementById('uploaded-preview');
const predBox = document.getElementById('predictions');
const modeSelect = document.getElementById('mode-select');
// token stage selector (shown only for Swin token modes)
const tokenStageSelect = document.getElementById('token-stage-select');
const tokenStageContainer = document.getElementById('token-stage-container');
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

// Fetch service health on load and update UI
async function refreshHealth(){
  const el = document.getElementById('service-status');
  if (!el) return;
  try{
    const res = await fetch('/health');
    if (!res.ok) {
      el.textContent = 'Service not ready';
      el.style.color = '#b00';
      return;
    }
    const j = await res.json();
    const parts = [];
    parts.push(j.loaded_weights || 'no weights');
    parts.push('device: ' + (j.device || 'unknown'));
    parts.push('EffNet: ' + (j.effnet_loaded ? 'available' : 'unavailable'));
    el.textContent = 'Service ready — ' + parts.join(' · ');
    el.style.color = j.effnet_loaded ? '#0a0' : '#444';
  }catch(err){
    el.textContent = 'Service status unavailable';
    el.style.color = '#b00';
  }
}
window.addEventListener('load', () => refreshHealth());

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
  const stage = tokenStageSelect?.value || '7';
  const enhance = !!enhanceToggle?.checked;
    const perPixel = !!document.getElementById('perpixel-toggle')?.checked;
    const alphaMin = document.getElementById('alpha-min')?.value || '0.0';
    const alphaMax = document.getElementById('alpha-max')?.value || '0.6';
    const q = new URLSearchParams({ mode, token_stage: stage, enhance: String(enhance), per_pixel: String(perPixel), alpha_min: String(alphaMin), alpha_max: String(alphaMax) });
  const res = await fetch(`/predict?${q.toString()}`, { method: 'POST', body: data });
    const out = await res.json();
    if (!res.ok) {
      throw new Error(out?.error || out?.message || 'Server error');
    }

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

// Show/hide token-stage selector depending on selected mode
function updateTokenSelectorVisibility(){
  const mode = modeSelect?.value || 'fusion';
  if (mode === 'swin_patchcam' || mode === 'fusion'){
    tokenStageContainer && (tokenStageContainer.style.display = 'inline-block');
  } else {
    tokenStageContainer && (tokenStageContainer.style.display = 'none');
  }
}
modeSelect?.addEventListener('change', updateTokenSelectorVisibility);
window.addEventListener('load', updateTokenSelectorVisibility);
