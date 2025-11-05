const form = document.getElementById('upload-form');
const fileInput = document.getElementById('file-input');
const preview = document.getElementById('uploaded-preview');
const predBox = document.getElementById('predictions');
const modeSelect = document.getElementById('mode-select');
// token stage selector (shown only for Swin token modes)
const tokenStageSelect = document.getElementById('token-stage-select');
const tokenStageContainer = document.getElementById('token-stage-container');
const enhanceToggle = document.getElementById('enhance-toggle');

// Right-panel multi-slot Grad-CAM visuals
const slider = document.getElementById('slider');

function setSlider(percent){
  percent = Math.max(0, Math.min(100, percent));
  // Apply slider percent to all overlay images as opacity (0.0-1.0)
  const p = percent / 100.0;
  for (let i = 0; i < 4; ++i){
    const ov = document.getElementById('res-overlay-' + i);
    if (ov) ov.style.opacity = String(p);
  }
}

slider.addEventListener('input', () => setSlider(slider.value));
setSlider(50);

// Per-slot upload state: store the File for each preview slot and track explicit slot clicks
let slotFiles = [null, null, null, null];
let currentUploadSlot = null; // when set, the next selected file(s) will be placed starting at this slot

// Make preview slots clickable to target an upload into that slot
function initSlotClickHandlers(){
  for (let i = 0; i < 4; ++i){
    const slotEl = document.getElementById('slot-' + i);
    if (!slotEl) continue;
    slotEl.style.cursor = 'pointer';
    slotEl.addEventListener('click', () => {
      currentUploadSlot = i;
      // trigger file chooser
      fileInput.click();
    });

    // Also wire the centered button for explicit per-tile upload
    const btn = document.getElementById('upload-btn-' + i);
    if (btn){
      btn.addEventListener('click', (ev) => {
        ev.stopPropagation();
        currentUploadSlot = i;
        fileInput.click();
      });
    }
  }
}
window.addEventListener('load', initSlotClickHandlers);

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
  const files = Array.from(fileInput.files || []).slice(0, 4);
  if (files.length === 0) return;

  // If a slot was explicitly selected by click, place files starting at that slot.
  // Otherwise, place each file into the first empty slot.
  let target = currentUploadSlot;
  for (let i = 0; i < files.length; ++i){
    const f = files[i];
    // find slot
    if (target === null || target === undefined){
      // first-empty slot
      target = slotFiles.findIndex(x => x === null);
      if (target === -1) {
        // no empty slot, overwrite the first slot
        target = 0;
      }
    }

  // assign
  slotFiles[target] = f;
  const url = URL.createObjectURL(f);
  const el = document.getElementById('orig-' + target);
  const placeholder = document.getElementById('placeholder-' + target);
  const captionEl = document.getElementById('caption-' + target);
  if (el) el.src = url;
  if (placeholder) placeholder.style.display = 'none';
  if (captionEl) captionEl.textContent = f.name;
  console.debug('[uploader] assigned slot', target, 'file', f.name);

    // prepare next target (if multiple files selected and user clicked a slot, increment)
    if (currentUploadSlot !== null && currentUploadSlot !== undefined){
      target = target + 1;
      if (target > 3) break;
    } else {
      // reset target so next file searches for first-empty slot again
      target = null;
    }
  }

  // reset currentUploadSlot so future selections without click go to first-empty
  currentUploadSlot = null;
});

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  // multiple files handled below

  if (predBox){
    predBox.classList.remove('hidden');
    predBox.textContent = 'Processing...';
  }

  try{
    const mode = modeSelect?.value || 'fusion';
    const stage = tokenStageSelect?.value || '7';
    const enhance = !!enhanceToggle?.checked;
    const perPixel = !!document.getElementById('perpixel-toggle')?.checked;
    const alphaMin = document.getElementById('alpha-min')?.value || '0.0';
    const alphaMax = document.getElementById('alpha-max')?.value || '0.6';

  // Build list of tasks from per-slot files so each slot keeps its identity
    const filesBySlot = slotFiles.map((f, idx) => ({ f, idx })).filter(x => x.f !== null);
    if (filesBySlot.length === 0) return alert('Please select at least one image');

    // Process files concurrently (each task knows its target slot index)
    if (predBox){
      predBox.classList.remove('hidden');
      predBox.textContent = 'Processing...';
    }

    const tasks = filesBySlot.map(({f, idx}) => (async () => {
      console.debug('[predict] starting task for slot', idx, 'file', f.name);
      const fd = new FormData();
      fd.append('file', f);
      const q = new URLSearchParams({ mode, token_stage: stage, enhance: String(enhance), per_pixel: String(perPixel), alpha_min: String(alphaMin), alpha_max: String(alphaMax) });
      try{
        const res = await fetch(`/predict?${q.toString()}`, { method: 'POST', body: fd });
        const out = await res.json();
        console.debug('[predict] response slot', idx, 'status', res.status, out);
        if (!res.ok) throw new Error(out?.error || out?.message || 'Server error');

        // update result in the right panel slot: set base and overlay images
        const overlayImg = document.getElementById('res-overlay-' + idx);
        const baseRight = document.getElementById('res-base-' + idx);
        const caption = document.getElementById('caption-' + idx);
        const orig = document.getElementById('orig-' + idx);
        const placeholder = document.getElementById('placeholder-' + idx);
        if (overlayImg) {
          overlayImg.src = out.gradcam_url;
          // ensure overlay uses current slider opacity value
          const val = parseFloat(slider.value || '50')/100.0;
          overlayImg.style.opacity = String(val);
          overlayImg.style.display = 'block';
        }
        if (baseRight) baseRight.src = out.uploaded_url;
        if (orig) orig.src = out.uploaded_url; // ensure orig uses server-saved version
        if (placeholder) placeholder.style.display = 'none';
        if (caption) {
          const probs = out.probs || {};
          const lines = Object.entries(probs)
            .sort((a,b) => b[1] - a[1])
            .map(([k,v]) => `${k}: ${(v*100).toFixed(1)}%`)
            .join(' | ');
          caption.innerHTML = `<b>${out.pred_label}</b><div style="font-size:12px;color:white;margin-top:4px">${lines}</div>`;
        }
      }catch(err){
        console.error('[predict] error slot', idx, err);
        const caption = document.getElementById('caption-' + idx);
        if (caption) caption.textContent = 'Error: ' + (err.message || err);
      }
    })());

    await Promise.all(tasks);
    if (predBox) predBox.textContent = 'Done';
  }catch(err){
    console.error(err);
    if (predBox) predBox.textContent = 'Error: ' + err.message;
  }
});

// Show/hide token-stage selector depending on selected mode
function updateTokenSelectorVisibility(){
  const mode = modeSelect?.value || 'fusion';
  if (mode === 'swin_patchcam' || mode === 'fusion' || mode === 'swin'){
    tokenStageContainer && (tokenStageContainer.style.display = 'inline-block');
  } else {
    tokenStageContainer && (tokenStageContainer.style.display = 'none');
  }
}
modeSelect?.addEventListener('change', updateTokenSelectorVisibility);
window.addEventListener('load', updateTokenSelectorVisibility);
