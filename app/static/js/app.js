const form = document.getElementById('upload-form');
const fileInput = document.getElementById('file-input');
const slider = document.getElementById('slider');
const fileCount = document.getElementById('file-count');
const progressDiv = document.getElementById('progress');

const models = ['fusion', 'effnet', 'swin'];
let selectedFiles = [];

function setSlider(percent){
  percent = Math.max(0, Math.min(100, percent));
  const p = percent / 100.0;
  document.querySelectorAll('.overlay-img').forEach(ov => {
    ov.style.opacity = String(p);
  });
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
  const files = Array.from(fileInput.files || []).slice(0, 10);
  selectedFiles = files;
  
  console.log('[fileInput change] Selected files:', files.length);
  
  if (files.length === 0) {
    fileCount.textContent = '';
    models.forEach(model => {
      document.getElementById('orig-row-' + model).innerHTML = '';
    });
    return;
  }
  
  fileCount.textContent = `${files.length} file(s) selected`;
  
  // Clear and show preview for each file in all model rows
  models.forEach(model => {
    const row = document.getElementById('orig-row-' + model);
    row.innerHTML = '';
    
    files.forEach((file, idx) => {
      const imgCard = document.createElement('div');
      imgCard.className = 'image-card';
      imgCard.innerHTML = `
        <div class="image-container">
          <img id="orig-${model}-${idx}" alt="Image ${idx + 1}" />
        </div>
        <div class="image-label">Image ${idx + 1}</div>
      `;
      row.appendChild(imgCard);
      
      // Load preview
      const url = URL.createObjectURL(file);
      const img = document.getElementById(`orig-${model}-${idx}`);
      if (img) img.src = url;
    });
  });
  
  console.log('[fileInput change] Preview created for all models');
});


form.addEventListener('submit', async (e) => {
  e.preventDefault();
  
  console.log('[form submit] Starting, selectedFiles:', selectedFiles.length);
  
  if (selectedFiles.length === 0) return alert('Please select at least one image');
  
  // Clear result rows
  models.forEach(model => {
    document.getElementById('result-row-' + model).innerHTML = '';
  });
  
  progressDiv.style.display = 'block';
  progressDiv.textContent = 'Processing images...';
  
  try {
    // Process each file
    for (let idx = 0; idx < selectedFiles.length; idx++) {
      const file = selectedFiles[idx];
      progressDiv.textContent = `Processing image ${idx + 1} of ${selectedFiles.length}...`;
      
      // Create result cards for each model
      models.forEach(model => {
        const resultRow = document.getElementById('result-row-' + model);
        const resultCard = document.createElement('div');
        resultCard.className = 'image-card';
        resultCard.innerHTML = `
          <div class="result-container">
            <div class="image-wrapper">
              <img id="res-base-${model}-${idx}" class="base-img" alt="Result" />
              <img id="res-overlay-${model}-${idx}" class="overlay-img" alt="Overlay" />
            </div>
          </div>
          <div class="prediction-label" id="pred-${model}-${idx}">Processing...</div>
        `;
        resultRow.appendChild(resultCard);
      });
      
      // Make API call
      const fd = new FormData();
      fd.append('file', file);
      
      const res = await fetch('/predict_all_models', { method: 'POST', body: fd });
      const results = await res.json();
      
      console.log(`[predict_all_models] results for image ${idx}:`, results);
      
      // Update each model's results
      models.forEach(model => {
        const result = results[model];
        if (!result) return;
        
        const resBase = document.getElementById(`res-base-${model}-${idx}`);
        const resOverlay = document.getElementById(`res-overlay-${model}-${idx}`);
        const predLabel = document.getElementById(`pred-${model}-${idx}`);
        
        if (result.error) {
          if (predLabel) {
            predLabel.innerHTML = `<span style="color:#f88">Error</span>`;
          }
          return;
        }
        
        // Update images
        if (resBase && result.uploaded_url) resBase.src = result.uploaded_url;
        if (resOverlay && result.gradcam_url) {
          resOverlay.src = result.gradcam_url;
          resOverlay.style.display = 'block';
          const val = parseFloat(slider.value || '50') / 100.0;
          resOverlay.style.opacity = String(val);
        }
        
        // Update prediction label
        if (predLabel && result.pred_label && result.probs) {
          const predClass = result.pred_label;
          const predProb = (result.probs[predClass] * 100).toFixed(1);
          const classColor = predClass.toLowerCase() === 'malignant' ? '#ff4444' : '#44ff44';
          
          predLabel.innerHTML = `<span style="color:${classColor};font-weight:700">${predClass}</span><br><span style="color:#3b82f6;font-size:18px;font-weight:700">${predProb}%</span>`;
        }
      });
    }
    
    progressDiv.textContent = `Completed! Processed ${selectedFiles.length} image(s)`;
    setTimeout(() => { progressDiv.style.display = 'none'; }, 3000);
    
  } catch (err) {
    console.error(err);
    progressDiv.textContent = 'Error: ' + err.message;
    progressDiv.style.color = '#f88';
  }
});

