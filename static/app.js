const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

const statusEl = document.getElementById('status');
const datasetEl = document.getElementById('dataset');
const labelEl = document.getElementById('label');
const samplesEl = document.getElementById('samples');
const waitEl = document.getElementById('wait');

const startBtn = document.getElementById('start');
const stopBtn = document.getElementById('stop');

let recording = false;
let samplesCollected = 0;
let targetSamples = 0;
let currentLabel = '';
let currentDataset = '';
let pending = false;

function setStatus(text) {
  statusEl.textContent = text;
}

function normalizeLandmarks(landmarks) {
  const base = landmarks[0];
  const out = [];
  for (const lm of landmarks) {
    out.push(lm.x - base.x, lm.y - base.y, lm.z - base.z);
  }
  return out;
}

async function saveSample(features, label, dataset) {
  pending = true;
  try {
    const res = await fetch('/save_sample', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ features, label, dataset })
    });
    const data = await res.json();
    if (data.ok) {
      samplesCollected += 1;
      setStatus(`Recording ${currentLabel}... ${samplesCollected}/${targetSamples}`);
      if (samplesCollected >= targetSamples) {
        recording = false;
        setStatus(`Finished recording ${currentLabel}`);
      }
    } else {
      setStatus(`Save failed: ${data.error || 'unknown error'}`);
    }
  } catch (err) {
    setStatus('Save failed: network error');
  } finally {
    pending = false;
  }
}

startBtn.addEventListener('click', () => {
  const label = (labelEl.value || '').trim().toUpperCase();
  const dataset = (datasetEl.value || '').trim();
  if (!label) {
    setStatus('Please enter a label.');
    return;
  }
  if (!dataset) {
    setStatus('Please enter a dataset name.');
    return;
  }
  const samples = parseInt(samplesEl.value, 10);
  if (!Number.isFinite(samples) || samples <= 0) {
    setStatus('Samples must be a positive integer.');
    return;
  }
  const wait = parseInt(waitEl.value, 10);
  if (!Number.isFinite(wait) || wait < 0) {
    setStatus('Wait must be a non-negative integer.');
    return;
  }

  currentLabel = label;
  currentDataset = dataset;
  targetSamples = samples;
  samplesCollected = 0;
  setStatus('Stabilizing hand...');
  setTimeout(() => {
    recording = true;
    setStatus(`Recording ${currentLabel}... 0/${targetSamples}`);
  }, wait);
});

stopBtn.addEventListener('click', () => {
  recording = false;
  setStatus('Recording stopped.');
});

async function setupCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
  video.srcObject = stream;
  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      video.play();
      resolve();
    };
  });
}

const hands = new Hands({
  locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
});
hands.setOptions({
  maxNumHands: 1,
  modelComplexity: 1,
  minDetectionConfidence: 0.7,
  minTrackingConfidence: 0.7,
  selfieMode: true
});

hands.onResults((results) => {
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  ctx.save();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(results.image, 0, 0, canvas.width, canvas.height);

  if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
    const landmarks = results.multiHandLandmarks[0];
    drawConnectors(ctx, landmarks, HAND_CONNECTIONS, { color: '#00FF00', lineWidth: 2 });
    drawLandmarks(ctx, landmarks, { color: '#FF0000', lineWidth: 1 });

    if (recording && currentLabel && !pending && samplesCollected < targetSamples) {
      const features = normalizeLandmarks(landmarks);
      saveSample(features, currentLabel, currentDataset);
    }
  }

  ctx.restore();
});

async function loop() {
  await hands.send({ image: video });
  requestAnimationFrame(loop);
}

(async () => {
  try {
    await setupCamera();
    loop();
  } catch (err) {
    setStatus('Camera permission denied or unavailable.');
  }
})();
