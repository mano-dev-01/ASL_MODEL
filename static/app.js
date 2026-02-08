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
  const ref = landmarks[9];
  const refDx = ref.x - base.x;
  const refDy = ref.y - base.y;
  const refDz = ref.z - base.z;
  const scale = Math.hypot(refDx, refDy, refDz) || 1;
  return landmarks.map((lm) => ({
    x: (lm.x - base.x) / scale,
    y: (lm.y - base.y) / scale,
    z: (lm.z - base.z) / scale
  }));
}

function flattenCoords(normLandmarks) {
  const out = [];
  for (const lm of normLandmarks) {
    out.push(lm.x, lm.y, lm.z);
  }
  return out;
}

function angleAt(a, b, c) {
  const bax = a.x - b.x;
  const bay = a.y - b.y;
  const baz = a.z - b.z;
  const bcx = c.x - b.x;
  const bcy = c.y - b.y;
  const bcz = c.z - b.z;
  const dot = bax * bcx + bay * bcy + baz * bcz;
  const baLen = Math.hypot(bax, bay, baz);
  const bcLen = Math.hypot(bcx, bcy, bcz);
  const denom = baLen * bcLen;
  if (!denom) {
    return 0;
  }
  const cos = Math.min(1, Math.max(-1, dot / denom));
  return Math.acos(cos);
}

function computeAngles(normLandmarks) {
  const idx = (i) => normLandmarks[i];
  return [
    angleAt(idx(1), idx(2), idx(3)),
    angleAt(idx(2), idx(3), idx(4)),
    angleAt(idx(5), idx(6), idx(7)),
    angleAt(idx(6), idx(7), idx(8)),
    angleAt(idx(9), idx(10), idx(11)),
    angleAt(idx(10), idx(11), idx(12)),
    angleAt(idx(13), idx(14), idx(15)),
    angleAt(idx(14), idx(15), idx(16)),
    angleAt(idx(17), idx(18), idx(19)),
    angleAt(idx(18), idx(19), idx(20))
  ];
}

function buildHandFeatures(landmarks) {
  const norm = normalizeLandmarks(landmarks);
  const coords = flattenCoords(norm);
  const angles = computeAngles(norm);
  return coords.concat(angles);
}

function zeroHandFeatures() {
  return new Array(73).fill(0);
}

function getHandLabel(results, index) {
  if (!results.multiHandedness || !results.multiHandedness[index]) {
    return 'Unknown';
  }
  const info = results.multiHandedness[index];
  if (info.label) {
    return info.label;
  }
  if (info.classification && info.classification[0] && info.classification[0].label) {
    return info.classification[0].label;
  }
  return 'Unknown';
}

async function saveSample(features, label, dataset, onlyPrimaryHand) {
  pending = true;
  try {
    const res = await fetch('/save_sample', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ features, label, dataset, only_primary_hand: onlyPrimaryHand })
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
  maxNumHands: 2,
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
    let rightFeatures = null;
    let leftFeatures = null;
    const unknownFeatures = [];

    results.multiHandLandmarks.forEach((landmarks, idx) => {
      drawConnectors(ctx, landmarks, HAND_CONNECTIONS, { color: '#00FF00', lineWidth: 2 });
      drawLandmarks(ctx, landmarks, { color: '#FF0000', lineWidth: 1 });

      const features = buildHandFeatures(landmarks);
      const label = getHandLabel(results, idx);
      if (label === 'Right') {
        rightFeatures = features;
      } else if (label === 'Left') {
        leftFeatures = features;
      } else {
        unknownFeatures.push(features);
      }
    });

    if (!rightFeatures && unknownFeatures.length > 0) {
      rightFeatures = unknownFeatures.shift();
    }
    if (!leftFeatures && unknownFeatures.length > 0) {
      leftFeatures = unknownFeatures.shift();
    }

    if (recording && currentLabel && !pending && samplesCollected < targetSamples) {
      const primary = rightFeatures || zeroHandFeatures();
      const secondary = leftFeatures || zeroHandFeatures();
      const onlyPrimaryHand = rightFeatures && !leftFeatures ? 1 : 0;
      const features = primary.concat(secondary);
      saveSample(features, currentLabel, currentDataset, onlyPrimaryHand);
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
