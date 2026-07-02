// Fully client-side ASL fingerspelling reader.
// MediaPipe HandLandmarker (WASM) extracts 21 landmarks from the webcam in the
// browser; a tiny MLP (exported from the trained PyTorch model, BatchNorm
// folded in) classifies them into 29 classes. No server, no video ever leaves
// the machine.

import {
  HandLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/vision_bundle.mjs";

// --- commit/debounce tuning (mirrors server.py) -----------------------------
const WINDOW = 15;
const STABLE_VOTES = 12;
const CONF_THRESHOLD = 0.8;
const COOLDOWN = 1.2; // seconds

// MediaPipe hand connection topology (pairs of landmark indices)
const CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],         // thumb
  [0, 5], [5, 6], [6, 7], [7, 8],         // index
  [5, 9], [9, 10], [10, 11], [11, 12],    // middle
  [9, 13], [13, 14], [14, 15], [15, 16],  // ring
  [13, 17], [17, 18], [18, 19], [19, 20], // pinky
  [0, 17],                                // palm base
];

const canvas = document.getElementById("skeleton");
const ctx = canvas.getContext("2d");
const statusEl = document.getElementById("status");
const gestureEl = document.getElementById("gesture");
const confbar = document.getElementById("confbar");
const conftext = document.getElementById("conftext");
const sentenceEl = document.getElementById("sentence");
const video = document.getElementById("video");

document.getElementById("clear").onclick = () => (sentence = "");

let sentence = "";

// --- model (loaded from model.json) -----------------------------------------
let net = null; // { classes: [...], layers: [{W, b, act}, ...] }

async function loadModel() {
  const res = await fetch("model.json");
  if (!res.ok) throw new Error(`model.json ${res.status}`);
  const spec = await res.json();
  // pre-flatten weights into typed arrays for speed
  spec.layers = spec.layers.map((l) => ({
    out: l.W.length,
    in: l.W[0].length,
    W: Float32Array.from(l.W.flat()),
    b: Float32Array.from(l.b),
    act: l.act,
  }));
  net = spec;
}

function forward(x) {
  // x: Float32Array(63) -> logits Float32Array(nClasses)
  let a = x;
  for (const layer of net.layers) {
    const out = new Float32Array(layer.out);
    const { W, b, in: nin, out: nout } = layer;
    for (let i = 0; i < nout; i++) {
      let acc = b[i];
      const base = i * nin;
      for (let j = 0; j < nin; j++) acc += W[base + j] * a[j];
      out[i] = layer.act === "relu" ? Math.max(acc, 0) : acc;
    }
    a = out;
  }
  return a;
}

function softmaxArgmax(logits) {
  let max = -Infinity;
  for (const v of logits) if (v > max) max = v;
  let sum = 0;
  const exps = new Float32Array(logits.length);
  for (let i = 0; i < logits.length; i++) {
    exps[i] = Math.exp(logits[i] - max);
    sum += exps[i];
  }
  let bestIdx = 0,
    bestP = 0;
  for (let i = 0; i < exps.length; i++) {
    const p = exps[i] / sum;
    if (p > bestP) {
      bestP = p;
      bestIdx = i;
    }
  }
  return { idx: bestIdx, conf: bestP };
}

// --- normalization (must match utils.normalize_landmarks exactly) -----------
function normalize(landmarks) {
  // landmarks: array of {x, y, z}, length 21 -> Float32Array(63)
  const out = new Float32Array(63);
  const wx = landmarks[0].x,
    wy = landmarks[0].y,
    wz = landmarks[0].z;
  // translate so wrist is origin
  const rel = landmarks.map((l) => [l.x - wx, l.y - wy, l.z - wz]);
  // scale by wrist -> middle-MCP (landmark 9) distance
  let scale = Math.hypot(rel[9][0], rel[9][1], rel[9][2]);
  if (scale < 1e-6) scale = 1e-6;
  for (let i = 0; i < 21; i++) {
    out[i * 3] = rel[i][0] / scale;
    out[i * 3 + 1] = rel[i][1] / scale;
    out[i * 3 + 2] = rel[i][2] / scale;
  }
  return out;
}

function drawHand(pts) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!pts || pts.length === 0) return;
  const W = canvas.width,
    H = canvas.height;
  ctx.strokeStyle = "rgba(52,245,197,0.55)";
  ctx.lineWidth = 2.5;
  for (const [a, b] of CONNECTIONS) {
    ctx.beginPath();
    ctx.moveTo(pts[a].x * W, pts[a].y * H);
    ctx.lineTo(pts[b].x * W, pts[b].y * H);
    ctx.stroke();
  }
  for (let i = 0; i < pts.length; i++) {
    ctx.beginPath();
    ctx.arc(pts[i].x * W, pts[i].y * H, i % 4 === 0 ? 5 : 3.5, 0, Math.PI * 2);
    ctx.fillStyle = i % 4 === 0 ? "#34f5c5" : "#ffffff";
    ctx.fill();
  }
}

// --- main loop --------------------------------------------------------------
const votes = []; // ring buffer of {idx, conf}; idx = -1 means "no hand"
let lastCommit = { letter: "", t: 0 };

function pushVote(v) {
  votes.push(v);
  if (votes.length > WINDOW) votes.shift();
}

function tryCommit(classes) {
  if (votes.length < WINDOW) return;
  const counts = new Map();
  for (const { idx } of votes) counts.set(idx, (counts.get(idx) || 0) + 1);
  let topIdx = -1,
    topCount = 0;
  for (const [idx, c] of counts) {
    if (c > topCount) {
      topCount = c;
      topIdx = idx;
    }
  }
  if (topIdx < 0 || topCount < STABLE_VOTES) return;

  const relevant = votes.filter((v) => v.idx === topIdx);
  const meanConf =
    relevant.reduce((s, v) => s + v.conf, 0) / relevant.length;
  const letter = classes[topIdx];
  const now = performance.now() / 1000;

  if (
    meanConf >= CONF_THRESHOLD &&
    (letter !== lastCommit.letter || now - lastCommit.t > COOLDOWN)
  ) {
    if (letter === "space") sentence += " ";
    else if (letter === "del") sentence = sentence.slice(0, -1);
    else if (letter !== "nothing") sentence += letter;
    lastCommit = { letter, t: now };
    votes.length = 0; // clear window after a commit
  }
}

let landmarker = null;
let lastVideoTime = -1;

function loop() {
  if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const result = landmarker.detectForVideo(video, performance.now());

    let label = "nothing",
      conf = 0;
    if (result.landmarks && result.landmarks.length > 0) {
      const hand = result.landmarks[0];
      const vec = normalize(hand);
      const logits = forward(vec);
      const { idx, conf: c } = softmaxArgmax(logits);
      label = net.classes[idx];
      conf = c;
      pushVote({ idx, conf: c });
      drawHand(hand);
    } else {
      pushVote({ idx: -1, conf: 0 });
      drawHand(null);
    }
    tryCommit(net.classes);

    gestureEl.textContent = label === "nothing" ? "–" : label;
    const pct = Math.round(conf * 100);
    confbar.style.width = pct + "%";
    conftext.textContent = pct + "%";
    sentenceEl.textContent = sentence;
  }
  requestAnimationFrame(loop);
}

async function main() {
  try {
    statusEl.textContent = "loading model…";
    await loadModel();

    statusEl.textContent = "loading hand tracker…";
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm"
    );
    landmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numHands: 1,
      minHandDetectionConfidence: 0.5,
    });

    statusEl.textContent = "requesting camera…";
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480 },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();

    statusEl.textContent = "● live";
    statusEl.classList.add("live");
    loop();
  } catch (err) {
    console.error(err);
    statusEl.textContent = "error: " + (err.message || err);
    statusEl.classList.remove("live");
  }
}

main();
