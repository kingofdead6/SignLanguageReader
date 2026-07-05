import { useEffect, useRef, useState } from "react";
import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";
import { API_BASE, WS_URLS } from "../lib/api";
import { drawSkeleton } from "../lib/hand";

const SEND_FPS = 20;
const ACCENT = "#2de1c2";

export default function RecognizePanel() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const stateRef = useRef({ alive: true, wsIdx: 0, lastSend: 0, raf: 0 });

  const [status, setStatus] = useState("loading hand tracker…");
  const [live, setLive] = useState(false);
  const [gesture, setGesture] = useState("–");
  const [confidence, setConfidence] = useState(0);
  const [sentence, setSentence] = useState("");
  const [flash, setFlash] = useState(0);

  useEffect(() => {
    const S = stateRef.current;
    S.alive = true;
    let landmarker = null;
    let stream = null;

    function connect() {
      if (!S.alive) return;
      const url = WS_URLS[S.wsIdx % WS_URLS.length];
      const ws = new WebSocket(url);
      wsRef.current = ws;
      let opened = false;
      ws.onopen = () => {
        opened = true;
        setLive(true);
        setStatus("socket open — waiting for server hello…");
      };
      ws.onclose = () => {
        if (!S.alive) return;
        if (!opened) S.wsIdx++; // never connected -> try the other candidate
        setLive(false);
        setStatus("API disconnected — retrying…");
        setTimeout(connect, 1500);
      };
      ws.onmessage = (e) => {
        let d;
        try { d = JSON.parse(e.data); } catch { return; }
        if (!d) return;
        if (d.type === "hello") {
          // Backend's connection sign, mirrored to the UI
          setStatus(`live · session #${d.session} · ${d.classes} classes`);
          return;
        }
        if (d.type !== "frame") return;
        setGesture(d.gesture === "nothing" ? "–" : d.gesture);
        setConfidence(d.confidence);
        setSentence(d.sentence);
        if (d.committed) setFlash((f) => f + 1);
      };
    }

    // Free-tier hosts sleep when idle; an HTTP hit wakes them. Ping
    // /health first so the WS isn't hammering a cold box.
    async function wakeThenConnect() {
      setStatus("waking API… (cold start can take up to a minute)");
      try {
        const ctrl = new AbortController();
        const t = setTimeout(() => ctrl.abort(), 70000);
        const r = await fetch(`${API_BASE}/health`, { signal: ctrl.signal });
        clearTimeout(t);
        const d = await r.json();
        if (!S.alive) return;
        setStatus(`API awake (${d.classes} classes) — opening socket…`);
      } catch {
        if (!S.alive) return;
        setStatus("health check failed — trying WebSocket anyway…");
      }
      connect();
    }

    async function init() {
      const fileset = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
      );
      landmarker = await HandLandmarker.createFromOptions(fileset, {
        baseOptions: {
          modelAssetPath:
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
          delegate: "GPU",
        },
        runningMode: "VIDEO",
        numHands: 1,
      });
      if (!S.alive) return;

      setStatus("requesting camera…");
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
        audio: false,
      });
      const video = videoRef.current;
      if (!S.alive || !video) return;
      video.srcObject = stream;
      await new Promise((r) => (video.onloadedmetadata = r));
      const canvas = canvasRef.current;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      wakeThenConnect();
      loop();
    }

    function loop() {
      if (!S.alive) return;
      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (landmarker && video && video.readyState >= 2) {
        const result = landmarker.detectForVideo(video, performance.now());
        const hand = result.landmarks && result.landmarks[0];

        const ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        if (hand) {
          drawSkeleton(
            ctx,
            hand.map((p) => [p.x * canvas.width, p.y * canvas.height]),
            ACCENT
          );
        }

        const ws = wsRef.current;
        const now = performance.now();
        if (ws && ws.readyState === 1 && now - S.lastSend > 1000 / SEND_FPS) {
          S.lastSend = now;
          ws.send(
            JSON.stringify(
              hand
                ? {
                    type: "landmarks",
                    landmarks: hand.map((p) => [
                      +p.x.toFixed(4), +p.y.toFixed(4), +p.z.toFixed(4),
                    ]),
                  }
                : { type: "empty" }
            )
          );
        }
      }
      S.raf = requestAnimationFrame(loop);
    }

    init().catch((err) => setStatus("error: " + err.message));

    return () => {
      S.alive = false;
      cancelAnimationFrame(S.raf);
      if (wsRef.current) wsRef.current.close();
      if (stream) stream.getTracks().forEach((t) => t.stop());
      if (landmarker) landmarker.close();
    };
  }, []);

  const clear = () => {
    const ws = wsRef.current;
    if (ws && ws.readyState === 1) ws.send(JSON.stringify({ type: "clear" }));
  };

  return (
    <section className="bg-panel border border-edge rounded-2xl p-5 flex flex-col gap-4
                        shadow-[0_16px_40px_-20px_rgba(0,0,0,0.8)]">
      <header className="flex items-center justify-between gap-3">
        <h2 className="font-display font-700 tracking-[0.14em] text-recognize flex items-center gap-2.5">
          <span className="w-1.5 h-5 rounded-full bg-recognize/80" aria-hidden="true" />
          CAMERA → TEXT
        </h2>
        <span className="flex items-center gap-2 text-xs text-mist bg-well border border-edge
                         rounded-full pl-2.5 pr-3 py-1 max-w-[55%]">
          <span className={`shrink-0 w-2 h-2 rounded-full ${
            live ? "bg-recognize pulse-dot" : "bg-mist/50"}`} aria-hidden="true" />
          <span className="truncate">{status}</span>
        </span>
      </header>

      <div className="relative rounded-xl overflow-hidden bg-well border border-edge aspect-4/3">
        <video ref={videoRef} autoPlay playsInline muted
               className="w-full h-full object-cover block -scale-x-100" />
        <canvas ref={canvasRef}
                className="absolute inset-0 w-full h-full pointer-events-none -scale-x-100" />
      </div>

      <div className="flex items-center gap-5">
        <div key={flash}
             className={`font-display text-5xl font-700 min-w-20 py-1 text-center text-recognize
                         bg-well border border-edge rounded-xl ${flash ? "commit-flash" : ""}`}>
          {gesture}
        </div>
        <div className="flex-1">
          <div className="flex items-baseline justify-between text-xs text-mist mb-1.5">
            <span>confidence</span>
            <span className="tabular-nums text-ink/80">{Math.round(confidence * 100)}%</span>
          </div>
          <div className="h-1.5 rounded-full bg-edge overflow-hidden"
               role="meter" aria-label="prediction confidence"
               aria-valuemin={0} aria-valuemax={100}
               aria-valuenow={Math.round(confidence * 100)}>
            <div className="h-full rounded-full bg-recognize transition-[width] duration-100"
                 style={{ width: `${Math.round(confidence * 100)}%` }} />
          </div>
        </div>
      </div>

      <div className="bg-well border border-edge rounded-xl px-4 py-3 min-h-13">
        {sentence
          ? <span className="text-xl tracking-widest break-all">{sentence}</span>
          : <span className="text-sm text-mist/60 tracking-wide">
              hold a letter steady to write…
            </span>}
        <span className="inline-block w-0.5 h-5 ml-0.5 align-text-bottom bg-recognize cursor-blink" />
      </div>

      <button onClick={clear}
              className="self-start text-sm text-mist border border-edge rounded-lg px-4 py-2
                         cursor-pointer hover:text-recognize hover:border-recognize/60
                         hover:bg-recognize/5 active:scale-[0.98] transition-all">
        Clear sentence
      </button>
    </section>
  );
}