import { useEffect, useRef, useState } from "react";
import { API_BASE } from "../lib/api";
import { CONNECTIONS, easeInOutCubic, lerpPose } from "../lib/hand";

const ACCENT = "#f5b84c";
const HAND_SCALE = 55;
const ANCHOR = { x: 250, y: 205 }; // end of the raised forearm
const CANVAS_W = 340;
const CANVAS_H = 430;

function drawBody(ctx) {
  const c = "rgba(233,237,245,0.8)";
  const line = (x1, y1, x2, y2) => {
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.lineWidth = 3.5;
    ctx.strokeStyle = c;
    ctx.lineCap = "round";
    ctx.stroke();
  };
  ctx.beginPath(); // head
  ctx.arc(150, 90, 26, 0, Math.PI * 2);
  ctx.lineWidth = 3.5;
  ctx.strokeStyle = c;
  ctx.stroke();
  ctx.beginPath(); // smile
  ctx.arc(150, 94, 11, 0.25 * Math.PI, 0.75 * Math.PI);
  ctx.lineWidth = 2;
  ctx.stroke();
  line(150, 116, 150, 260);          // torso
  line(150, 260, 118, 360);          // legs
  line(150, 260, 182, 360);
  line(150, 150, 105, 230);          // relaxed left arm
  line(150, 150, 228, 185);          // raised right arm
  line(228, 185, ANCHOR.x, ANCHOR.y + 18);
}

function drawFigure(ctx, pose) {
  ctx.clearRect(0, 0, CANVAS_W, CANVAS_H);
  drawBody(ctx);
  if (!pose) return;
  const P = pose.map((p) => [
    ANCHOR.x + p[0] * HAND_SCALE,
    ANCHOR.y + p[1] * HAND_SCALE,
  ]);
  ctx.strokeStyle = ACCENT;
  ctx.lineWidth = 3;
  ctx.lineCap = "round";
  for (const [a, b] of CONNECTIONS) {
    ctx.beginPath();
    ctx.moveTo(P[a][0], P[a][1]);
    ctx.lineTo(P[b][0], P[b][1]);
    ctx.stroke();
  }
  P.forEach((p, i) => {
    ctx.beginPath();
    ctx.arc(p[0], p[1], i % 4 === 0 ? 4.5 : 3, 0, Math.PI * 2);
    ctx.fillStyle = i % 4 === 0 ? ACCENT : "#ffffff";
    ctx.fill();
  });
}

export default function SignPanel() {
  const canvasRef = useRef(null);
  const animRef = useRef({ raf: 0, playing: false, rest: null });

  const [text, setText] = useState("");
  const [status, setStatus] = useState("loading poses…");
  const [letters, setLetters] = useState([]); // [{ch, active}]
  const [ready, setReady] = useState(false);

  // Bootstrap: fetch a rest pose so the idle figure draws immediately
  useEffect(() => {
    const A = animRef.current;
    let cancelled = false;
    (async () => {
      try {
        const r = await fetch(`${API_BASE}/poses`);
        if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
        const poses = await r.json();
        if (cancelled) return;
        A.rest = poses["B"] || Object.values(poses)[0];
        setStatus(`${Object.keys(poses).length} letter poses loaded`);
        setReady(true);
        drawFigure(canvasRef.current.getContext("2d"), A.rest);
      } catch (err) {
        if (!cancelled) setStatus("poses unavailable: " + err.message);
      }
    })();
    return () => {
      cancelled = true;
      cancelAnimationFrame(A.raf);
      A.playing = false;
    };
  }, []);

  async function play() {
    const A = animRef.current;
    if (!text.trim() || !A.rest) return;
    setStatus("requesting animation…");

    let plan;
    try {
      const r = await fetch(`${API_BASE}/sign`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
      plan = await r.json();
    } catch (err) {
      setStatus("sign request failed: " + err.message);
      return;
    }

    // The server owns the plan: poses, order, timing.
    A.rest = plan.rest_pose;
    const seq = plan.sequence;
    const chain = [...seq, { ch: null, pose: plan.rest_pose }];
    const ctx = canvasRef.current.getContext("2d");

    cancelAnimationFrame(A.raf);
    A.playing = true;
    setLetters(seq.map((s) => ({ ch: s.ch === " " ? "·" : s.ch, active: false })));

    let idx = 0;
    let phase = "transition";
    let phaseStart = performance.now();
    let from = plan.rest_pose;

    const mark = (i) =>
      setLetters(seq.map((s, j) => ({
        ch: s.ch === " " ? "·" : s.ch,
        active: j === i,
      })));

    mark(0);
    setStatus(seq[0].ch === " " ? "signing (space)" : `signing ${seq[0].ch}`);

    const tick = (now) => {
      if (!A.playing) return;
      const target = chain[idx].pose;
      const elapsed = now - phaseStart;

      if (phase === "transition") {
        const t = Math.min(elapsed / plan.transition_ms, 1);
        drawFigure(ctx, lerpPose(from, target, easeInOutCubic(t)));
        if (t >= 1) { phase = "hold"; phaseStart = now; }
      } else {
        drawFigure(ctx, target);
        const hold = chain[idx].ch === null ? 0 : plan.hold_ms;
        if (elapsed >= hold) {
          from = target;
          idx++;
          if (idx >= chain.length) {
            A.playing = false;
            setLetters([]);
            setStatus("done");
            return;
          }
          phase = "transition";
          phaseStart = now;
          if (chain[idx].ch !== null) {
            mark(idx);
            setStatus(chain[idx].ch === " " ? "signing (space)" : `signing ${chain[idx].ch}`);
          } else {
            setLetters((ls) => ls.map((l) => ({ ...l, active: false })));
          }
        }
      }
      A.raf = requestAnimationFrame(tick);
    };
    A.raf = requestAnimationFrame(tick);
  }

  function stop() {
    const A = animRef.current;
    A.playing = false;
    cancelAnimationFrame(A.raf);
    setLetters([]);
    setStatus("stopped");
    if (A.rest) drawFigure(canvasRef.current.getContext("2d"), A.rest);
  }

  return (
    <section className="bg-panel border border-edge rounded-2xl p-5 flex flex-col gap-4">
      <header className="flex items-baseline justify-between">
        <h2 className="font-display font-700 tracking-wide text-synth">
          TEXT → SIGN
        </h2>
        <span className="text-xs text-mist">{status}</span>
      </header>

      <div className="flex gap-2 flex-wrap">
        <input
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && play()}
          maxLength={60}
          placeholder="Type something… e.g. HELLO"
          className="flex-1 min-w-44 bg-well border border-edge rounded-lg px-4 py-2.5
                     tracking-widest outline-none focus:border-synth transition-colors"
        />
        <button
          onClick={play}
          disabled={!ready}
          className="text-sm border border-edge rounded-lg px-4 py-2 text-mist
                     hover:text-synth hover:border-synth transition-colors
                     disabled:opacity-40 disabled:hover:text-mist disabled:hover:border-edge"
        >
          ▶ Sign it
        </button>
        <button
          onClick={stop}
          className="text-sm border border-edge rounded-lg px-4 py-2 text-mist
                     hover:text-synth hover:border-synth transition-colors"
        >
          ■ Stop
        </button>
      </div>

      <div className="min-h-7 text-lg tracking-[0.35em]">
        {letters.map((l, i) => (
          <span
            key={i}
            className={l.active ? "text-synth font-600" : "text-mist"}
          >
            {l.ch}
          </span>
        ))}
      </div>

      <canvas
        ref={canvasRef}
        width={CANVAS_W}
        height={CANVAS_H}
        className="bg-well rounded-xl mx-auto max-w-full"
      />
    </section>
  );
}