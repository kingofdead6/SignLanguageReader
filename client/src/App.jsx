import RecognizePanel from "./components/RecognizePanel.jsx";
import SignPanel from "./components/SignPanel.jsx";

export default function App() {
  return (
    <div className="min-h-screen px-5 py-10 max-w-6xl mx-auto">
      <header className="mb-10 text-center">
        <p className="text-[11px] uppercase tracking-[0.3em] text-mist mb-3">
          ASL fingerspelling · real-time keypoint model
        </p>
        <h1 className="font-display text-4xl sm:text-5xl font-700 tracking-[0.1em]">
          <span className="text-recognize drop-shadow-[0_0_18px_rgba(45,225,194,0.35)]">
            SIGN
          </span>
          <span className="text-mist/60 mx-4 font-normal">⇄</span>
          <span className="text-synth drop-shadow-[0_0_18px_rgba(245,184,76,0.3)]">
            TEXT
          </span>
        </h1>
        <p className="text-mist text-sm mt-4 max-w-md mx-auto leading-relaxed">
          Fingerspell at the camera to write · type to watch it signed back
        </p>
      </header>

      <main className="grid gap-6 lg:grid-cols-2 items-start">
        <RecognizePanel />
        <SignPanel />
      </main>

      <footer className="mt-12 text-center text-xs text-mist/70">
        21 hand landmarks · 29 classes · inference on the server, tracking in
        your browser — no video ever leaves this page
      </footer>
    </div>
  );
}
