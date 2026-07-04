import RecognizePanel from "./components/RecognizePanel.jsx";
import SignPanel from "./components/SignPanel.jsx";

export default function App() {
  return (
    <div className="min-h-screen px-5 py-10 max-w-6xl mx-auto">
      <header className="mb-10 text-center">
        <h1 className="font-display text-3xl font-700 tracking-[0.12em]">
          <span className="text-recognize">SIGN</span>
          <span className="text-mist mx-3">⇄</span>
          <span className="text-synth">TEXT</span>
        </h1>
        <p className="text-mist text-sm mt-2">
          Fingerspell at the camera to write · type to watch it signed back
        </p>
      </header>

      <main className="grid gap-6 lg:grid-cols-2 items-start">
        <RecognizePanel />
        <SignPanel />
      </main>

      <footer className="mt-10 text-center text-xs text-mist">
        ASL fingerspelling · keypoint model · Softweave Elevation
      </footer>
    </div>
  );
}