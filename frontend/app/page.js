"use client";
import { useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

export default function Home() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  function onPick(e) {
    const f = e.target.files[0];
    if (!f) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
    setError("");
  }

  async function analyze() {
    if (!file) return;
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const body = new FormData();
      body.append("file", file);
      const res = await fetch(`${API}/predict`, { method: "POST", body });
      if (!res.ok) throw new Error((await res.json()).detail || "Request failed");
      setResult(await res.json());
    } catch (err) {
      setError(err.message || "Could not reach the server. Is the backend running?");
    } finally {
      setLoading(false);
    }
  }

  const pct = (c) => (c * 100).toFixed(2) + "%";
  const label = (s) => s.replaceAll("_", " ");

  return (
    <main className="page">
      <header>
        <h1>Tomato Leaf Disease Detection</h1>
        <p>Upload a tomato leaf photo to identify the disease and get treatment advice.</p>
      </header>

      <section className="panel">
        <label className="upload">
          <input type="file" accept="image/*" onChange={onPick} />
          {file ? "Choose a different image" : "Choose a leaf image"}
        </label>

        {preview && <img className="preview" src={preview} alt="Selected leaf" />}

        <button className="analyze" onClick={analyze} disabled={!file || loading}>
          {loading ? "Analyzing..." : "Analyze leaf"}
        </button>

        {error && <p className="error">{error}</p>}
      </section>

      {result && (
        <section className="panel result">
          <h2>{result.info?.name || label(result.prediction)}</h2>
          <p className="conf">Confidence: <b>{pct(result.confidence)}</b></p>

          {result.low_confidence && (
            <p className="warn">
              Low confidence. Try a clearer, well-lit photo of a single leaf.
            </p>
          )}

          <h3>Top 3 predictions</h3>
          {result.top3.map((t) => (
            <div className="bar-row" key={t.class}>
              <span>{label(t.class)}</span>
              <div className="bar"><div style={{ width: pct(t.confidence) }} /></div>
              <span>{pct(t.confidence)}</span>
            </div>
          ))}

          {result.info && (
            <div className="info">
              <div><h3>Cause</h3><p>{result.info.cause}</p></div>
              <div><h3>Symptoms</h3><p>{result.info.symptoms}</p></div>
              <div><h3>Treatment</h3><p>{result.info.treatment}</p></div>
              <div><h3>Prevention</h3><p>{result.info.prevention}</p></div>
            </div>
          )}

          <p className="note">
            This is general guidance. Confirm treatments with a local agricultural expert.
          </p>
        </section>
      )}
    </main>
  );
}