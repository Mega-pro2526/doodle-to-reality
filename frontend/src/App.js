import React, { useRef, useState } from 'react';
import CanvasDraw from 'react-canvas-draw';
import './App.css';

function App() {
  const canvasRef = useRef();
  const [style, setStyle] = useState("sketch");
  const [prediction, setPrediction] = useState("");
  const [story, setStory] = useState("");

  const clearCanvas = () => {
    canvasRef.current.clear();
    setPrediction("");
    setStory("");
  };

  const saveDrawing = async () => {
    try {
      const dataURL = canvasRef.current.canvas.drawing.toDataURL("image/png");

      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          image: dataURL.split(",")[1],
          style: style
        })
      });

      const result = await response.json();
      if (!response.ok) throw new Error(result.error);
      setPrediction(result.prediction);

      const storyResponse = await fetch("http://127.0.0.1:5000/story", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ label: result.prediction })
      });

      const storyData = await storyResponse.json();
      if (!storyResponse.ok) throw new Error(storyData.error);
      setStory(storyData.story);

    } catch (err) {
      setPrediction("");
      setStory("");
      alert("❌ Error: " + err.message);
    }
  };

  return (
    <div className="App" style={{ textAlign: 'center', fontFamily: 'Arial' }}>
      <h1>Doodle to Reality 🎨</h1>

      <CanvasDraw
        ref={canvasRef}
        brushRadius={2}
        canvasWidth={400}
        canvasHeight={400}
      />

      <div style={{ marginTop: "1rem" }}>
        <label><strong>Select Style:</strong> </label>
        <select value={style} onChange={(e) => setStyle(e.target.value)}>
          <option value="sketch">Sketch</option>
          <option value="painting">Painting</option>
          <option value="van-gogh">Van Gogh</option>
        </select>
      </div>

      <div style={{ marginTop: "1rem" }}>
        <button onClick={saveDrawing}>🧠 Predict</button>
        <button onClick={clearCanvas}>❌ Clear</button>
      </div>

      {prediction && (
        <div style={{ marginTop: "2rem", padding: "1rem", border: "1px solid #ccc", borderRadius: "8px", width: "80%", marginLeft: "auto", marginRight: "auto" }}>
          <h3>🧠 Prediction: <em>{prediction}</em></h3>
          <h4>🎨 Style: <em>{style}</em></h4>
          <p>📖 <strong>Story:</strong> {story}</p>
        </div>
      )}
    </div>
  );
}

export default App;
