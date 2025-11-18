import type { PredictionResult } from "../../types";

interface ResultHighlightProps {
  result: PredictionResult;
}

const ResultHighlight = ({ result }: ResultHighlightProps) => {
  return (
    <div className="result-highlight">
      <div className="image-stack">
        {result.gradcamUrl ? (
          <img className="base" src={result.gradcamUrl} alt="analysis" />
        ) : (
          <div className="gradcam-placeholder">
            <div className="ring ring-1" />
            <div className="ring ring-2" />
            <div className="ring ring-3" />
          </div>
        )}
        <div className="overlay" />
      </div>
      <div className="cards">
        <div className="card glass">
          <p className="eyebrow">진단명</p>
          <h2>{result.diagnosis}</h2>
          <p>확률 {(result.probability * 100).toFixed(1)}%</p>
          <span className={`badge ${result.riskLevel}`}>{result.riskLevel.toUpperCase()}</span>
        </div>
        <div className="card">
          <h3>조치 안내</h3>
          <p>{result.recommendations}</p>
          <p className="referral">{result.referral}</p>
        </div>
      </div>
    </div>
  );
};

export default ResultHighlight;
