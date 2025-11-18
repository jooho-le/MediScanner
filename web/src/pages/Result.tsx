import { useLocation, useNavigate } from "react-router-dom";
import ResultHighlight from "../components/result/ResultHighlight";
import type { PredictionResult } from "../types";

const fallbackResult: PredictionResult = {
  id: "fallback",
  createdAt: new Date().toISOString(),
  diagnosis: "결과 없음",
  probability: 0.0,
  riskLevel: "uncertain",
  gradcamUrl: "/sample-gradcam.png",
  recommendations: "먼저 사진을 업로드해 분석을 진행해 주세요.",
  referral: "필요 시 가까운 의료기관을 방문하세요."
};

const ResultPage = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const result = (location.state as { result?: PredictionResult })?.result ?? fallbackResult;

  return (
    <div className="page result-page">
      <header className="page-header">
        <p className="eyebrow">RESULT</p>
        <h1>분석 결과</h1>
        <p>AI 분석과 Grad-CAM 근거는 의사의 판독을 돕는 참고 정보입니다.</p>
        <button className="cta primary" onClick={() => navigate("/upload")}>
          다른 사진 분석하기
        </button>
      </header>
      <ResultHighlight result={result} />
    </div>
  );
};

export default ResultPage;
