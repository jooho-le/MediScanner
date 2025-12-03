import { useNavigate } from "react-router-dom";
import UploadPhoto from "../components/UploadPhoto";
import { analyzeImage } from "../api/routes";
import type { PredictionResult } from "../types";

const UploadPage = () => {
  const navigate = useNavigate();

  const handleAnalyze = async (file: File, memo?: string) => {
    try {
      const data = await analyzeImage({ file, notes: memo });
      const mapped: PredictionResult = {
        id: String(data.id),
        createdAt: data.created_at ?? new Date().toISOString(),
        diagnosis: data.diagnosis,
        probability: data.probability,
        riskLevel: (data.risk_level ?? "uncertain") as PredictionResult["riskLevel"],
        gradcamUrl: data.gradcam_url,
        recommendations: data.recommendations,
        referral: data.referral,
        notes: data.notes,
      };
      navigate("/result", { state: { result: mapped } });
    } catch (err) {
      console.error(err);
      alert("분석 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.");
    }
  };

  return (
    <div className="page upload-page">
      <header className="page-header">
        <p className="eyebrow">UPLOAD</p>
        <h1>피부 병변 사진 업로드</h1>
        <p>업로드는 의료용 목적이 아닌 사전 안내 용도입니다.</p>
      </header>
      <UploadPhoto onAnalyze={handleAnalyze} />
    </div>
  );
};

export default UploadPage;
