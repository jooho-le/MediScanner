import { useNavigate } from "react-router-dom";
import UploadPhoto from "../components/UploadPhoto";
import type { PredictionResult } from "../types";

const mockResult: PredictionResult = {
  id: "sample",
  createdAt: new Date().toISOString(),
  diagnosis: "AKIEC (광선각화증)",
  probability: 0.78,
  riskLevel: "medium",
  gradcamUrl: "/sample-gradcam.png",
  recommendations: "일주일 내 전문의 진료를 권장합니다. 자외선 노출은 피하세요.",
  referral: "가까운 피부과 예약을 권장합니다."
};

const UploadPage = () => {
  const navigate = useNavigate();

  const handleAnalyze = async (file: File, memo?: string) => {
    console.log("업로드한 파일:", file.name, memo);
    // TODO: FastAPI /analyze 연동
    navigate("/result", { state: { result: mockResult } });
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
