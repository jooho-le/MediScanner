import { useState } from "react";

interface UploadPhotoProps {
  onAnalyze: (file: File, memo?: string) => Promise<void> | void;
}

const UploadPhoto = ({ onAnalyze }: UploadPhotoProps) => {
  const [file, setFile] = useState<File | null>(null);
  const [notes, setNotes] = useState("");
  const [preview, setPreview] = useState<string | null>(null);
  const [isUploading, setUploading] = useState(false);

  const handleFile = (files: FileList | null) => {
    const selected = files?.[0];
    if (!selected) return;
    setFile(selected);
    setPreview(URL.createObjectURL(selected));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;
    setUploading(true);
    await onAnalyze(file, notes || undefined);
    setUploading(false);
  };

  return (
    <div className="upload-panel">
      <form className="card upload-card" onSubmit={handleSubmit}>
        <h3>피부 사진 업로드</h3>
        <p>깨끗한 조명, 10~15cm 거리로 촬영해 주세요.</p>
        <label className="file-drop">
          <input type="file" accept="image/*" onChange={(e) => handleFile(e.target.files)} />
          <span>{file ? file.name : "클릭하거나 파일을 드래그하세요"}</span>
        </label>
        <label>
          증상 메모 (선택)
          <textarea placeholder="가렵거나 통증 여부를 적어주세요" value={notes} onChange={(e) => setNotes(e.target.value)} />
        </label>
        <button type="submit" disabled={!file || isUploading}>
          {isUploading ? "분석 중..." : "AI 분석 시작"}
        </button>
      </form>
      <div className="preview-card">
        <h4>미리보기</h4>
        {preview ? <img src={preview} alt="preview" /> : <p>사진을 선택하면 미리보기로 표시됩니다.</p>}
        <ul>
          <li>빛 반사를 줄이고, 병변이 중앙에 오도록 촬영합니다.</li>
          <li>한 번에 한 개의 병변만 촬영해 주세요.</li>
          <li>분석은 의료 전문가 상담을 대체하지 않습니다.</li>
        </ul>
      </div>
    </div>
  );
};

export default UploadPhoto;
