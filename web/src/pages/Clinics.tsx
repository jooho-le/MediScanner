import { useMemo, useState } from "react";
import ClinicCard from "../components/clinic/ClinicCard";

const clinics = [
  { name: "서울테라피 피부과", specialty: "피부종양 · 레이저 클리닉", address: "서울 강남구 테헤란로 123", distance: "2.1km" },
  { name: "바른빛 피부센터", specialty: "색소질환 · 광선치료", address: "서울 서초구 서초대로 45", distance: "4.0km" },
  { name: "해안 피부클리닉", specialty: "모반 · 종양 수술", address: "부산 해운대구 센텀중앙로 12", distance: "부산" }
];

const regions = ["전체", "서울", "부산"];

const ClinicsPage = () => {
  const [region, setRegion] = useState("전체");
  const filtered = useMemo(() => {
    if (region === "전체") return clinics;
    return clinics.filter((c) => c.address.includes(region));
  }, [region]);

  return (
    <div className="page clinics-page">
      <header className="page-header">
        <p className="eyebrow">CLINIC GUIDE</p>
        <h1>전문의 협력 병원</h1>
        <p>지역/전문분야 필터로 가까운 병원을 찾아보세요.</p>
        <div className="filters">
          <label>
            지역
            <select value={region} onChange={(e) => setRegion(e.target.value)}>
              {regions.map((r) => (
                <option key={r}>{r}</option>
              ))}
            </select>
          </label>
        </div>
      </header>
      <div className="clinic-grid">
        {filtered.map((clinic) => (
          <ClinicCard key={clinic.name} clinic={clinic} />
        ))}
      </div>
    </div>
  );
};

export default ClinicsPage;
