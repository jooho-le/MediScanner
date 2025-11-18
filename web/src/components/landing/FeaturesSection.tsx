const features = [
  {
    title: "자동 분석 & 위험도",
    description: "7가지 피부병 범주 확률과 안전 임계값 기반 불확실 안내",
    icon: "🔍"
  },
  {
    title: "Grad-CAM 근거",
    description: "AI가 주목한 피부 부위를 히트맵으로 시각화",
    icon: "🔥"
  },
  {
    title: "치료 가이드",
    description: "병변 유형에 따른 셀프케어·전문의 상담 안내 문구",
    icon: "🩺"
  },
  {
    title: "병원 추천",
    description: "전문의 정보, 진료 과목, 지도 연동까지 한눈에",
    icon: "📍"
  }
];

const FeaturesSection = () => {
  return (
    <section className="features-section">
      <div className="section-header">
        <p className="eyebrow">WHY MEDISCANNER</p>
        <h2>환자 친화적인 AI 보조 진단 경험</h2>
      </div>
      <div className="feature-grid">
        {features.map((item) => (
          <div key={item.title} className="feature-card">
            <span className="icon">{item.icon}</span>
            <div>
              <h3>{item.title}</h3>
              <p>{item.description}</p>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
};

export default FeaturesSection;
