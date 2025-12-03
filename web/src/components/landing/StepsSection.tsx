const steps = [
  {
    title: "사진 업로드",
    detail: "스마트폰/카메라로 촬영한 병변 사진을 업로드합니다."
  },
  {
    title: "AI 분석 + 근거",
    detail: "몇 초 만에 위험도와 Grad-CAM 히트맵을 확인합니다."
  },
  {
    title: "맞춤 가이드",
    detail: "권장 조치·치료 가이드와 전문 클리닉을 추천받습니다."
  }
];

const StepsSection = () => (
  <section className="steps-section">
    <div className="section-header">
      <p className="eyebrow">HOW IT WORKS</p>
      <h2>3단계로 끝나는 사전 검사</h2>
    </div>
    <div className="step-grid">
      {steps.map((step, index) => (
        <div key={step.title} className="step-card">
          <div className="step-index">{index + 1}</div>
          <h3>{step.title}</h3>
          <p>{step.detail}</p>
        </div>
      ))}
    </div>
  </section>
);

export default StepsSection;
