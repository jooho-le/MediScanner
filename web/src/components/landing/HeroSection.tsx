import { Link } from "react-router-dom";

const HeroSection = () => {
  return (
    <section className="hero-section">
      <div className="hero-content">
        <p className="eyebrow">AI SKIN CHECK-UP</p>
        <h1>
          피부과 가기 전,
          <br />
          집에서 미리 확인하세요.
        </h1>
        <p>
          MediScanner는 딥러닝 기반 피부 병변 분석과 Grad-CAM 근거 시각화를 제공해 의사 상담 전 준비가 한층 더
          수월하도록 도와드립니다.
        </p>
        <div className="hero-ctas">
          <Link to="/upload" className="cta primary">
            사진 업로드
          </Link>
          <Link to="/clinics" className="cta secondary">
            병원 둘러보기
          </Link>
        </div>
      </div>
      <div className="hero-visual">
        <div className="hex-grid">
          {[...Array(7)].map((_, idx) => (
            <div key={idx} className={`hex hex-${idx + 1}`} />
          ))}
        </div>
        <div className="hero-card">
          <p>평균 판독 시간</p>
          <strong>12초</strong>
          <span>AI + Grad-CAM</span>
        </div>
      </div>
    </section>
  );
};

export default HeroSection;
