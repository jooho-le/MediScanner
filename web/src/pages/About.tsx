const AboutPage = () => (
  <div className="page about-page">
    <header className="page-header">
      <p className="eyebrow">ABOUT & POLICY</p>
      <h1>MediScanner 서비스 소개</h1>
      <p>AI 분석은 의료 전문가의 진단을 대체할 수 없으며, 사전 안내 목적입니다.</p>
    </header>
    <section className="card">
      <h3>서비스 목적</h3>
      <ul>
        <li>피부과 상담 전 병변 우선순위 파악</li>
        <li>의사-환자 커뮤니케이션을 돕는 Grad-CAM 근거 제공</li>
        <li>치료 가이드 및 협력 병원 추천</li>
      </ul>
    </section>
    <section className="card">
      <h3>개인정보 보호</h3>
      <p>이미지는 분석 직후 자동으로 암호화 저장되며 24시간 내 삭제됩니다. 계정에 로그인할 경우 결과 요약 및 피드백 메모만 저장됩니다.</p>
    </section>
    <section className="card">
      <h3>이용약관 핵심</h3>
      <ol>
        <li>본 서비스는 의료행위가 아니며, 진단 및 치료 결정은 의료 전문가에게 문의하십시오.</li>
        <li>AI 예측 결과에 대한 책임은 사용자에게 있으며, 응급 상황은 즉시 119 또는 병원을 이용해 주세요.</li>
        <li>서비스 이용 시 생성된 데이터는 모델 고도화에 익명화된 형태로 활용될 수 있습니다.</li>
      </ol>
    </section>
  </div>
);

export default AboutPage;
