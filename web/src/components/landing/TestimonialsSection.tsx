const testimonials = [
  {
    quote: "집에서 미리 위험도를 보고 피부과에 갈 수 있어서 시간 절약이 됐어요.",
    name: "김소연 님"
  },
  {
    quote: "히트맵으로 어디가 문제인지 명확해져서 의사 상담 때 큰 도움이 되었습니다.",
    name: "이준호 님"
  },
  {
    quote: "부모님께 보내드려서 편하게 체크하고 병원 추천까지 받았어요.",
    name: "최은지 님"
  }
];

const TestimonialsSection = () => (
  <section className="testimonials-section">
    <div className="section-header">
      <p className="eyebrow">USER VOICES</p>
      <h2>사용자 피드백</h2>
    </div>
    <div className="testimonial-grid">
      {testimonials.map((item) => (
        <blockquote key={item.name}>
          <p>“{item.quote}”</p>
          <cite>{item.name}</cite>
        </blockquote>
      ))}
    </div>
  </section>
);

export default TestimonialsSection;
