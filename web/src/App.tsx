import { useEffect, useState } from "react";
import { Link, NavLink, Route, Routes } from "react-router-dom";
import HomePage from "./pages/Home";
import UploadPage from "./pages/Upload";
import ResultPage from "./pages/Result";
import ClinicsPage from "./pages/Clinics";
import AboutPage from "./pages/About";
import { useAuth } from "./hooks/useAuth";
import AuthForm from "./components/AuthForm";

const navItems = [
  { path: "/", label: "서비스 소개" },
  { path: "/upload", label: "사진 업로드" },
  { path: "/result", label: "결과 미리보기" },
  { path: "/clinics", label: "추천 병원" },
  { path: "/about", label: "About & 약관" }
];

const App = () => {
  const { mode, setMode, isLoading: authLoading, error, authenticate, logout, token } = useAuth();
  const [authOpen, setAuthOpen] = useState(false);

  useEffect(() => {
    if (token) {
      setAuthOpen(false);
    }
  }, [token]);

  return (
    <>
      {!token && authOpen && (
        <div className="auth-drawer open">
          <div className="drawer-content">
            <button className="close" onClick={() => setAuthOpen(false)}>
              ✕
            </button>
            <AuthForm
              mode={mode}
              isLoading={authLoading}
              error={error}
              onSubmit={authenticate}
              toggleMode={() => setMode(mode === "login" ? "register" : "login")}
            />
          </div>
        </div>
      )}

      <div className="site-shell">
        <nav className="top-nav">
          <Link to="/" className="brand">
            MediScanner
          </Link>
          <div className="nav-links">
            {navItems.map((item) => (
              <NavLink key={item.path} to={item.path} className={({ isActive }) => (isActive ? "active" : "")}>
                {item.label}
              </NavLink>
            ))}
          </div>
          {token ? (
            <button className="cta secondary" onClick={logout}>
              로그아웃
            </button>
          ) : (
            <button className="cta" onClick={() => setAuthOpen(true)}>
              로그인
            </button>
          )}
        </nav>

        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/upload" element={<UploadPage />} />
          <Route path="/result" element={<ResultPage />} />
          <Route path="/clinics" element={<ClinicsPage />} />
          <Route path="/about" element={<AboutPage />} />
        </Routes>

        <footer className="footer">
          <p>© {new Date().getFullYear()} MediScanner. 의료 전문 진단 전 단계 지원용.</p>
          <div>
            <Link to="/about">개인정보 보호</Link>
            <span>·</span>
            <a href="mailto:hello@mediscanner.ai">문의하기</a>
          </div>
        </footer>
      </div>
    </>
  );
};

export default App;
