import { useState } from "react";
import type { AuthMode, UserCredentials } from "../types";

interface AuthFormProps {
  mode: AuthMode;
  isLoading: boolean;
  error?: string | null;
  onSubmit: (data: UserCredentials) => void;
  toggleMode: () => void;
}

const AuthForm = ({ mode, isLoading, error, onSubmit, toggleMode }: AuthFormProps) => {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit({ email, password, name: name || undefined });
  };

  return (
    <div className="card auth-form">
      <h2>{mode === "login" ? "로그인" : "회원가입"}</h2>
      <form onSubmit={handleSubmit}>
        {mode === "register" && (
          <label>
            <span>이름</span>
            <input value={name} onChange={(e) => setName(e.target.value)} required />
          </label>
        )}
        <label>
          <span>이메일</span>
          <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} required />
        </label>
        <label>
          <span>비밀번호</span>
          <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} required />
        </label>
        {error && <p className="error">{error}</p>}
        <button type="submit" disabled={isLoading}>
          {isLoading ? "처리 중..." : mode === "login" ? "로그인" : "회원가입"}
        </button>
      </form>
      <button className="link switch" type="button" onClick={toggleMode}>
        {mode === "login" ? "계정이 없으신가요? 회원가입" : "이미 계정이 있다면 로그인"}
      </button>
    </div>
  );
};

export default AuthForm;
