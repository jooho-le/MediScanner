import { useState } from "react";
import { loginUser, registerUser } from "../api/routes";
import { setAuthToken } from "../api/client";
import type { AuthMode, UserCredentials } from "../types";

interface UseAuthResult {
  mode: AuthMode;
  setMode: (value: AuthMode) => void;
  isLoading: boolean;
  error: string | null;
  authenticate: (payload: UserCredentials) => Promise<void>;
  logout: () => void;
  token: string | null;
}

export const useAuth = (): UseAuthResult => {
  const initialToken = localStorage.getItem("mediscanner_token");
  if (initialToken) {
    setAuthToken(initialToken);
  }
  const [mode, setMode] = useState<AuthMode>("login");
  const [token, setToken] = useState<string | null>(initialToken);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setLoading] = useState(false);

  const authenticate = async (payload: UserCredentials) => {
    setLoading(true);
    setError(null);
    try {
      const fn = mode === "login" ? loginUser : registerUser;
      const response = await fn(payload);
      setAuthToken(response.access_token);
      setToken(response.access_token);
    } catch (err: unknown) {
      console.error(err);
      if (err && typeof err === "object" && "response" in err) {
        const resp = (err as any).response;
        setError(resp?.data?.detail ?? "인증에 실패했습니다. 다시 시도해 주세요.");
      } else {
        setError("인증에 실패했습니다. 다시 시도해 주세요.");
      }
    } finally {
      setLoading(false);
    }
  };

  const logout = () => {
    setToken(null);
    setAuthToken(null);
  };

  return { mode, setMode, isLoading, error, authenticate, logout, token };
};
