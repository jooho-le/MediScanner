export type AuthMode = "login" | "register";

export interface AuthResponse {
  access_token: string;
  token_type: string;
}

export interface UserCredentials {
  email: string;
  password: string;
  name?: string;
}

export interface AnalysisRequest {
  file: File;
  notes?: string;
}

export interface PredictionResult {
  id: string;
  createdAt: string;
  diagnosis: string;
  probability: number;
  riskLevel: "low" | "medium" | "high" | "uncertain";
  gradcamUrl?: string;
  recommendations: string;
  referral: string;
  notes?: string;
}
