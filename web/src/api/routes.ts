import api from "./client";
import type {
  AnalysisRequest,
  AuthResponse,
  PredictionResult,
  UserCredentials
} from "../types";

export const registerUser = async (payload: UserCredentials): Promise<AuthResponse> => {
  const { data } = await api.post<AuthResponse>("/auth/register", payload);
  return data;
};

export const loginUser = async (payload: UserCredentials): Promise<AuthResponse> => {
  const { data } = await api.post<AuthResponse>("/auth/login", payload);
  return data;
};

export const analyzeImage = async (payload: AnalysisRequest): Promise<PredictionResult> => {
  const formData = new FormData();
  formData.append("image", payload.file);
  if (payload.notes) {
    formData.append("notes", payload.notes);
  }
  const { data } = await api.post<PredictionResult>("/analysis/analyze", formData, {
    headers: { "Content-Type": "multipart/form-data" }
  });
  return data;
};

export const fetchHistory = async (): Promise<PredictionResult[]> => {
  const { data } = await api.get<PredictionResult[]>("/analysis/history");
  return data;
};
