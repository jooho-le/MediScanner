import axios from "axios";

const api = axios.create({
  baseURL: "/api",
  withCredentials: false
});

export const setAuthToken = (token: string | null) => {
  if (token) {
    api.defaults.headers.common.Authorization = `Bearer ${token}`;
    localStorage.setItem("mediscanner_token", token);
  } else {
    delete api.defaults.headers.common.Authorization;
    localStorage.removeItem("mediscanner_token");
  }
};

export default api;
