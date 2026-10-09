/* HDFC AI Platform web UI. Every page reads real data from the FastAPI server (/v1/...).
   Login gives a token; the server checks the user's role on every request. */
import { createRoot } from "react-dom/client";
import { App } from "./App";
import "./index.css";

createRoot(document.getElementById("root")).render(<App />);
