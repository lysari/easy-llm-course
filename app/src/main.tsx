import "@/shared/styles/palette.css";
import "@/shared/styles/variables.css";
import "@/shared/styles/fonts.css";
import "@/shared/styles/global.css";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router";
import { Layout } from "@/core/layout/layout";

createRoot(document.getElementById("app")!).render(
  <StrictMode>
    <BrowserRouter useTransitions={false}>
      <Layout />
    </BrowserRouter>
  </StrictMode>,
);
