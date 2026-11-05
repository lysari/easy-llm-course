import { byId } from "@/shared/utils/html";
import { showSidebarPage } from "./sidebar";

interface Page {
  open(): void;
  close(): void;
}

export function startRouter(lessons: Page, playground: Page) {
  function showPage() {
    const isPlayground = location.hash === "#playground";
    byId("lessons").hidden = isPlayground;
    byId("playground").hidden = !isPlayground;
    showSidebarPage(isPlayground);
    if (isPlayground) {
      lessons.close();
      playground.open();
    } else {
      playground.close();
      lessons.open();
    }
  }

  window.addEventListener("hashchange", showPage);
  showPage();
}
