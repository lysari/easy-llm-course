import "./layout.css";
import { createLessonsPage, lessonsPageHtml } from "@/features/lessons/lessons-page";
import { createPlayground } from "@/features/playground/playground-page";
import { byId } from "@/shared/utils/html";
import { startRouter } from "./router";
import { sidebarHtml } from "./sidebar";

export function mountLayout() {
  byId("app").innerHTML = `${sidebarHtml()}
    ${lessonsPageHtml()}
    <main class="page page--playground" id="playground" hidden></main>`;
  const lessons = createLessonsPage();
  const playground = createPlayground(byId("playground"));
  startRouter(lessons, playground);
}
