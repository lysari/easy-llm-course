/** Escapes text for HTML. Use it for every word placed in markup. */
export const escapeHtml = (text: string) =>
  text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

/** Finds an element by id, and throws when it is missing. Pass the id without "#". */
export const byId = (id: string) => {
  const element = document.getElementById(id);
  if (!element) throw new Error(`Missing element: ${id}`);
  return element;
};

/** Sets an element's HTML only when it changed, so hover, focus and scroll survive a redraw. */
export const setHtml = (element: Element, html: string) => {
  if (element.getAttribute("data-html") === html) return;
  element.setAttribute("data-html", html);
  element.innerHTML = html;
};
