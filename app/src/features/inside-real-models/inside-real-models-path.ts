import { matchPath } from "react-router";

export const INSIDE_REAL_MODELS_PATH = "/inside-real-models";

export const OLD_INSIDE_REAL_MODELS_PATH = "/take-apart";

export function movedPath(oldPathname: string) {
  const rest = matchPath(`${OLD_INSIDE_REAL_MODELS_PATH}/*`, oldPathname)?.params["*"];
  return rest ? `${INSIDE_REAL_MODELS_PATH}/${rest}` : INSIDE_REAL_MODELS_PATH;
}
