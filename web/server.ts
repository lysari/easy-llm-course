/// <reference types="bun-types" />
import { join } from "path";
import { readdir, readFile, writeFile, mkdir, unlink } from "fs/promises";

const ROOT = join(import.meta.dir, "..");
const LESSONS_DIR = join(ROOT, "lessons");
const PLAYGROUND_DIR = join(ROOT, ".tmp");
const PUBLIC_DIR = join(import.meta.dir, "public");
const PORT = Number(process.env.PORT) || 3030;

// ── Route handlers ──

async function getLessons() {
  const dirs = (await readdir(LESSONS_DIR)).sort();
  const results = await Promise.all(
    dirs.map(async (slug) => {
      try {
        const md = await readFile(join(LESSONS_DIR, slug, "lesson.md"), "utf-8");
        const match = md.match(/^#\s+(.+)$/m);
        return { slug, title: match ? match[1].trim() : slug };
      } catch {
        return null;
      }
    })
  );
  return results.filter(Boolean);
}

async function getLesson(slug: string) {
  if (!/^[\w-]+$/.test(slug)) throw new Error("Invalid slug");
  const dir = join(LESSONS_DIR, slug);
  const markdown = await readFile(join(dir, "lesson.md"), "utf-8");
  let code = "";
  try {
    code = await readFile(join(dir, "index.ts"), "utf-8");
  } catch {
    code = `// No code file for this lesson\nconsole.log("Hello from ${slug}!");`;
  }
  return { markdown, code };
}

// Stream code execution output as Server-Sent Events
async function streamRun(code: string): Promise<Response> {
  if (code.length > 50_000) {
    return Response.json({ error: "Code too large" }, { status: 400 });
  }

  await mkdir(PLAYGROUND_DIR, { recursive: true });
  const tmpFile = join(PLAYGROUND_DIR, `play_${Date.now()}.ts`);
  await writeFile(tmpFile, code);

  const proc = Bun.spawn(["bun", "run", tmpFile], {
    cwd: ROOT,
    stdout: "pipe",
    stderr: "pipe",
  });

  const enc = new TextEncoder();
  const sse = (event: string, data: string) =>
    enc.encode(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);

  const body = new ReadableStream({
    async start(ctrl) {
      const timer = setTimeout(() => proc.kill(), 10_000);

      // Drain a pipe and forward each chunk as an SSE event
      const drain = async (src: ReadableStream<Uint8Array>) => {
        const reader = src.getReader();
        const dec = new TextDecoder();
        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            ctrl.enqueue(sse("out", dec.decode(value)));
          }
        } catch { /* process killed or pipe closed */ }
      };

      await Promise.all([drain(proc.stdout!), drain(proc.stderr!)]);
      clearTimeout(timer);
      await proc.exited;
      ctrl.enqueue(sse("done", String(proc.exitCode ?? 0)));
      ctrl.close();
      await unlink(tmpFile).catch(() => {});
    },
  });

  return new Response(body, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      "Access-Control-Allow-Origin": "*",
      "X-Accel-Buffering": "no",
    },
  });
}

// ── Static file helper ──

const MIME: Record<string, string> = {
  ".html": "text/html; charset=utf-8",
  ".css":  "text/css",
  ".js":   "application/javascript",
  ".ts":   "application/javascript",
  ".json": "application/json",
  ".svg":  "image/svg+xml",
  ".ico":  "image/x-icon",
};

async function serveStatic(pathname: string): Promise<Response | null> {
  const rel = pathname === "/" ? "index.html" : pathname.replace(/^\//, "");
  // Block path traversal
  const abs = join(PUBLIC_DIR, rel);
  if (!abs.startsWith(PUBLIC_DIR)) return new Response("Forbidden", { status: 403 });

  const file = Bun.file(abs);
  if (!(await file.exists())) return null;

  const ext = rel.lastIndexOf(".") !== -1 ? rel.slice(rel.lastIndexOf(".")) : "";
  return new Response(file, {
    headers: { "Content-Type": MIME[ext] ?? "application/octet-stream" },
  });
}

// ── Server ──

const server = Bun.serve({
  port: PORT,

  async fetch(req) {
    const { pathname } = new URL(req.url);
    const json = (data: unknown, status = 200) =>
      Response.json(data, { status, headers: { "Access-Control-Allow-Origin": "*" } });

    try {
      // GET /api/lessons
      if (pathname === "/api/lessons" && req.method === "GET") {
        return json(await getLessons());
      }

      // GET /api/lessons/:slug
      const lessonMatch = pathname.match(/^\/api\/lessons\/([\w-]+)$/);
      if (lessonMatch && req.method === "GET") {
        try {
          return json(await getLesson(lessonMatch[1]!));
        } catch {
          return json({ error: "Not found" }, 404);
        }
      }

      // POST /api/run — streams SSE
      if (pathname === "/api/run" && req.method === "POST") {
        const { code } = (await req.json()) as { code: string };
        return streamRun(code ?? "");
      }

      // Static files
      const file = await serveStatic(pathname);
      if (file) return file;

      return json({ error: "Not found" }, 404);
    } catch (err) {
      return json({ error: String(err) }, 500);
    }
  },
});

console.log(`\n  LLM Learning  →  http://localhost:${server.port}\n`);
