/* global marked, hljs, monaco, require */

let editor = null;
let originalCode = '';
let currentSlug = null;
let lessonCount = 0;

// ── Themes ──
const THEMES = {
  'tokyo-night':    { label: 'Tokyo Night',    monaco: 'vs-dark', hljs: 'atom-one-dark' },
  'one-dark':       { label: 'One Dark',        monaco: 'vs-dark', hljs: 'atom-one-dark' },
  'dracula':        { label: 'Dracula',          monaco: 'vs-dark', hljs: 'base16/dracula' },
  'solarized-dark': { label: 'Solarized Dark',  monaco: 'vs-dark', hljs: 'base16/solarized-dark' },
  'github-light':   { label: 'GitHub Light',    monaco: 'vs',      hljs: 'github' },
  'solarized-light':{ label: 'Solarized Light', monaco: 'vs',      hljs: 'base16/solarized-light' },
  'monokai':        { label: 'Monokai',          monaco: 'vs-dark', hljs: 'monokai' },
};
const HLJS_BASE = 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/';

function applyTheme(id) {
  const theme = THEMES[id];
  if (!theme) return;

  document.body.className = id === 'tokyo-night' ? '' : `theme-${id}`;
  document.getElementById('hljs-theme').href = `${HLJS_BASE}${theme.hljs}.min.css`;
  if (editor) monaco.editor.setTheme(theme.monaco);

  document.querySelectorAll('.swatch').forEach(s =>
    s.classList.toggle('active', s.dataset.theme === id)
  );
  localStorage.setItem('llm-theme', id);
}

function setupThemePicker() {
  document.querySelectorAll('.swatch').forEach(btn => {
    btn.addEventListener('click', () => applyTheme(btn.dataset.theme));
  });
  // Load saved theme
  applyTheme(localStorage.getItem('llm-theme') || 'tokyo-night');
}

// ── Markdown renderer with hljs ──
const renderer = new marked.Renderer();
renderer.code = (code, lang) => {
  const language = lang && hljs.getLanguage(lang) ? lang : 'plaintext';
  const highlighted = hljs.highlight(code, { language }).value;
  return `<pre><code class="hljs language-${language}">${highlighted}</code></pre>`;
};
renderer.link = (href, title, text) => {
  // Make relative links non-navigating (they point to .ts / .md files)
  if (!href || href.startsWith('http')) {
    return `<a href="${href}" target="_blank" rel="noopener"${title ? ` title="${title}"` : ''}>${text}</a>`;
  }
  return `<span class="md-link" title="${href}">${text}</span>`;
};

marked.setOptions({ renderer, gfm: true, breaks: false });

// ── Initialization ──
window.addEventListener('DOMContentLoaded', async () => {
  setupThemePicker();
  initMonaco();
  await loadLessonList();
  setupTabs();
  setupKeyboard();

  document.getElementById('run-btn').addEventListener('click', runCode);
  document.getElementById('reset-btn').addEventListener('click', resetCode);
  document.getElementById('clear-btn').addEventListener('click', clearConsole);
  setupResizeHandle();

  const hash = location.hash.slice(1);
  if (hash) loadLesson(hash);
});

// ── Monaco Editor ──
function initMonaco() {
  require.config({ paths: { vs: 'https://cdn.jsdelivr.net/npm/monaco-editor@0.45.0/min/vs' } });
  require(['vs/editor/editor.main'], () => {
    const savedTheme = THEMES[localStorage.getItem('llm-theme') || 'tokyo-night'];
    editor = monaco.editor.create(document.getElementById('editor-container'), {
      value: '// Select a lesson and open the Playground tab.',
      language: 'typescript',
      theme: savedTheme?.monaco || 'vs-dark',
      fontSize: 13,
      fontFamily: "'JetBrains Mono', 'Cascadia Code', 'Fira Code', monospace",
      lineHeight: 22,
      minimap: { enabled: false },
      scrollBeyondLastLine: false,
      automaticLayout: true,
      wordWrap: 'on',
      padding: { top: 14, bottom: 14 },
      renderLineHighlight: 'gutter',
      bracketPairColorization: { enabled: true },
    });

    // Ctrl/Cmd+Enter → run
    editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.Enter, runCode);
  });
}

// ── Lesson list ──
async function loadLessonList() {
  const lessons = await apiFetch('/api/lessons');
  if (!lessons) return;

  lessonCount = lessons.length;
  const list = document.getElementById('lesson-list');
  list.innerHTML = '';

  lessons.forEach(({ slug, title }) => {
    const num = slug.split('-')[0];
    const displayTitle = cleanTitle(title);

    const item = document.createElement('div');
    item.className = 'lesson-item';
    item.dataset.slug = slug;
    item.innerHTML = `
      <span class="lesson-num">${num}</span>
      <span class="lesson-title">${displayTitle}</span>
    `;
    item.addEventListener('click', () => loadLesson(slug));
    list.appendChild(item);
  });

  updateProgress(0);
}

// ── Load a lesson ──
async function loadLesson(slug) {
  if (slug === currentSlug) return;
  currentSlug = slug;
  location.hash = slug;

  // Highlight sidebar
  document.querySelectorAll('.lesson-item').forEach(el => {
    el.classList.toggle('active', el.dataset.slug === slug);
  });

  // Scroll active item into view
  const activeEl = document.querySelector('.lesson-item.active');
  if (activeEl) activeEl.scrollIntoView({ block: 'nearest', behavior: 'smooth' });

  const data = await apiFetch(`/api/lessons/${slug}`);
  if (!data) return;

  // Breadcrumb
  const title = cleanTitle(extractTitle(data.markdown));
  document.getElementById('breadcrumb').textContent = title;
  document.title = `${title} — LLM Learning`;

  // Render markdown
  const markdownBody = document.getElementById('markdown-body');
  markdownBody.innerHTML = marked.parse(data.markdown);

  // Store & set editor code
  originalCode = data.code;
  if (editor) editor.setValue(data.code);

  clearConsole();
  updateProgress(parseInt(slug.split('-')[0], 10) + 1);
}

// ── Tab switching ──
function setupTabs() {
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const tab = btn.dataset.tab;
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById(`${tab}-tab`).classList.add('active');
      if (tab === 'code' && editor) editor.layout();
    });
  });
}

// ── Keyboard ──
function setupKeyboard() {
  document.addEventListener('keydown', e => {
    // Ctrl/Cmd+Enter anywhere → switch to Code tab and run
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      const codeTab = document.getElementById('code-tab');
      if (!codeTab.classList.contains('active')) {
        document.querySelector('[data-tab="code"]').click();
      }
      runCode();
    }
  });
}

// ── Run code (SSE streaming) ──
async function runCode() {
  if (!editor || !currentSlug) return;

  const code = editor.getValue();
  const runBtn = document.getElementById('run-btn');
  const runStatus = document.getElementById('run-status');
  const consoleBody = document.getElementById('console-body');

  runBtn.disabled = true;
  runStatus.textContent = 'Running…';
  consoleBody.innerHTML = '';

  const outputEl = document.createElement('pre');
  outputEl.className = 'console-output';
  consoleBody.appendChild(outputEl);

  const start = Date.now();

  try {
    const res = await fetch('/api/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ code }),
    });

    if (!res.body) throw new Error('No response body');

    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buf += dec.decode(value, { stream: true });

      // Process complete SSE blocks (each ends with \n\n)
      let boundary;
      while ((boundary = buf.indexOf('\n\n')) !== -1) {
        const block = buf.slice(0, boundary);
        buf = buf.slice(boundary + 2);

        let event = 'out', data = '';
        for (const line of block.split('\n')) {
          if (line.startsWith('event: ')) event = line.slice(7).trim();
          else if (line.startsWith('data: ')) data = line.slice(6);
        }
        if (!data) continue;

        const text = JSON.parse(data);
        if (event === 'out') {
          outputEl.textContent += text;
          consoleBody.scrollTop = consoleBody.scrollHeight;
        } else if (event === 'done' && parseInt(text) !== 0) {
          outputEl.className = 'console-error';
        }
      }
    }
  } catch (err) {
    outputEl.textContent = `Error: ${err.message}`;
    outputEl.className = 'console-error';
  }

  if (!outputEl.textContent.trim()) outputEl.textContent = '(no output)';

  const elapsed = ((Date.now() - start) / 1000).toFixed(2);
  runBtn.disabled = false;
  runStatus.textContent = `${elapsed}s`;
  setTimeout(() => { runStatus.textContent = ''; }, 3000);
}

// ── Reset code ──
function resetCode() {
  if (editor && originalCode) {
    editor.setValue(originalCode);
    clearConsole();
  }
}

// ── Clear console ──
function clearConsole() {
  document.getElementById('console-body').innerHTML =
    '<div class="console-hint">Click Run (or Ctrl+Enter) to execute the code.</div>';
}

// ── Progress bar ──
function updateProgress(completed) {
  const pct = lessonCount ? Math.min(100, (completed / lessonCount) * 100) : 0;
  document.getElementById('progress-bar').style.width = `${pct}%`;
  document.getElementById('progress-label').textContent = `${completed} / ${lessonCount} lessons`;
}

// ── Console resize ──
function setupResizeHandle() {
  const handle = document.getElementById('resize-handle');
  const consolePane = document.getElementById('console-pane');

  let dragging = false;
  let startY = 0;
  let startH = 0;

  handle.addEventListener('mousedown', (e) => {
    dragging = true;
    startY = e.clientY;
    startH = consolePane.offsetHeight;
    handle.classList.add('dragging');
    document.body.style.cursor = 'row-resize';
    document.body.style.userSelect = 'none';
    e.preventDefault();
  });

  document.addEventListener('mousemove', (e) => {
    if (!dragging) return;
    const delta = startY - e.clientY; // drag up → bigger console
    const newH = Math.max(48, Math.min(startH + delta, window.innerHeight - 200));
    consolePane.style.height = `${newH}px`;
    if (editor) editor.layout();
  });

  document.addEventListener('mouseup', () => {
    if (!dragging) return;
    dragging = false;
    handle.classList.remove('dragging');
    document.body.style.cursor = '';
    document.body.style.userSelect = '';
  });
}

// ── Helpers ──
async function apiFetch(url, opts = {}) {
  try {
    const res = await fetch(url, {
      headers: { 'Content-Type': 'application/json' },
      ...opts,
      body: opts.body ? JSON.stringify(opts.body) : undefined,
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  } catch (err) {
    console.error('API error:', err);
    return null;
  }
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function extractTitle(markdown) {
  const match = markdown.match(/^#\s+(.+)$/m);
  return match ? match[1].trim() : 'Lesson';
}

function cleanTitle(title) {
  // Remove markdown bold/italic markers
  return title.replace(/[*_]/g, '');
}
