# Project Rules

## Tech Stack
- Language & Framework: TypeScript 5.9 (strict), React 19 with JSX (`react-jsx`), React Router 8 (declarative mode, imported from `react-router`), Vite 8 with `@vitejs/plugin-react`
- Package Manager: bun (`bun.lock`). Never add a second lockfile.
- Tests: `bun:test`
- Code colours: `sugar-high` (token API only). Prettier is a dev dependency used only by the lesson snippet test.
- Runs entirely in the browser. There is no server, database or auth.
- Pages are path routes: `/` is Lessons, `/playground` is Playground and `/take-apart/:model/:mode/*` is Take apart (`/take-apart` alone goes to `/take-apart/gpt-2-small/use`), through React Router. Any host must rewrite every path to `index.html`, and `dist/` can no longer be opened straight from the file system.

## Architecture & Directory Map

```text
src/
├── main.tsx                    # loads global styles and renders <Layout /> inside <BrowserRouter>, nothing else
├── core/
│   ├── layout/                 # layout.tsx (routes, and the shell that holds the selected stage), sidebar.tsx (page NavLinks for Lessons, Playground and Take apart; the stage buttons show on Lessons only) and layout.css
│   └── model/                  # the real models both pages run: tensor, tiny-gpt, adam, corpus, generate, lab, line-model, small-network, random
├── features/
│   ├── lessons/                # Lessons page: lessons-page, lesson-cards (StageCard/PartCard adapters), part-drawer (drawer body), lesson-plots (.tsx), the parts (*-steps.tsx), flow.ts, session.ts and lessons.css
│   │   ├── hooks/              # use-lessons (state and actions), use-session (session and frame loop)
│   │   └── interfaces/         # types private to Lessons
│   ├── playground/             # Playground page: playground-page, builder, menu, result (.tsx), playground-run.ts and playground.css
│   │   ├── hooks/              # use-playground (state and actions), use-training-loop, use-menu
│   │   └── interfaces/         # types private to Playground
│   └── take-apart/             # Take apart page: take-apart-page, take-apart-header (mode switch, model select), sentence-bar, breadcrumb, part-column, part-card, part-drawer, live-example (.tsx), models.ts (routes and paths), sentence.ts (reads the user's sentence), tiny-parts.ts (live tiny model readouts) and take-apart.css
│       ├── trees/              # one tree of real parts per model (gpt-2-*, llama-3-*), the step builders for both modes (inference-steps, lifecycle-steps, training-steps), sources, part-builders and attention-math
│       ├── hooks/              # use-take-apart (selected path, model, drawer, column scrolling), use-tiny-lab (the page's own ModelLab and its frame loop)
│       └── interfaces/         # model-part (ModelPart, RealModel, RealNote, StepFacts, TinyLink, TinyReadout), sentence (Sentence)
└── shared/
    ├── components/             # React components used by 2 or more modules: Icon, SparkleIcon, ChanceBars, LossPlot, FlowCard, StatusPill, FlowLinks, Drawer, FormulaBlock, CodeBlock, CopyButton, each with its own stylesheet beside it
    ├── hooks/                  # React hooks used by 2 or more modules: use-drawer, use-flow-links, use-window-width
    ├── interfaces/             # types used across core and features: card (Status, CardAction), formula, link, line, model
    ├── styles/                 # palette, variables, fonts, global
    └── utils/                  # helpers grouped by kind: format (percent, count, showWord, shortList)

public/fonts/                   # Inter, served from the project
tests/                          # mirrors src/ at the same relative paths
docs/                           # one doc per page, kept out of git by docs/.gitignore
```

- `core/model`: real model code only. It never imports from `features/` and never touches the DOM.
- `features/<page>`: everything private to one page. A feature never imports from another feature.
- `shared`: only what 2 or more modules use, with no page-specific logic. Never place a file loose in `shared/`.
- New page code belongs in: `src/features/<page>/`, with a sidebar link in `src/core/layout/`
- Shared primitives belong in: `src/shared/<kind>/`
- Each UI region has its own file. `main.tsx` only mounts the layout.
- A feature's React hooks live in `src/features/<page>/hooks/`; hooks used by 2 or more modules live in `src/shared/hooks/`.
- Keep a file readable in one pass. Split by responsibility when it passes about 300 lines.

## Import Aliases & Paths
- `@/*`: `src/*`, set in both `tsconfig.json` (`compilerOptions.paths`) and `vite.config.ts` (`resolve.alias`).
- Use `@/` for imports across folders, and `./` for imports within the same folder.

## Naming Conventions
- Files: `kebab-case.ts`, and `kebab-case.tsx` for files with JSX (components). Hooks: `use-<name>.ts` exporting `use<Name>`. Shared types live in `<name>.interface.ts`.
- Components: PascalCase named function exports, e.g. `export function LossPlot`.
- Exports: named exports only, no default exports. The only exception is what `vite.config.ts` requires.
- Functions and variables: `camelCase`. Module constants: `UPPER_SNAKE_CASE`. Booleans read as questions (`isDone`, `hasAttention`).
- Types and interfaces: PascalCase with no `I` prefix. `interface` for object shapes, `type` for unions and aliases.
- CSS: kebab-case classes, `is-*` for states, `<block>--<variant>` for variants.
- Domain terms:
  - Use `layer`, never `block`, for one attention + feed-forward unit in code. Lesson text may say "transformer block" where it explains the idea.
  - Use `round`, never `step`, for one training pass (guess, score, find the cause, fix). "Step size" stays as the plain name for the learning rate.
  - `stage` is a left-column card. `part` is what the screen calls a right-column card, and the code calls it `FlowStep` today. `FlowStep` and `STEPS` are not training rounds.
  - `weights` in code are the "numbers" on screen. `token` in code is a "word" on screen.

## Core Shared Helpers (Do Not Duplicate)
- Markup: JSX only. Never build HTML strings; JSX escapes text itself. Keep keys stable so hover, focus and scroll survive a redraw.
- Formatting: `percent`, `count`, `showWord`, `shortList` (`shared/utils/format.ts`), `joinTokens` (`core/model/corpus.ts`).
- Tiny corpus constants: `START_CHOICES`, `TEST_WORDS` and `TEST_ANSWER` (the test sentence and its right next word) and `GUESSING_LOSS` (the loss of random guessing) live in `core/model/corpus.ts`, used by every page.
- Charts: the `ChanceBars` and `LossPlot` components in `src/shared/components/`. `PLOT` (plot size and margins) lives in `loss-plot.tsx`; Lessons' own plots read it from there.
- Sampling and writing: `applySampling`, `writeText`, `nextOptions`, `pickToken` in `core/model/generate.ts`.
- Seeded randomness: `seededRandom` in `core/model/random.ts`. Never use `Math.random` for anything a test or a comparison depends on.
- Icons: the `Icon` component (`<Icon name="chevron" />`) and `SparkleIcon` in `src/shared/components/icon.tsx`, reading paths from the one `ICON_PATHS` table in the same file. Draw new icons in the same style.
- Flow cards: `FlowCard` in `src/shared/components/flow-card.tsx` (styled by `flow-card.css`) draws the `<article class="card">` with icon tile, title, lesson label, `StatusPill`, task line, foot value and action buttons, and clicks itself on Enter or Space. Props: `ref`, `id`, `icon`, `title`, `lesson`, `status: Status`, `task`, `foot: ReactNode`, `actions: CardAction[]`, `onCommand(command)`, an optional `className` added after `card` (state classes `is-dim`, `is-entering`, `is-active`, `is-open`), optional `children` drawn under the task line, and any other article attribute (`aria-*`, `style`, `onClick`) passed through. An action button stops its click, so the card's `onClick` does not run. `aria-label` is the title.
- Status pill: `StatusPill` (`{ status: Status }`) in `src/shared/components/status-pill.tsx`. `Status` and `CardAction` live in `src/shared/interfaces/card.interface.ts`. Its `.pill` styles and the card heading classes (`.card-head`, `.tile`, `.card-title`, `.card-lesson`) live in `global.css`, because both `FlowCard` and `Drawer` use them.
- Flow links: `FlowLinks` (`{ id, drawing: LinkDrawing | null }`, renders `svg.links` with that id, styled by `flow-links.css`; Lessons passes `links`, each Take apart column `take-apart-links-<parent id>`, so mounted pages never share an id) and `STAGGER_MS` in `src/shared/components/flow-links.tsx`; `useFlowLinks(elements, options)` in `src/shared/hooks/use-flow-links.ts` measures the lines. `elements`: `flowRef` (the positioned container the SVG fills), `childColumnRef` (the child column; the trunk runs halfway to it), `parentCards` and `childCards` (`RefObject<Map<string, HTMLElement>>` by id). `options`: `parentId`, `childIds` (memoized), optional `nestedIds` (a nested child links from the card before it, with a dot), `isShown`, `width` (measures again on change). `LinkDrawing` lives in `src/shared/interfaces/link.interface.ts`.
- Drawer: `Drawer` in `src/shared/components/drawer.tsx` (styled by `drawer.css`) draws the right-side overlay `aside` with header (tile, title, lesson, `StatusPill`, ✕) and a scrolling body holding `children`. Props: `ref`, `id`, `icon`, `title`, `lesson`, `status`, `bodyKey` (a new key remounts the body so its scroll resets, without sliding in again), `onClose`, `children`. `useDrawer(triggers)` in `src/shared/hooks/use-drawer.ts` returns `{ openId, drawerRef, toggle(id), close }` (`close` is stable), and closes on Escape or a click outside the drawer and the cards in `triggers`, moves focus in on open and back to the card on close. Width is `--drawer-width`; at narrow width it is a full-screen sheet.
- Narrow width: `useWindowWidth()` and `NARROW_WIDTH` (960px, matching the stylesheets' `max-width: 960px`) in `src/shared/hooks/use-window-width.ts`.
- Snippets: `FormulaBlock` (`{ formula: Formula }`), `CodeBlock` (`{ code: string }`) and `CopyButton` (`{ text, label }`) in `src/shared/components/` (`formula-block.tsx`, `code-block.tsx`, `copy-button.tsx`, styled by `snippet-blocks.css`). `Formula` and `FormulaTerm` live in `src/shared/interfaces/formula.interface.ts`. Code colours come from `sugar-high` tokens rendered as JSX spans, never its HTML string.
- Frame loops: `use-session.ts` (Lessons), `use-training-loop.ts` (Playground) and `use-tiny-lab.ts` (Take apart) stay in their features; they drive different objects in different ways, so there is no shared loop hook.
- Network, auth and notifications: none exist, and none are needed.

## UI & Styling Conventions
- Rendering: React components with JSX, under `StrictMode`. No `dangerouslySetInnerHTML` and no `innerHTML`. Every effect cleans up its listeners, timers and frames.
- Live training: keep the mutable run object in state and redraw on each animation frame from a hook (`use-training-loop.ts` in Playground, `use-session.ts` in Lessons); never copy the model into state.
- Styling: plain CSS with custom properties, no preprocessor and no CSS framework.
- Flow layout: `.top`, `.top-stats`, `.column`, `.part-column` and the drawer body classes `.tasks`, `.task`, `.why` live in `global.css`, used by Lessons and Take apart. `.fleet`, `.flow` and `.part-row` stay in `lessons.css`.
- Global styles live in `src/shared/styles/`:
  - `palette.css`: base colours first, then design colours as `var()` of a base colour, for light and dark. Components read design colours only. Code colours are `--code-identifier`, `--code-keyword`, `--code-string`, `--code-comment`, `--code-class` (numbers and capitalized names), `--code-property`, `--code-sign` and `--code-line-number`.
  - `variables.css`: radius, spacing, type, shadow, motion and layout sizes, including their `:root` overrides at each breakpoint.
  - `fonts.css`: `@font-face` and font variables. Inter is served from `public/fonts/`, never from a font CDN. `--font-mono` and `--font-math` are system font stacks.
  - `global.css`: element defaults and shared classes.
- A page's own variables stay in its stylesheet, named `--<page>-*`. Sizes a shared component reads (`--tile-size`, `--card-foot-height`, `--drawer-width`) live in `variables.css`.
- Never write a colour, spacing, radius, type, border, shadow or timing value in component CSS. Read a token.
- Motion: 150-350ms with ease-out, through `--motion-*` tokens. The spinner's constant rotation is the one linear animation.
- Take apart page look: the Lessons Fleet cards and lines in left → right columns that keep the whole selected path visible, scrolling sideways inside their own pane (stacked at 960px and below), with a breadcrumb that only shows the mode and path and scrolls to a card, a "Use a model | Build a model" segmented switch and a native `<select>` for the model, and a sentence bar for the live example. Column 1 holds the numbered step cards.
- Lessons page look: reference video of a "Fleet" dashboard. Stages on the left, parts on the right, joined by lines. Clicking a part card opens its detail in a right-side overlay drawer. A single long row running to the right was tried and rejected.
- Playground page look: reference video of a "Guard risky commands" rule builder.
- Every chip and control must change something real. Never add a control that only looks like the reference.
- Text is simple English, written directly in the code.

## Verification Commands
- Lint: none. There is no linter or formatter for project source. Prettier only checks lesson code snippets, inside `bun run test`.
- Typecheck: `bunx tsc --noEmit`
- Test: `bun run test`
- Build: `bun run build` (runs the typecheck first)
- Run: `bun run dev`
- UI changes: check them by hand in Chrome, in light and dark and at phone width, and report anything left unchecked.

## Decisions
- Folder layout: follow `core/`, `shared/`, `features/` (`references/file-organization.md:11`), answered 2026-11-05
- UI regions: follow one file per region, entry only mounts (`references/file-organization.md:15`, `:17`), answered 2026-11-05
- Shared types: follow the shared interfaces folder for types used by 2 or more files (`frameworks/typescript/rules.md:11`), answered 2026-11-05
- Palette: follow base colours plus design colours (`frameworks/css/rules.md:13`), answered 2026-11-05
- Design values: follow tokens only, converting all existing CSS now (`references/design.md:11`), answered 2026-11-05
- Global CSS: follow separate global, variables and page stylesheets, and lock the shell (`frameworks/css/rules.md:7`, `:17`, `:19`, `:20`, `:24`), answered 2026-11-05
- Font: follow self-hosted Inter (`frameworks/css/rules.md:18`), answered 2026-11-05
- Docs: follow one `docs/<page-name>.md` per page (`references/code-structure.md:33`), answered 2026-11-05
- Terms: follow one term per concept, using `layer` and `round` (`references/naming.md:7`), answered 2026-11-05
- Fixed lists: keep `STAGES`, `STEPS` and `ROWS` as lists plus a loop, an exception to `references/code-structure.md:20-21`, because tests loop over them to check every lesson 00-30 has a part, answered 2026-11-05
- Linter and formatter: keep none, `tsc` only (`references/project-hygiene.md:30-31`), answered 2026-11-05
- Code snippets: lesson code snippets are Prettier-formatted and checked by a test; Prettier is a dev dependency used only for that, chosen by the user 2026-11-09
- UI tests: keep none. Logic stays in tested model and feature code, and screens are checked by hand in Chrome (`references/project-hygiene.md:20`, `:23`), answered 2026-11-05
- Routing: React Router 8 (declarative mode) with path routes `/` and `/playground` (and `/take-apart/:model/*` since the Take apart page), chosen by the user 2026-11-09, replacing a hand-written History API router, and upgraded from React Router 7 to 8 the same day
- Rendering: React with full JSX and no `dangerouslySetInnerHTML` (rules are silent), chosen by the user 2026-11-09, replacing "HTML strings with no framework" from 2026-11-05. Both pages are moved
- File size: follow splitting large files (`references/code-structure.md:15`), answered 2026-11-05
- Shared helpers: follow splitting `pieces.ts` by who uses it (`references/file-organization.md:13`, `:20`), answered 2026-11-05
- Language: keep simple English in code with no translation system (`references/code-structure.md:73`), answered 2026-11-05
- Imports: add the `@/` alias (rules are silent), answered 2026-11-05
- Motion: follow 150-350ms through tokens (`references/design.md:44`), answered 2026-11-05
- Part detail: right-side overlay drawer, opened by clicking a part card only, chosen by the user 2026-11-09, replacing the floating panel under the card
- Shared flow UI: the card, status pill, flow links, drawer, snippet blocks and the window-width hook moved from Lessons to `src/shared/` for the coming "Take apart" page (`/take-apart`), approved by the user 2026-11-09, ahead of its second user
- Take apart page: top-down tour of a real LLM (GPT-2 small or Llama 3 8B from a dropdown), Fleet look in left → right columns that keep the whole selected path visible; the breadcrumb only shows the path and scrolls to a card; real sizes plus the tiny model live, down to a single multiply-and-add; chosen by the user 2026-11-09. The user chose this row of columns for this page even though a single long row was rejected for Lessons
- Take apart steps: two modes, Use a model (5 steps) and Build a model (6 stages), as numbered step cards in the first column that open into the parts tree; live tiny example from the user's own sentence; chosen by the user 2026-11-09
