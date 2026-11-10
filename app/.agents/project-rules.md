# Project Rules

## Tech Stack
- Language & Framework: TypeScript 5.9 (strict), React 19 with JSX (`react-jsx`), React Router 8 (declarative mode, imported from `react-router`), Vite 8 with `@vitejs/plugin-react`
- Package Manager: bun (`bun.lock`). Never add a second lockfile.
- Tests: `bun:test`
- Code colours: `sugar-high` (token API only). Prettier is a dev dependency used only by the lesson snippet test.
- Page text extraction: `@mozilla/readability` (pinned, Apache-2.0), used only by the Inside real models page's data demos. `happy-dom` is a dev dependency used only by those demos' tests, to parse HTML in `bun:test`.
- GPT-2's merge rules: `public/gpt-2/vocab.bpe` with OpenAI's Modified MIT `LICENSE` beside it, fetched from the app's own folder only by the Inside real models page's tokenizer demos.
- Runs entirely in the browser. There is no server, database or auth.
- Pages are path routes: `/` is Lessons, `/playground` is Playground and `/inside-real-models/:model/:mode/*` is Inside real models (`/inside-real-models` alone goes to `/inside-real-models/gpt-2-small/use`), through React Router. Old `/take-apart/...` links are replaced with the same path under `/inside-real-models` (`movedPath` in `features/inside-real-models/inside-real-models-path.ts`). Any host must rewrite every path to `index.html`, and `dist/` can no longer be opened straight from the file system.

## Architecture & Directory Map

```text
src/
├── main.tsx                    # loads global styles and renders <Layout /> inside <BrowserRouter>, nothing else
├── core/
│   ├── layout/                 # layout.tsx (routes, and the shell that holds the selected stage), sidebar.tsx (page NavLinks for Lessons, Playground and Inside real models; the stage buttons show on Lessons only) and layout.css
│   └── model/                  # the real models both pages run: tensor, tiny-gpt, adam, corpus, generate, lab, line-model, small-network, random
├── features/
│   ├── lessons/                # Lessons page: lessons-page, lesson-cards (StageCard/PartCard adapters), part-drawer (drawer body), lesson-plots (.tsx), the parts (*-steps.tsx), flow.ts, session.ts and lessons.css
│   │   ├── hooks/              # use-lessons (state and actions), use-session (session and frame loop)
│   │   └── interfaces/         # types private to Lessons
│   ├── playground/             # Playground page: playground-page, builder, menu, result (.tsx), playground-run.ts and playground.css
│   │   ├── hooks/              # use-playground (state and actions), use-training-loop, use-menu
│   │   └── interfaces/         # types private to Playground
│   └── inside-real-models/     # Inside real models page: inside-real-models-page, inside-real-models-header (mode switch, model select), sentence-bar, breadcrumb, part-column, part-card (its own card, not `FlowCard`), part-numbers (the real numbers strip: headline and one key fact), in-out (the in → out chips), part-drawer, fact-rows (.tsx), models.ts (routes and paths), real-place.ts (where a part sits in the real model: path, copies, share, runs per word, the card's key fact, `showAmount`), inside-real-models-path.ts (`INSIDE_REAL_MODELS_PATH`, and `OLD_INSIDE_REAL_MODELS_PATH` with `movedPath` for old links, read by core/layout), sentence.ts (reads the user's sentence), tiny-parts.ts (`viewTiny`, the live tiny view the demos read) and inside-real-models.css
│       ├── trees/              # one tree of real parts per model (gpt-2-*, llama-3-*), each model's Build facts (gpt-2-build-facts, llama-3-build-facts) and model-tree reasons (gpt-2-reasons, llama-3-reasons), the step builders for both modes (inference-steps, generate-steps, lifecycle-steps, training-steps, evaluation-steps), sources (every real fact's source, with its URL), part-builders and attention-math
│       ├── story/              # the drawer's "Why it exists" and "When you'd do this" (StoryInfo) and "Watch it happen" (StoryDemo) in story-sections, the lazy demo table (story-demos), shared demo pieces (watch-parts), story.css, and one folder per Build stage (data, tokenizer, architecture, training, post-training, evaluation) or Use step with its own demos (output, generate) holding its demos (.tsx), their tested logic (.ts) and a small stylesheet
│       ├── hooks/              # use-inside-real-models (selected path, model, drawer, column scrolling), use-tiny-lab (the page's own ModelLab and its frame loop), use-page-extract (the page demo's state and the saved page's extract), use-gpt-2-vocab (loads vocab.bpe once), use-demo-lab (a tiny model the user trains in a demo)
│       └── interfaces/         # model-part (ModelPart, PartStory, DemoId, DemoProps, RealModel, TinyLink, TinyView), real-facts (RealFacts, RealNote, KeyFact, Amount, Headline, Unstated, StepFacts, BuildFacts, UseReasons), part-place (PartPlace), inference-spec (InferenceSpec, StopWord, WritingNotes), sentence (Sentence), page-extract (PageExtract)
└── shared/
    ├── components/             # React components used by 2 or more modules: Icon, SparkleIcon, ChanceBars, LossPlot, FlowCard, StatusPill, FlowLinks, Drawer, FormulaBlock, CodeBlock, CopyButton, each with its own stylesheet beside it
    ├── hooks/                  # React hooks used by 2 or more modules: use-drawer, use-flow-links, use-window-width
    ├── interfaces/             # types used across core and features: card (Status, CardAction), formula, link, line, model
    ├── styles/                 # palette, variables, fonts, global
    └── utils/                  # helpers grouped by kind: format (percent, count, showWord, shortList), keyboard (activateOnKey)

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
- Formatting: `percent`, `count`, `showWord`, `shortList` (`shared/utils/format.ts`), `joinTokens` (`core/model/corpus.ts`). the Inside real models page's compact card numbers (124M, 8.03B) are `showAmount` in `features/inside-real-models/real-place.ts`.
- Tiny corpus constants: `START_CHOICES`, `TEST_WORDS` and `TEST_ANSWER` (the test sentence and its right next word) and `GUESSING_LOSS` (the loss of random guessing) live in `core/model/corpus.ts`, used by every page.
- Charts: the `ChanceBars` and `LossPlot` components in `src/shared/components/`. `PLOT` (plot size and margins) lives in `loss-plot.tsx`; Lessons' own plots read it from there.
- Sampling and writing: `applySampling`, `writeText`, `nextOptions`, `pickToken` in `core/model/generate.ts`.
- Seeded randomness: `seededRandom` in `core/model/random.ts`. Never use `Math.random` for anything a test or a comparison depends on.
- Icons: the `Icon` component (`<Icon name="chevron" />`) and `SparkleIcon` in `src/shared/components/icon.tsx`, reading paths from the one `ICON_PATHS` table in the same file. Draw new icons in the same style.
- Flow cards: `FlowCard` in `src/shared/components/flow-card.tsx` (styled by `flow-card.css`) draws the `<article class="card">` with `.card-head` (icon tile, title, lesson label, `StatusPill`) and `.card-body` (task line, foot value and action buttons), and clicks itself on Enter or Space (`activateOnKey` in `src/shared/utils/keyboard.ts`). The active card (`.card.is-active`), the card the user is focused on, shows the gradient header band (`--card-gradient`: indigo → sky → lavender → pink, fading to `--card-shell` below `--card-band-height`) around the lighter rounded `.card-body`, with white header text, tile and round pills (`--band-*` colours, darker in light mode so every header text and icon keeps 4.5:1); it has no coloured ring. The active card is the card whose drawer is open, if any, else the deepest selected card: on Lessons the open part, else the selected stage (plus the part a training run is on), and on Inside real models the open card, else the deepest card of the selected path (`activeIdOf` in `models.ts`). Every other selected card (the selected stage while a part is open, the rest of the selected path) gets the quiet `.card.is-selected` (neutral border and shadow). An open card has no ring of its own; keyboard focus keeps the `:focus-visible` outline. Props: `ref`, `id`, `icon`, `title`, `lesson`, `status: Status`, `task`, `foot: ReactNode`, `actions: CardAction[]`, `onCommand(command)`, an optional `className` added after `card` (state classes `is-dim`, `is-entering`, `is-active`, `is-selected`), and any other article attribute (`aria-*`, `style`, `onClick`) passed through. An action button stops its click, so the card's `onClick` does not run. `aria-label` is the title. Only Lessons uses `FlowCard`; the Inside real models page draws its own `PartCard` from the same shared pieces (`.card` and its state classes, `.card-body`, `.tile`, `.card-title`, `Icon`, `StatusPill`, `activateOnKey`, `.action`).
- Status pill: `StatusPill` (`{ status: Status }`) in `src/shared/components/status-pill.tsx`. `Status` and `CardAction` live in `src/shared/interfaces/card.interface.ts`. `Status` also has `"ground-zero"`, used only by the Inside real models page. Its `.pill` styles, the card heading classes (`.card-head`, `.tile`, `.card-title`, `.card-lesson`) and the `.card` shell + `.card-body` inner surface with its state classes (`is-dim`, `is-active`, `is-selected`, `is-entering`) live in `global.css`, because `FlowCard`, the Inside real models page's `PartCard` and `Drawer` use them; `flow-card.css` keeps only `FlowCard`'s task line and foot.
- Flow links: `FlowLinks` (`{ id, drawing: LinkDrawing | null, activeIds: string[] }`, renders `svg.links` with that id, styled by `flow-links.css`; Lessons passes `links`, each Inside real models page column `inside-real-models-links-<parent id>`, so mounted pages never share an id) and `STAGGER_MS` in `src/shared/components/flow-links.tsx`; `useFlowLinks(elements, options)` in `src/shared/hooks/use-flow-links.ts` measures the lines (`measureLinks`). Each path keeps `fromId`, the card it starts from (the parent, or the card before a nested child), and `toId`, the card it ends at; each dot keeps `fromId`. `activeIds` are the active cards (Lessons passes the open part, else the selected stage, and during a training run the selected stage and the running part, as `activeIdsOf` in `flow.ts` returns; every Inside real models column passes the active card, so the column that holds it finds the line into it). A line whose `fromId` or `toId` is in `activeIds` gets its own `userSpaceOnUse` `<linearGradient>` (`<svg id>-gradient-<toId>`, from its `startX` to its `endX`, stops `--link-indigo`, `--link-sky`, `--link-lavender`, `--link-pink` at the card band's offsets), and its start dot is `is-active` in `--link-indigo`. Only a line whose `toId` is in `activeIds` (the line into an active card) also gets the moving glow (`.link-glow`, the same path with a short dash moved along it, one pass per `--links-glow-duration` if a container sets it, else `--motion-flow-duration`, eased by `--motion-flow-ease`, hidden under reduced motion; Lessons sets `--links-glow-duration` to `STEP_ORDER_MS` on `.flow` during a training run, `RUN_GLOW_STYLE` in `flow.ts`, so one pass fits one training step); lines out of an active card stay still. Every other line and dot is the neutral `--link-quiet`. `elements`: `flowRef` (the positioned container the SVG fills), `childColumnRef` (the child column; the trunk runs halfway to it), `parentCards` and `childCards` (`RefObject<Map<string, HTMLElement>>` by id). `options`: `parentId`, `childIds` (memoized), optional `nestedIds` (a nested child links from the card before it, with a dot), `isShown`, `width` (measures again on change). `LinkDrawing` lives in `src/shared/interfaces/link.interface.ts`.
- Drawer: `Drawer` in `src/shared/components/drawer.tsx` (styled by `drawer.css`) draws the right-side overlay `aside` with header (tile, title, lesson, `StatusPill`, ✕) and a scrolling body holding `children`. Props: `ref`, `id`, `icon`, `title`, `lesson`, optional `status` (no pill when left out), `bodyKey` (a new key remounts the body so its scroll resets, without sliding in again), `onClose`, `children`, and an optional `panel` (a second, wider column beside the body; the drawer then opens wide as `.drawer--wide`, `--drawer-wide-width` / `--drawer-wide-share`). `useDrawer(triggers)` in `src/shared/hooks/use-drawer.ts` returns `{ openId, drawerRef, toggle(id), close }` (`close` is stable), and closes on Escape or a click outside the drawer and the cards in `triggers`, moves focus in on open and back to the card on close. Width is `--drawer-width`; at narrow width it is a full-screen sheet.
- Narrow width: `useWindowWidth()` and `NARROW_WIDTH` (960px, matching the stylesheets' `max-width: 960px`) in `src/shared/hooks/use-window-width.ts`.
- Snippets: `FormulaBlock` (`{ formula: Formula }`), `CodeBlock` (`{ code: string }`) and `CopyButton` (`{ text, label }`) in `src/shared/components/` (`formula-block.tsx`, `code-block.tsx`, `copy-button.tsx`, styled by `snippet-blocks.css`). `Formula` and `FormulaTerm` live in `src/shared/interfaces/formula.interface.ts`. Code colours come from `sugar-high` tokens rendered as JSX spans, never its HTML string.
- Frame loops: `use-session.ts` (Lessons), `use-training-loop.ts` (Playground) and `use-tiny-lab.ts` (Inside real models) stay in their features; they drive different objects in different ways, so there is no shared loop hook.
- Network, auth and notifications: none exist, and none are needed. A saved web page is stored in the app, never fetched. Large data files a demo needs (GPT-2's `vocab.bpe`) are served from the app's own `public/` folder and fetched only when that demo opens; nothing is fetched from another site.

## UI & Styling Conventions
- Rendering: React components with JSX, under `StrictMode`. No `dangerouslySetInnerHTML` and no `innerHTML`. Every effect cleans up its listeners, timers and frames.
- Live training: keep the mutable run object in state and redraw on each animation frame from a hook (`use-training-loop.ts` in Playground, `use-session.ts` in Lessons); never copy the model into state.
- Styling: plain CSS with custom properties, no preprocessor and no CSS framework.
- Flow layout: `.top`, `.top-stats`, `.column`, `.part-column` and the drawer body classes `.tasks`, `.task`, `.why` live in `global.css`, used by Lessons and Inside real models. `.fleet`, `.flow` and `.part-row` stay in `lessons.css`.
- Global styles live in `src/shared/styles/`:
  - `palette.css`: base colours first, then design colours as `var()` of a base colour, for light and dark. Components read design colours only. Code colours are `--code-identifier`, `--code-keyword`, `--code-string`, `--code-comment`, `--code-class` (numbers and capitalized names), `--code-property`, `--code-sign` and `--code-line-number`.
  - `variables.css`: radius, spacing, type, shadow, motion and layout sizes, including their `:root` overrides at each breakpoint.
  - `fonts.css`: `@font-face` and font variables. Inter is served from `public/fonts/`, never from a font CDN. `--font-mono` and `--font-math` are system font stacks.
  - `global.css`: element defaults and shared classes.
- A page's own variables stay in its stylesheet, named `--<page>-*`. Sizes a shared component reads (`--tile-size`, `--card-foot-height`, `--drawer-width`) live in `variables.css`.
- Never write a colour, spacing, radius, type, border, shadow or timing value in component CSS. Read a token.
- Motion: 150-350ms with ease-out, through `--motion-*` tokens. The spinner's constant rotation is the one linear animation. The glow on the line into an active card is the one other looping animation (`--motion-flow-duration` 1600ms, `--motion-flow-ease` ease-in-out; during a Lessons training run one pass per `STEP_ORDER_MS`, 450ms), and stops under reduced motion.
- Inside real models page look: the Lessons Fleet cards and lines in left → right columns that keep the whole selected path visible, scrolling sideways inside their own pane (stacked at 960px and below), with a breadcrumb that only shows the mode and path and scrolls to a card, a "Use a model | Build a model" segmented switch and a native `<select>` for the model, and a sentence bar for the live example. Column 1 holds the numbered step cards. Cards are "Simple first" (chosen by the user 2026-11-09): number badge or icon tile, title, lesson tag, one plain sentence (`summary`), in → out chips, a real-model numbers strip (the headline, then one more sourced real fact, stacked at full card width), and a foot with "N parts ›" and "Details ›"; everything else lives in the drawer.
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
- Routing: React Router 8 (declarative mode) with path routes `/` and `/playground` (and `/inside-real-models/:model/*` since the Inside real models page), chosen by the user 2026-11-09, replacing a hand-written History API router, and upgraded from React Router 7 to 8 the same day
- Rendering: React with full JSX and no `dangerouslySetInnerHTML` (rules are silent), chosen by the user 2026-11-09, replacing "HTML strings with no framework" from 2026-11-05. Both pages are moved
- File size: follow splitting large files (`references/code-structure.md:15`), answered 2026-11-05
- Shared helpers: follow splitting `pieces.ts` by who uses it (`references/file-organization.md:13`, `:20`), answered 2026-11-05
- Language: keep simple English in code with no translation system (`references/code-structure.md:73`), answered 2026-11-05
- Imports: add the `@/` alias (rules are silent), answered 2026-11-05
- Motion: follow 150-350ms through tokens (`references/design.md:44`), answered 2026-11-05
- Part detail: right-side overlay drawer, opened by clicking a part card only, chosen by the user 2026-11-09, replacing the floating panel under the card
- Shared flow UI: the card, status pill, flow links, drawer, snippet blocks and the window-width hook moved from Lessons to `src/shared/` for the coming "Inside real models" page (`/inside-real-models`), approved by the user 2026-11-09, ahead of its second user
- Inside real models page: top-down tour of a real LLM (GPT-2 small or Llama 3 8B from a dropdown), Fleet look in left → right columns that keep the whole selected path visible; the breadcrumb only shows the path and scrolls to a card; real sizes plus the tiny model live, down to a single multiply-and-add; chosen by the user 2026-11-09. The user chose this row of columns for this page even though a single long row was rejected for Lessons
- Inside real models steps: two modes, Use a model (5 steps) and Build a model (6 stages), as numbered step cards in the first column that open into the parts tree; live tiny example from the user's own sentence; chosen by the user 2026-11-09
- Real-model facts: every fact about GPT-2 small and Llama 3 8B on Inside real models (headlines, shapes, number counts, in/out labels, drawer rows, numbers in text) is stated in a published source or worked out exactly from published settings (marked `isComputed`), and links to an entry in `features/inside-real-models/trees/sources.ts`; a fact the makers never gave says "Not published", a step that does not apply says "None: …"; nothing is estimated, and the tiny model's values come only from the running tiny model and its hand-written corpus; chosen by the user 2026-11-10
- Why, watch and when: each Inside real models step gets "Why it exists" (what goes wrong without it, with the real model's reasons sourced or "Not published"), "Watch it happen" (a real, interactive before → after of exactly that step) and "When you'd do this" (when and what for, and the step it feeds), right after "What it does" in the drawer, as `ModelPart.story` plus `StepFacts.reasons`; a test lists the steps still missing one as todo; asked for by the user 2026-11-10 ("show a real cow eating grass"), "Copy one document in" first
- Wide drawer: a Inside real models part with a story opens a wide drawer with two columns, info on the left and the "Watch it happen" playground on the right, each scrolling on its own; stacked at 960px and below; parts with no story keep the narrow drawer; asked for by the user 2026-11-10
- Lazy loading: the Inside real models page and every demo are `React.lazy`, and big demo data is fetched from `public/` on demand, so the main chunk stays under Vite's 500 kB warning (349 kB, down from 561.64 kB); chosen 2026-11-10
- Demo honesty: a "Watch it happen" demo runs a real tool on real data live in the browser (a real public-domain page saved whole, with its URL and fetch date, and Mozilla Readability) and never shows a made-up output; when the tool differs from what the model's makers used it says so and names theirs, or "Not published"; when the tiny model lacks a part, the demo runs the real model's formula from its code on the tiny model's real numbers, or shows the nearest real step, and any number the user sets by hand is labelled so; chosen 2026-11-10
- Page name: "Take apart" was renamed "Inside real models" (route `/inside-real-models`, folder `features/inside-real-models`), with "Real models" as the phone top bar label, because the page looks inside the real models GPT-2 small and Llama 3 8B; old `/take-apart/...` links redirect with their path kept; chosen by the user 2026-11-10
- Real model only: Inside real models cards and info show only the real model; no tiny-model comparison. Demos may use the tiny model with honest labels. Each card shows the real headline plus one more sourced real fact (its first note, or how many times it runs per word, or its share of all numbers, worked out from the published settings), and the drawer's info column shows where the part sits, its shape, numbers, share and runs per word; asked for by the user 2026-11-10, replacing the "Real | Tiny model" strip and the "Live in our tiny model" section
