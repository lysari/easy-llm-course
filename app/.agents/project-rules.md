# Project Rules

## Tech Stack
- Language & Framework: TypeScript 5.9 (strict), plain DOM with no UI framework, Vite 8
- Package Manager: bun (`bun.lock`). Never add a second lockfile.
- Tests: `bun:test`
- Runs entirely in the browser. There is no server, database or auth.

## Architecture & Directory Map

```text
src/
├── main.ts                     # loads global styles and mounts the layout, nothing else
├── core/
│   ├── layout/                 # shell, sidebar, hash page switching and layout.css
│   └── model/                  # the real models both pages run: tensor, tiny-gpt, adam, corpus, generate, lab, line-model, small-network, random
├── features/
│   ├── lessons/                # Lessons page: flow, stages, parts, session, its plots and lessons.css
│   │   └── interfaces/         # types private to Lessons
│   └── playground/             # Playground page: builder, result, menu, playground run and playground.css
│       └── interfaces/         # types private to Playground
└── shared/
    ├── interfaces/             # types used across core and features
    ├── styles/                 # palette, variables, fonts, global
    └── utils/                  # helpers used by both pages, grouped by kind, plus icons

public/fonts/                   # Inter, served from the project
tests/                          # mirrors src/ at the same relative paths
docs/                           # one doc per page, kept out of git by docs/.gitignore
```

- `core/model`: real model code only. It never imports from `features/` and never touches the DOM.
- `features/<page>`: everything private to one page. A feature never imports from another feature.
- `shared`: only what 2 or more modules use, with no page-specific logic. Never place a file loose in `shared/`.
- New page code belongs in: `src/features/<page>/`, with a sidebar link in `src/core/layout/`
- Shared primitives belong in: `src/shared/<kind>/`
- Each UI region has its own file. `main.ts` only mounts the layout.
- Keep a file readable in one pass. Split by responsibility when it passes about 300 lines.

## Import Aliases & Paths
- `@/*`: `src/*`, set in both `tsconfig.json` (`compilerOptions.paths`) and `vite.config.ts` (`resolve.alias`).
- Use `@/` for imports across folders, and `./` for imports within the same folder.

## Naming Conventions
- Files: `kebab-case.ts`. Shared types live in `<name>.interface.ts`.
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
- HTML and DOM: `escapeHtml` for every word placed in markup, `setHtml` to skip unchanged HTML. Both must keep their current behaviour, because `setHtml` preserves hover, focus and scroll.
- Formatting: `percent`, `count`, `showWord`, `joinTokens`.
- Charts used by both pages: `chanceBars`, `lossPlot`.
- Sampling and writing: `applySampling`, `writeText`, `nextOptions`, `pickToken` in `core/model/generate.ts`.
- Seeded randomness: `seededRandom` in `core/model/random.ts`. Never use `Math.random` for anything a test or a comparison depends on.
- Icons: `iconSvg`, with paths in one table. Draw new icons in the same style.
- Network, auth and notifications: none exist, and none are needed.

## UI & Styling Conventions
- Rendering: plain TypeScript builds HTML strings and sets `innerHTML`. No UI framework.
- Styling: plain CSS with custom properties, no preprocessor and no CSS framework.
- Global styles live in `src/shared/styles/`:
  - `palette.css`: base colours first, then design colours as `var()` of a base colour, for light and dark. Components read design colours only.
  - `variables.css`: radius, spacing, type, shadow, motion and layout sizes, including their `:root` overrides at each breakpoint.
  - `fonts.css`: `@font-face` and font variables. Inter is served from `public/fonts/`, never from a font CDN.
  - `global.css`: element defaults and shared classes.
- A page's own variables stay in its stylesheet, named `--<page>-*`.
- Never write a colour, spacing, radius, type, border, shadow or timing value in component CSS. Read a token.
- Motion: 150-350ms with ease-out, through `--motion-*` tokens. The spinner's constant rotation is the one linear animation.
- Lessons page look: reference video of a "Fleet" dashboard. Stages on the left, parts on the right, joined by lines. A single long row running to the right was tried and rejected.
- Playground page look: reference video of a "Guard risky commands" rule builder.
- Every chip and control must change something real. Never add a control that only looks like the reference.
- Text is simple English, written directly in the code.

## Verification Commands
- Lint: none. There is no linter or formatter.
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
- UI tests: keep none. Logic stays in tested model and feature code, and screens are checked by hand in Chrome (`references/project-hygiene.md:20`, `:23`), answered 2026-11-05
- Rendering: keep HTML strings with no framework (rules are silent), answered 2026-11-05
- File size: follow splitting large files (`references/code-structure.md:15`), answered 2026-11-05
- Shared helpers: follow splitting `pieces.ts` by who uses it (`references/file-organization.md:13`, `:20`), answered 2026-11-05
- Language: keep simple English in code with no translation system (`references/code-structure.md:73`), answered 2026-11-05
- Imports: add the `@/` alias (rules are silent), answered 2026-11-05
- Motion: follow 150-350ms through tokens (`references/design.md:44`), answered 2026-11-05
