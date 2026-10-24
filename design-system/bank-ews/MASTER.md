# Design System Master File

> **LOGIC:** When building a specific page, first check `design-system/bank-ews/pages/[page-name].md`.
> If that file exists, its rules **override** this Master file.
> If not, strictly follow the rules below.

---

**Project:** Bank EWS ("Two ways banks fail": a research write-up with one interactive bank lookup)
**Generated:** 2026-10-23 by ui-ux-pro-max (`"fintech risk analytics dashboard data report" --design-system --density 7`)
**Taste-checked:** 2026-10-23 with frontend-design. Every value below is final, and the revisions are recorded at the end.
**Category:** Analytics Dashboard, used as a **Read** surface (a supervisory report) with one **Operate** section (the lookup)
**Design Dials:** Density 7/10 (Standard)

---

## Global Rules

### Color Palette: UI and text

| Role | Light | Dark | CSS variable on the page |
|------|-------|------|--------------------------|
| Background | `#F8FAFC` | `#0F172A` | `--bg` |
| Card / figure surface | `#FFFFFF` | `#152036` | `--surface` |
| Heading ink (Foreground) | `#1E3A8A` | `#E8EEF8` | `--ink-strong` |
| Body ink | `#0F172A` | `#E2E8F0` | `--ink` |
| Secondary text (Muted Foreground) | `#475569` | `#A3B1C6` | `--ink-2` |
| Axis labels, placeholders | `#64748B` | `#8494AB` | `--muted` |
| Border | `#E2E8F0` | `#26344D` | `--rule` |
| Axis lines | `#CBD5E1` | `#3A4A66` | `--axis` |
| Muted fill (highlight, row hover) | `#E9EEF6` | `#1C2A44` | `--hl` |
| Primary: links, focus ring, pressed controls | `#1E40AF` | `#93C5FD` | `--primary` |
| On Primary | `#FFFFFF` | `#0F172A` | `--on-primary` |
| Destructive: the "flagged" or "failed" status only | `#DC2626` | `#F87171` | `--fail` |

### Color Palette: chart series (validated with dataviz `validate_palette.js --pairs all`)

| Series | Light (on `#FFFFFF`) | Dark (on `#152036`) | Variable |
|--------|----------------------|---------------------|----------|
| Run lens (Secondary) | `#3B82F6` | `#4B8EF0` | `--run` |
| Credit lens (Accent) | `#D97706` | `#C98012` | `--credit` |
| Failure model | `#00806A` | `#1E9E70` | `--model` |

- Light passes all five checks: worst CVD ΔE 11.1, worst normal-vision ΔE above 15, contrast at least 3:1. Dark also passes all five, with worst CVD ΔE 8.6.
- `--fail` is a status colour, never a series. It always ships with a text label ("Flagged", "Failed").
- The fragile-zone tint is `--fail` at 7% light and 9% dark (`--zone`).

### Typography

- **Display (h1–h3):** Fira Sans Condensed, 600/700
- **Body and UI:** Fira Sans, 400/500/600
- **Data (axis ticks, table numbers, tooltips):** Fira Sans with `font-variant-numeric: tabular-nums`. No monospace face anywhere.
- **Google Fonts:** `https://fonts.googleapis.com/css2?family=Fira+Sans+Condensed:wght@600;700&family=Fira+Sans:ital,wght@0,400;0,500;0,600;1,400&display=swap`
- **Scale:** body 16/1.6. Lede 19. h1 clamp(44px, 8vw, 80px)/0.95. h2 clamp(26px, 3.6vw, 34px)/1.15. h3 19. Figure title 16. Caption 14. Chart text 12 minimum.
- **Labels:** sentence case everywhere, including table headers. No tracked-out all caps.

### Spacing Variables (density 7)

| Token | Value | Usage |
|-------|-------|-------|
| `--space-xs` | `4px` | Tight gaps |
| `--space-sm` | `8px` | Inline spacing, chip gaps |
| `--space-md` | `16px` | Standard padding, phone gutter |
| `--space-lg` | `24px` | Figure and panel padding |
| `--space-xl` | `32px` | Grid gaps |
| `--space-2xl` | `48px` | Hero top padding |
| `--space-3xl` | `64px` | Gap between sections |

### Radius and elevation

- **Radius by role:** controls and inputs 8px; figures, panels and memos 8px; chips and pills 999px; tooltips 6px.
- **Elevation:** surfaces are separated by a 1px `--rule` border, not a shadow. `--shadow-lg` (`0 10px 15px rgba(15,23,42,.12)`) is reserved for things that float: the results listbox and the chart tooltip.

---

## Component Specs

### Inputs (the bank search)
```css
.input { padding: 12px 16px; min-height: 46px; border: 1px solid var(--axis); border-radius: 8px; font-size: 16px;
         background: var(--surface); transition: border-color 150ms ease, box-shadow 150ms ease; }
.input:focus-visible { border-color: var(--primary); outline: none; box-shadow: 0 0 0 3px color-mix(in srgb, var(--primary) 25%, transparent); }
```

### Buttons (segmented control, chips, copy-link)
- These are quiet controls. Rest: `--surface` background, `--rule` border, `--ink` text.
- Hover: `--hl` background.
- Pressed or current: `--primary` background, `--on-primary` text.
- Minimum 44px target. Transition: colour and background over 150ms. `cursor: pointer`.
- There is no amber call-to-action button. The page has nothing to sell.

### Figures
- `--surface`, 1px `--rule` border, 8px radius, `--space-lg` padding (`--space-md` under 640px).
- No hover lift and no pointer cursor: figures are not clickable.

### Tables
- Sentence-case headers in `--ink-2`, weight 600.
- Rows hover-highlight with `--hl`. Numbers are tabular and right-aligned.

---

## Style Guidelines

**Style:** Data-Dense Dashboard, applied **inside figures and the lookup only**. The prose stays one ~68ch reading column.

**Key effects, as kept:**
- hover tooltips on every chart mark;
- row highlighting on hover;
- 150ms colour transitions on controls.

**Dropped from the generated list:** chart zoom on click, filter animations and loading spinners. There are no filters, and loading is a text state.

### Page pattern: a research report with a lookup (replaces the generated "Enterprise Gateway")
Header (h1, lede, lens key, contents) > four findings > glossary > 2023 panic > where the 2023 failures sat > 2008 crisis > look up a bank > the failure model > what went wrong first > how it was kept honest > footer.

---

## Anti-Patterns (do not use)

- ❌ Ornate design, or decoration without data
- ❌ Monospace as a "technical" costume, including the generated Fira Code headings
- ❌ All-caps tracked labels, and eyebrows above headings
- ❌ Shadowed cards that lift on hover
- ❌ Colour alone carrying meaning. Statuses get words; series get legends and tables.
- ❌ Emojis as icons
- ❌ Layout-shifting hovers
- ❌ Low-contrast text (below 4.5:1)
- ❌ Invisible focus states

---

## Pre-Delivery Checklist
- [ ] Every chart series colour comes from the validated table above
- [ ] `cursor: pointer` on every clickable control
- [ ] Hover and pressed transitions run 150ms and are off under `prefers-reduced-motion`
- [ ] Text contrast is at least 4.5:1 in both themes
- [ ] Focus is visible: a 2px `--primary` outline, offset 2px
- [ ] Responsive at 375, 768, 1024 and 1440 with no horizontal scroll

---

## Taste-check revisions (frontend-design, 2026-10-23)

| Generated | Revised to | Why |
|---|---|---|
| Pattern "Enterprise Gateway" (contact-sales CTA, client logos, mega menu) | Research report plus lookup | The page sells nothing. Its job is to let a fintech risk reader judge the evidence and look up a bank. A gateway pattern would be a template, not a choice. |
| Heading font Fira Code (mono) | Fira Sans Condensed | Mono headings are the "technical" costume both Impeccable and frontend-design flag. The condensed cut keeps the Fira family and gives headlines the dense, document-like voice of supervisory reports. |
| Body text in Foreground `#1E3A8A` | Navy for headings only; body `#0F172A` | Long navy prose tires the eye. Navy headings keep the "high integrity" colour strategy. |
| Accent `#D97706` as an amber CTA button | Amber becomes the **credit lens** series colour | There is no CTA. The old credit orange (`#d85a24`) and the fail red failed the normal-vision floor (ΔE 9.2). Amber against red does better, and red is now a status only. |
| Secondary `#3B82F6` | The run lens series colour | It keeps blue as the "run" identity the page already used, and passes validation. |
| Failure-model green `#0d8a5c` (kept from the old page) | `#00806A` light, `#1E9E70` dark | It cleared the colour-blind separation from amber (protan ΔE 7.3 → 11.1). |
| Border `#DBEAFE` | `#E2E8F0` (the generated input border) | `#DBEAFE` disappears against `#F8FAFC`, and figure edges need to read. |
| Card shadow-md and hover lift, `cursor: pointer` on cards | Border only; no lift | Figures aren't clickable. Lift and pointer would promise an action that doesn't exist. |
| Light mode only | Dark tokens added (slate, not near-black) | The page must work in both themes. Slate `#0F172A` avoids the near-black-with-acid-accent default. |
