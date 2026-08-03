# Documentation Instructions

[ENGINEERING.md](../ENGINEERING.md) governs prose and documentation structure.
These rules add repository-specific constraints.

- Write for GitHub rendering first. Use relative links, useful link labels,
  readable source blocks, and images whose paths resolve from the page.
- Organize the set with Diataxis: tutorials create a first success, how-to
  guides complete a task, reference pages state exact contracts, and
  explanations develop rationale and limitations. Give each page one dominant
  reader need and cross-link related needs.
- Do not number filenames or titles merely to impose reading order. Navigation
  and descriptive names carry that information.
- Document the present system. Git stores project history. Do not preserve
  migration narratives, internal conversations, review choreography, or old
  commands unless a page explicitly studies chronology.
- Write as a project maintainer, not as an assistant speaking for the author.
  Avoid readiness claims, self-congratulation, implementation diaries, and
  statements about absent placeholder code. State capabilities, contracts,
  limitations, and evidence directly.
- Integrate useful handoff material into the tutorial, how-to, reference, or
  explanation page that owns it. Do not retain a parallel handoff manual after
  its content has an appropriate permanent home.
- Leave Org `#+UPDATED_AT` lines to Emacs. Do not add, remove, or manually edit
  them.
- When commands, paths, boundary names, statuses, or output formats change,
  update every affected page in the same change and run the documented command
  where practical.
- Research reports are working presentation aids, not dashboard UIs. Prefer a
  plain Typst document, normal headings, figures, captions, equations, and
  compact prose. Avoid cards, decorative boxes, and large summary tables.
  Use Typst's default Libertinus Serif unless the user asks otherwise.
- Center the project report on OpenFOAM results. Treat teammate Fluent results
  only as clearly attributed external comparison data.
- Every screenshot, graph, log excerpt, and numerical result must identify its
  case or source artifact. Separate observed values from derived quantities and
  label numerical convergence independently from physical validity.
