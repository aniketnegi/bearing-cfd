# Engineering and Documentation Standard

## Aim

Write software that a competent engineer can understand, change, test, and
review with bounded context.

Optimize for:

- correctness;
- conceptual integrity;
- local reasoning;
- explicit state and dependencies;
- change locality;
- readable control flow;
- stable, narrow contracts.

Do not optimize for line counts, file counts, function counts, class counts,
or the number of design patterns used.

The rules below are defaults serving that aim. Do not satisfy them
mechanically. When an exception produces clearer or safer code, make the
exception and state the trade-off briefly.

## Before changing code

1. Inspect the relevant implementation, tests, public contracts, and local
   repository conventions before proposing a design.
2. Identify:
   - the behaviour being changed;
   - the invariants that must remain true;
   - the owner of the affected data and policy;
   - the side effects involved;
   - compatibility constraints;
   - how the change will be verified.
3. For non-trivial work, present a plan in no more than five lines.
4. Make the smallest coherent change. Do not redesign, rename, reformat, or
   “clean up” unrelated code.
5. Ask for clarification only when ambiguity could alter behaviour, destroy
   data, weaken security, or change a public contract. Otherwise inspect the
   repository and proceed.

## Modules and boundaries

A module is a coherent responsibility. It is not necessarily a file, class,
package, service, or directory.

Before introducing a module, layer, interface, wrapper, factory, helper, or
new file, answer:

1. What decision, invariant, policy, lifecycle, authority, external effect,
   or knowledge does it own?
2. What complexity does it hide?
3. What likely change does it keep local?
4. Why is the additional interface, call hop, or navigation cost worthwhile?

If these questions have no clear answer, do not introduce the boundary.

Decompose by ownership of knowledge and reasons to change. Do not decompose
merely according to the order in which processing steps execute.

An interface should be substantially simpler than the implementation it
hides.

Every layer must add something meaningful: policy, invariant enforcement,
ownership, effect isolation, translation, lifecycle management, or a useful
capability. Collapse layers that merely forward arguments and results.

A single implementation behind an interface is a warning, not a prohibition.
Keep the interface when it establishes a stable public contract or isolates
unstable, external, privileged, or difficult-to-test code. Remove it when its
only purpose is hypothetical future substitution.

Use a factory when construction genuinely involves selection, ownership,
lifecycle, caching, dependency assembly, or multiple implementations. Do not
use one merely to hide a single constructor.

Abstract from concrete evidence. Two pieces of code belong behind one
abstraction when they have the same meaning, obey the same invariant, and
change for the same reason. The number of occurrences is not proof.
Temporary duplication is preferable to a guessed common abstraction.

Dependencies should point towards stable domain contracts. Keep databases,
networks, filesystems, operating-system calls, user interfaces, and vendor
APIs at explicit edges. Avoid dependency cycles.

## Functions, files, and control flow

Use no fixed line limit for functions or files.

Keep a cohesive, sequential algorithm together when extraction would:

- obscure its order of execution;
- scatter shared state;
- enlarge the parameter surface;
- add single-use call hops;
- force the reader to alternate between files.

Extract a part when doing so gives it a meaningful contract, isolates a side
effect, reduces state or nesting, represents a real domain operation, or
allows it to change and be tested independently.

A small helper is justified when its name establishes a useful domain concept
or invariant. It is not justified merely because it shortens its caller.

Split a file when it contains responsibilities that change for independent
reasons. Merge files when understanding one operation requires repeated
navigation and the boundaries add no policy, invariant, or reusable
capability.

Do not assume one class or type per file. Keep small, closely related types
and functions together when that improves reading.

Prefer direct call paths. For each additional hop, identify the semantic work
performed at that hop. Collapse calls that only rename or forward.

Reduce deep nesting with guard clauses, better data structures, or explicit
state machines when those make behaviour easier to see. Do not transform
straightforward code merely to satisfy a complexity metric.

## Data, state, and types

Design the data model, ownership, and invariants before designing control
flow.

Make dependencies and mutable state explicit. Avoid hidden globals, implicit
registries, service locators, and ambient context unless the platform
requires them.

Prefer pure transformations for computation and keep mutation local. Do not
force functional style where controlled in-place mutation is clearer or
required for measured performance.

Use types when they express domain meaning, encode ownership, or exclude
invalid states. Do not create decorative wrapper types that add no checking,
behaviour, or clarity.

Validate untrusted data at system boundaries. Once a boundary establishes an
invariant, internal code may rely on it rather than repeating defensive
checks everywhere.

Do not swallow errors. An error should identify the attempted operation, the
relevant object or value, and the underlying cause. Preserve useful context
when propagating errors. Ensure every failure path releases or rolls back
owned resources correctly.

Do not add performance complexity without measurement, except where the
repository already documents a known hard constraint. Report the benchmark
and the trade-off when optimizing.

## Tests

Test observable behaviour, contracts, and invariants rather than private
implementation structure.

Add a regression test for every fixed defect when the defect can be reproduced
reliably.

Test boundary conditions and important failure paths, not only the happy
path.

Keep tests deterministic unless randomness is the subject of the test. Make
external dependencies explicit and controlled.

Run the focused tests first, then the relevant wider suite, formatter, linter,
type checker, and static analysis tools. State exactly what was run. Never
claim that a test or command succeeded when it was not executed.

## Comments

Comments preserve information that code cannot express clearly.

Comment:

- rationale and rejected alternatives;
- invariants, ownership, and lifecycle;
- preconditions, postconditions, and side effects;
- non-obvious compatibility or performance constraints;
- unusual error handling;
- mathematical, protocol, regulatory, or algorithmic sources;
- reasons an apparently simpler implementation is incorrect.

Do not comment visible mechanics, repeat a signature, narrate each statement,
or compensate for unclear code. Rewrite the code first.

Document public contracts and surprising internal contracts. Do not require
boilerplate comments for every function.

Delete stale comments and commented-out code. Version control is the archive.

## Documentation

Plan the documentation set before creating pages. Identify its audiences,
their tasks, the concepts they must understand, and the information they must
look up.

Give each page one dominant reader need:

- tutorial: acquire a skill through a guided first success;
- how-to: complete a real task;
- reference: look up exact behaviour and contracts;
- explanation: understand design, rationale, limits, and trade-offs.

These are organizing categories, not purity laws. Include enough local
context to prevent needless page switching.

A page must complete the task or answer the question stated by its title.
Links should provide optional depth, adjacent tasks, or full reference
material. Do not require the reader to follow a link for a missing essential
step.

Prefer a smaller number of substantial, navigable pages over many fragments.
Split a page when it serves independent reader needs or maintenance
boundaries, not merely because it contains several headings.

Cross-link deliberately:

- tutorials point to the next practical tasks;
- how-to guides point to exact reference entries;
- reference entries point to relevant usage and rationale;
- explanations point to the procedures and contracts they discuss.

Use one stable vocabulary for each concept. State limitations, unsupported
uses, compatibility constraints, and trade-offs plainly.

Examples must be correct, representative, and safe to copy. Show failure
behaviour where it matters.

## Prose

Write direct, educated prose.

Lead with the fact, decision, or action. Use concrete subjects and active
verbs. Keep subjects near their verbs. Introduce known context before new
information.

Remove:

- throat-clearing and ceremonial introductions;
- generic praise;
- marketing language;
- vague claims of quality;
- repeated summaries;
- false “not merely X but Y” contrasts;
- ornamental lists of three;
- unnecessary headings;
- rhetorical questions that delay the answer;
- meta-commentary about what the document will now discuss;
- conclusions that repeat the preceding section.

Do not enforce a mechanical word blacklist. Demand precision instead.
“Robust”, “scalable”, “simple”, “flexible”, and similar adjectives require an
explicit dimension or evidence: robust against what, scalable to what load,
simple for which operation, flexible along which axis?

Do not imitate conversational LLM mannerisms. Do not praise the question,
announce that you will explore the subject, or narrate your own writing
process.

## Review passes

Review non-trivial work in separate passes:

1. Scope and architecture
   - Is the change at the correct boundary?
   - Does each unit own a coherent responsibility?
   - Has a new cycle or misplaced dependency appeared?

2. Deletion and flattening
   - What code, layer, option, wrapper, parameter, or configuration can be
     removed?
   - Can a pass-through call be collapsed?
   - Has speculative scaffolding been introduced?

3. Correctness
   - Are invariants preserved?
   - Are boundary cases, failure paths, security checks, ownership, and
     resource cleanup correct?
   - Does the change preserve required compatibility?

4. Readability
   - Are names truthful and consistent?
   - Is state explicit?
   - Is control flow visible?
   - Does understanding the change require unnecessary file hopping?

5. Tests and documentation
   - Do tests establish the intended behaviour?
   - Do comments preserve rationale rather than mechanics?
   - Are affected public contracts and user documents updated?
   - Is the prose exact and free of filler?

6. Cold diff
   - Read the final diff without relying on the implementation process.
   - Remove anything that a reviewer should not need to understand the
     logical change.
   - Verify the stated commands and results.

## Commits

Make one logical change per commit. Include every file required to keep that
logical change working and testable.

Separate behavioural changes, pure refactors, file moves, generated changes,
and broad formatting when practical. Do not create an intermediate commit
that knowingly breaks the build or tests merely to preserve an artificial
separation.

Do not include drive-by cleanup.

Use an imperative subject. In the body, explain:

1. the existing problem;
2. its observable or architectural consequence;
3. why this change is the appropriate solution;
4. important trade-offs or rejected alternatives.

The diff shows the mechanics. The commit message preserves the reasoning.

## Agent conduct

When a request would create clear architectural debt, object once. State the
specific consequence and propose the simplest sound alternative.

Do not become obstructive. After the user makes an informed choice, proceed
unless the request would violate correctness, safety, security, or data
integrity. Record the trade-off briefly.

Do not create empty interfaces, placeholder abstractions, unused extension
points, TODO scaffolding, or configuration for hypothetical requirements
unless they are part of an explicit staged plan.

At completion, report only:

- what changed;
- why it changed;
- what was verified;
- any remaining material risk or limitation.

Do not provide a process diary.

## Numerical and research software

Treat numerical assumptions as part of the public contract.

State where relevant:

- physical units and dimensions;
- coordinate systems and sign conventions;
- array shapes, data types, layouts, and valid ranges;
- precision assumptions;
- tolerances and convergence criteria;
- random seeds and sources of nondeterminism;
- dataset, model, and parameter provenance;
- hardware- or compiler-dependent behaviour.

Preserve operation ordering when reassociation can alter stability or
reproducibility.

Validate numerical work with the strongest available evidence:

- analytical or manufactured solutions;
- trusted reference implementations;
- experimental measurements;
- conservation laws and physical invariants;
- dimensional checks;
- convergence and sensitivity studies;
- limiting and degenerate cases.

Keep the mathematical model, discretization, solver, and input/output
concerns separate only when each owns a real decision that may change. Do not
turn pipeline stages into modules merely because they execute in sequence.

Keep measured computational kernels flat and data-conscious when abstraction
would impose material cost. Record the benchmark, precision, hardware, and
input scale behind that decision.

Place equation, paper, standard, or derivation references close to the code
whose correctness depends on them.
