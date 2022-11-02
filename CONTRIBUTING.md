If you want to contribute to Aesara, have a look at the instructions here:
https://aesara.readthedocs.io/en/latest/dev_start_guide.html

## Contribution Expectations

This "Contribution Expectations" section is adapted from [Open Source
Archetypes](https://opentechstrategies.com/archetypes) and released under a
[CC-BY-SA](https://creativecommons.org/licenses/by-sa/4.0/) license.

The current focus of the Aesara project is rapid, high quality development of a
hackable, pure-Python symbolic tensor library with strong support for graph
rewrites and transpilation to multiple backends including C,
[JAX](https://github.com/google/jax>), and
[Numba](https://github.com/numba/numba).

We welcome patches with bug fixes, and we’re happy to answer questions if you’ve
already put in some effort to find the answer yourself. Please note, however,
that we’re _unlikely_ to consider new feature contributions or design changes
unless there’s a strong argument that they are fully in line with our stated
goals. If you’re not sure, just ask.

Our technical scope and project governance may open up later, of course, For
now, though, we would characterize this project as being a mix of the "Rocket
Ship To Mars" and "Specialty Library" archetypes (see
https://opentechstrategies.com/archetypes for details about RStM and other open
source archetypes).

## Issues and Discussions

We expect that Github Issues ("issues") indicate work that should be in Aesara
and can be picked up immediately by a contributor. This includes bugs, which
indicate something not working as advertised.

Discussions should be created when the scope or direction of the work, though
within the stated goals of the Aesara project, require additional clarification
or consideration before a course of action is chosen.

For issues a minimal working example (MWE) is strongly recommended when relevant
(fixing a typo in the documentation does not require a MWE). For discussions,
MWEs are generally required. All MWEs must be implemented using Aesara. Please
do not submit MWEs if they are not implemented in Aesara. In certain cases,
pseudocode may be acceptable, but an Aesara implementation is always preferable.
