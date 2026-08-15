# Suggested ChatGPT analysis prompt

Analyze this repository as a registered cross-dataset experiment, not as a
model leaderboard. Cite exact files and numbers for every conclusion.

Please answer:

1. Why did G1 fail even though some targets improved in five or six datasets?
2. Why did measured scalar internal descriptors fail G2, and what does the
   measured-scalar oracle result imply for predicted scalars?
3. Why did P1 pass while G1 and G2 failed? Distinguish scalarization loss from
   absence of internal mechanical information.
4. Why did P2 fail despite a 4.34% mean improvement over the equal-capacity
   student? Compare authentic MSPD with scalar, shuffled, and random teachers.
5. Are the P1 and P2 improvements aligned across held-out datasets, or are they
   driven by different domains?
6. Separate conclusions directly supported by experiments from plausible but
   unverified explanations involving sample size, domain shift, loss weights,
   representation choice, and inner-fold selection.
7. State the strongest defensible scientific conclusion and the minimum next
   experiment that could distinguish an observability bottleneck from a weak
   transfer objective.

Do not reinterpret stopped P3-P7 phases as executed results. Do not infer that
reproducibility validation makes a failed scientific gate pass.
