# Frozen manuscript revision inputs

The two input originals retain their supplied bytes locally. `inputs/`, `original_extracted.md`, `original_media/` and `original_tables.json` are private local inputs: Git ignores them and the delivery inventory excludes their contents. They must not be staged or uploaded to GitHub. Source bindings retain only path/hash metadata for verification.

`original_extracted.md` is the Pandoc structural extraction used for deterministic editorial anchors; it does not replace the originals. `original_media` retains the seven original image objects. The execution packet files preserve task scope. New manuscript files, evidence maps and publication status live in the separate paper/artifact roots. Re-preparing text requires these private local files; building the supplied canonical sources does not. No research source or frozen result was edited.
