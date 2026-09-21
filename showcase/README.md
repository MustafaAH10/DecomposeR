# Research viewer and videos

The public integration is at https://mustafaah10.github.io/DecomposeR/#research-viewer.
It replaces the project's two static plan figures, without changing the paper's
surrounding content. The static viewer does not require a server, model, API key
or ChatGPT login. All data is already generated.

`videos/` contains the three README examples; `previews/` contains their animated
GIF previews. All 20 validated demonstrations and full-resolution MP4s are in
`viewer/public/`. Original model text is preserved; animation timing is compressed.
Same-wave report animations illustrate the answerer's execution grouping, not
separate concurrent model calls or captured token timing.

## Rebuild the embedded viewer

Use Node 22 from `showcase/viewer`:

```sh
npm ci
npm run build
```

The complete portable output is `embed-dist/`. Copy its contents into the public
website's `DecomposeR/assets/viewer/` directory. Relative asset URLs allow the same
bundle to work below the GitHub Pages project path. The parent page must retain
`assets/viewer-embed.js` to receive origin-checked iframe height updates.

`app/page.tsx` supplies the player and centered node viewer; `lib/scene.mjs` supplies
the graph, incremental arrows, camera zooms and branch-writing animation.
The iframe embeds only the player portion, without the standalone site's header.
Geist's SIL Open Font License is retained in `embed/fonts/OFL.txt`.

## Run provenance

Generated with the public merged DecomposeR checkpoint at revision
`503e2bb12418b22259c0196700d3b0587ea00d42`, using the submitted prompts, Jina search
and Serper fetch. Initial generation used seed 17; demo04 and demo13 passed after
regeneration with seed 29 because their original outputs referenced unknown
citation IDs. Original and retry records are preserved in the local run backup.
The exporter checks DAG validity, complete node outputs, and citation IDs, not
whether every factual claim is correct. These are demonstrations, not new
benchmark results.
