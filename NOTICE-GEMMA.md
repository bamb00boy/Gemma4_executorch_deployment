# Notice: Gemma 4 weights and derivatives

> **This file is a notice, not the license text.** The authoritative documents
> are published by Google at the URLs below. Read them yourself before
> redistributing derivative weights.

## License for Gemma 4 (and weights derived from it)

**Gemma 4 is released under the Apache License 2.0** by Google DeepMind. Source of truth:

- **Apache 2.0 text (as Google publishes for Gemma 4):** https://ai.google.dev/gemma/apache_2
- **Gemma 4 model card:** https://ai.google.dev/gemma/docs/core/model_card_4
- **Gemma 4 GitHub repository:** https://github.com/google-deepmind/gemma

## Additional policies that apply on top of Apache 2.0

Even though the license itself is Apache 2.0, Google applies two policies to all Gemma models (including Gemma 4):

- **Gemma Prohibited Use Policy:** https://ai.google.dev/gemma/prohibited_use_policy
  Covers: rights violations, dangerous/illegal facilitation (CSAM, weapons, illegal drug synthesis, terrorism, etc.), harmful/hateful/discriminatory content, deceptive content (false attribution, impersonation, defamation), and sexually explicit content. Read the policy for the full list.
- **Gemma Intended Use Statement** (linked from the same Gemma 4 page).

These policies are written as *separate* documents from the Apache 2.0 license. By using or redistributing Gemma 4 (or any model derivative), you accept these policies as well.

## What this means for the `.pte` produced by this repo

The `.pte` file produced by `scripts/04_quantize.py` + `scripts/05_lower.py` is a **derivative work** of Gemma 4. Under Apache 2.0 + the Gemma policies:

1. **You may** redistribute it freely, including commercially.
2. **You must** include a copy of the Apache 2.0 license (this is the standard Apache requirement; it's in this repo and applies to the weights).
3. **You must** retain copyright notices and clearly mark modifications (Apache 2.0 §4).
4. **You must NOT** use it for purposes prohibited by the Gemma Prohibited Use Policy. If you redistribute, pass that obligation through to downstream users (e.g., link to the policy in your README / model card).
5. **You may** publish on HuggingFace as a normal (non-gated) model — Apache 2.0 does not require recipient agreement before download. Gating is optional.

## Code vs weights

This repository has **two distinct licenses** at play:

| Component | License | File |
|---|---|---|
| Code in this repo (Python scripts, runners, documentation) | MIT | `LICENSE` |
| Gemma 4 weights and any derivatives (the `.pte`) | Apache 2.0 + Gemma policies | this file + upstream sources |

If you publish the `.pte` on HuggingFace or anywhere else, you only need to comply with the Apache 2.0 + Gemma policy requirements for the weights — the MIT code license is unrelated.

## Citing Gemma 4

As of writing (May 2026), the **Gemma 4 technical report has not been released**. Google DeepMind's GitHub README labels it *"Gemma 4 (Coming soon)"*. Until the official paper / BibTeX entry is published, cite Gemma 4 via the model card and launch blog:

```bibtex
@misc{gemma4_2026,
  title        = {Gemma 4},
  author       = {{Gemma Team and Google DeepMind}},
  year         = {2026},
  month        = {April},
  howpublished = {\url{https://ai.google.dev/gemma/docs/core/model_card_4}},
  note         = {Apache 2.0; technical report forthcoming},
}
```

> **Action when Google publishes the Gemma 4 paper:** replace the BibTeX above
> with the official entry from Google DeepMind's GitHub repo
> (https://github.com/google-deepmind/gemma) or the arXiv preprint.

## Where to verify all of the above

If anything in this file disagrees with Google's published documents, **Google's documents control**. Re-read them periodically; the URLs are stable:

- https://ai.google.dev/gemma — Gemma documentation home
- https://ai.google.dev/gemma/apache_2 — Apache 2.0 as Google publishes it for Gemma
- https://ai.google.dev/gemma/prohibited_use_policy — Prohibited Use Policy
- https://ai.google.dev/gemma/docs/core/model_card_4 — Gemma 4 model card
- https://github.com/google-deepmind/gemma — official DeepMind repository
