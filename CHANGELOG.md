# Changelog

## Unreleased

### Added
- `/v1/embeddings` accepts an optional `prompt_name` (`query` | `document`) that selects the `Query: ` / `Document: ` input prefix of Jina v5 models, so `task=retrieval` can embed queries correctly. Omitting it keeps the previous behavior (model default `document`).
- `prompt_name` is part of the embedding cache key, the batcher group key, the GPU worker request, the `MAX_LENGTH` token counting, and the request log.
