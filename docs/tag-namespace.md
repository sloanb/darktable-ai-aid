# Tag namespace

`dt-aid` owns two top-level tag roots: `people` and `auto`. Everything else in your darktable library is considered user-owned and is never modified, renamed, or removed.

## Roots

| Root | Who writes it | Purpose |
|---|---|---|
| `people` | `dt-aid` | Named people and unknown-face clusters |
| `auto` | `dt-aid` | Algorithmic tags and provenance |
| *(anything else)* | User | Untouched |

## Patterns

| Pattern | Example | Source |
|---|---|---|
| `people\|<name>` | `people\|parker` | Face matches a named reference |
| `people\|unknown\|cluster-<NNN>` | `people\|unknown\|cluster-007` | Unmatched face assigned to an HDBSCAN cluster |
| `auto\|object\|<label>` | `auto\|object\|dog` | CLIP object detection |
| `auto\|scene\|<label>` | `auto\|scene\|beach` | CLIP scene classification |
| `auto\|attr\|<label>` | `auto\|attr\|sunset` | CLIP attribute classification |
| `auto\|_meta\|model-<kind>-<version>` | `auto\|_meta\|model-faces-buffalo-l-v1` | Provenance — which model produced the tags |

## Slug rules

Labels are run through `dt_aid.core.tagging.slug()`:

1. Unicode NFKD normalize
2. Strip non-ASCII
3. Lowercase
4. Spaces and underscores → `-`
5. Remove everything outside `[a-z0-9-]`
6. Strip leading/trailing `-`

Examples:
- `"Alice Smith"` → `alice-smith`
- `"Dining Table"` → `dining-table`
- `"café"` → `cafe`

Cluster IDs are zero-padded to three digits (`cluster-007`) so alphabetical sorting matches numerical sorting up to 999 clusters.

## Merge semantics

When writing XMP, `merge_managed(existing_tags, new_managed_tags)`:

1. Keeps every existing tag whose namespace is **not** `people|*` or `auto|*`.
2. Drops every existing `people|*` and `auto|*` tag (we own that namespace).
3. Adds all `new_managed_tags` (de-duplicated).

### Per-detector carry-over (idempotency)

`merge_managed` is category-agnostic — it drops *all* managed tags and re-adds whatever the caller passes. The pipeline runs two independently-versioned detectors (faces and elements), and either may be skipped for an image on a given scan because that detector's model version is already recorded in `state.db`. To keep the skipped detector's tags intact, `pipeline.scan()` carries them over from the existing XMP before writing:

- If `run_faces == False`, every existing tag matching `tagging.is_face_tag()` (i.e. `people|*` and `auto|_meta|model-faces-*`) is re-appended to the merged set.
- If `run_elements == False`, every existing tag matching `tagging.is_elements_tag()` (i.e. `auto|object|*`, `auto|scene|*`, `auto|attr|*`, `auto|_meta|model-elements-*`) is re-appended.

Without this, running `dt-aid scan --elements` on a library previously tagged by `scan --faces` would silently wipe every `people|<name>` tag, because the new managed tag set for that write contains only `auto|*` entries.

This means if you remove a person's reference and re-scan, the stale `people|<name>` tag is not automatically cleared. Promoting a cluster to a person is the forward path; a future `dt-aid prune` command may address the reverse.

## Hierarchy expansion in XMP

`lr:hierarchicalSubject` expands each leaf tag to include all ancestors. Writing `people|parker` to an image produces three entries:

```
people
people|parker
```

This is how darktable's tag panel shows the full tree. `dc:subject` gets only the leaf labels (e.g. `parker`) which is how other tools (Lightroom, digiKam) display flat keywords.

## Rationale for two roots

- **`people`** is the most commonly-consulted tag type and deserves a short prefix.
- **`auto`** makes bulk cleanup trivial: deleting the `auto` tag in darktable removes every algorithmic tag in one operation. `_meta` under `auto` keeps provenance out of the way while still being sortable and deletable with the rest.
- Keeping unknown-face clusters under `people|unknown|*` (not `auto|*`) reflects that they're placeholders for eventual user labels, not algorithmic output per se.
