# dt-aid user guide

This guide walks through `dt-aid` the way you'll actually use it day to day — what to run, in what order, and why. You don't need to understand the ML internals.

Use the CLI commands shown here, or do the same thing from darktable's lighttable using the **dt-aid** panel (see `plugins/darktable-lua/INSTALL.md`). The button labels mirror the commands.

---

## The big picture

`dt-aid` reads your darktable library, finds faces and (optionally) visual elements in your photos, and writes tags back to each photo's XMP sidecar so darktable picks them up. It never writes to `library.db`.

It tags into two hierarchies only:

- `people|<name>` — a face that matched a known person
- `people|unknown|cluster-<NNN>` — a face that didn't match anyone known, grouped with visually similar other unknowns
- `auto|object|*`, `auto|scene|*`, `auto|attr|*` — visual-element tags (only if you opt in with `--elements`)

Anything else in your existing tags is left alone.

The central idea is a loop: **scan → cluster → relabel → rematch**. Scans find faces. Clustering buckets the unknowns. Relabeling turns a bucket into a named person. Rematching propagates the new person across the rest of your library.

---

## Rules that never change

1. **Close darktable before running anything.** `dt-aid` refuses to start if darktable is open. Concurrent writers can corrupt `library.db`.
2. **You trigger everything.** Nothing is automatic on a schedule. Pick a time, close darktable, run the commands, reopen darktable.
3. **You will re-run things.** The whole system is idempotent — scans skip already-processed images, rematch only touches unlabeled faces. Running a step a second time is safe.

---

## Workflow 1 — your very first scan

Do this once, the first time you use `dt-aid`.

**When:** you've just installed the tool and want it to see your whole library for the first time.

**Steps:**

1. **(Optional) Prepare a few known faces.** Skip this if you don't care about names yet; you can label people later from clusters (Workflow 4).

   Make a directory like:
   ```
   known-faces/
   ├── alice/
   │   ├── 1.jpg
   │   └── 2.jpg
   └── bob/
       └── 1.jpg
   ```
   Then build references:
   ```
   dt-aid faces build-refs ./known-faces
   ```
   A few clear, front-facing shots per person is enough to start.

2. **Dry-run a scan** so you can see what would happen before writing anything:
   ```
   dt-aid scan --faces --dry-run
   ```
   Add `--elements` if you also want visual-element tags. This can take a while on a large library; the first run also downloads the face model (~300 MB).

3. **Real scan.** When you're happy:
   ```
   dt-aid scan --faces --elements
   ```

4. **Cluster the unknowns** — a scan detects faces but does **not** group the unmatched ones; you do that in a separate, fast step:
   ```
   dt-aid faces cluster
   ```
   You'll see `people|unknown|cluster-NNN` tags appear.

5. **Reload tags in darktable.** Either set **Preferences → Storage → "look for updated XMP files on startup"** once and restart darktable, or select images in lighttable and right-click **→ selected image(s) → reload the selected XMP files**.

At this point everybody who matched your references is tagged as `people|<name>`, everyone else is in an `unknown` cluster, and the library is ready for you to start naming clusters (Workflow 4).

---

## Workflow 2 — adding new images to a tagged library

**When:** you've imported new photos into darktable after a previous `dt-aid` run.

Think of this as "run the same first-scan steps, but only the new stuff gets processed." Previously-seen images are skipped automatically.

1. Close darktable.
2. `dt-aid scan --faces` (add `--elements` if you use them). Only new images do real work.
3. `dt-aid faces cluster` — groups any newly-detected unknowns. Existing clusters are not disturbed.
4. Optional: `dt-aid faces rematch` — if you've added any references since your last scan, this picks up any new faces that now match them.
5. Reload tags in darktable (see step 5 of Workflow 1).

**Tip — scan a subset.** If you only imported one folder, limit the scan:
```
dt-aid scan --faces --path /photos/2026/spring
```

**Tip — force re-processing.** If you deliberately want to re-detect faces on already-seen images (e.g. you think something was missed), use `--force`. Otherwise don't — it's slower and not needed for normal use.

---

## Workflow 3 — adding a known person (two ways)

You can teach `dt-aid` who someone is either in bulk (a directory of their photos) or one photo at a time.

### 3a. From a directory of labeled photos

**When:** you already have a small set of clean portrait photos of the person, outside your darktable library.

```
dt-aid faces build-refs ./known-faces
```

> **Heads-up — `build-refs` overwrites.** If `alice.npy` already exists, running `build-refs` again with a folder named `alice/` **replaces** it. Use this when you want to start that person's reference from scratch. To add more photos without losing the old ones, use Workflow 3b instead.

After `build-refs`, run `dt-aid faces rematch` to propagate the new references across every already-detected face.

### 3b. From one image in your library

**When:** you're scrolling through lighttable and spot a photo where a specific person is clearly visible. You want to teach dt-aid "that face = Alice" without preparing a folder.

```
dt-aid faces add-image /path/to/photo.jpg alice
```

This detects faces in the photo, picks the largest one, and **appends** its embedding to `alice`'s references (never overwrites). If the image was already in the embedding store, its parquet row and XMP are updated on the spot.

If the largest face isn't the right one, pick it explicitly:
```
dt-aid faces add-image /path/to/photo.jpg alice --face-index 1
```

Follow with `dt-aid faces rematch` to propagate.

**In the Lua plugin:** select one or more images in lighttable, type or choose a name, click **tag selected as…**. It calls `add-image` per image and then runs rematch.

### When to use which

- **build-refs:** first time setting up a person, or you want a clean slate.
- **add-image:** enriching an existing person with new examples, or naming someone you found in a single photo. Safer — it never loses vectors.

---

## Workflow 4 — reviewing clusters and naming people

**When:** after any scan that left new unmatched faces. This is where most of your labeling work happens.

1. **List clusters, biggest first.**
   ```
   dt-aid faces list --min-size 5
   ```
   Prints JSON with each cluster's ID, face count, and distinct-image count. Big clusters are the most efficient to label — one `relabel` call names many photos at once.

2. **Find out who's in a cluster.** Either look at the member images directly (the Lua plugin has a **filter grid to cluster** button) or sample a few paths from the parquet. In the CLI, a quick way is:
   ```
   python -c "import pyarrow.parquet as pq; t = pq.read_table('~/.local/share/dt-aid/faces/embeddings.meta.parquet'.replace('~', '$HOME')); print(t.filter(t['cluster_id'].cast('int64') == 7).to_pandas()['image_path'].head(10))"
   ```
   (The Lua plugin is much nicer for this.)

3. **Promote the cluster to a name.**
   ```
   dt-aid faces relabel 7 alice
   ```
   This:
   - saves every embedding in cluster 7 as `alice`'s reference (overwriting any previous `alice.npy`),
   - flips those face rows from `cluster-7` to `alice`,
   - rewrites each affected XMP so darktable sees `people|alice` instead of `people|unknown|cluster-007`.

4. **Propagate.** Run `dt-aid faces rematch` — any *other* unlabeled faces across your whole library that look like `alice` now also get tagged.

5. **Move to the next cluster.** Repeat until the big clusters are all named.

**Order matters:** always relabel the biggest cluster first, then rematch, then look at what's left. Rematching can empty out smaller clusters "for free" because faces that were in small clusters often get matched by the freshly-saved reference.

> **Heads-up — relabel overwrites too.** If you previously built a reference for `alice` via `build-refs`, calling `relabel <id> alice` replaces those vectors with the cluster's. If you want to keep both, rename one before promoting (there's currently no merge flag; this is a known limitation).

---

## Workflow 5 — enriching a person to catch edge cases

**When:** after running rematch, you notice Alice is mostly tagged but specific kinds of photos are missed — bad lighting, a side profile, a hat, a different age.

Pick one of those missed photos and teach it:

```
dt-aid faces add-image /path/to/side-profile.jpg alice
dt-aid faces rematch
```

`add-image` **appends** to `alice.npy` — old references are preserved, the new example expands coverage. Adding 3–5 well-chosen "hard" examples usually eliminates a stubborn miss-pattern.

Repeat for each person who needs it.

---

## Workflow 6 — fixing a wrong tag

**When:** dt-aid tagged a face with the wrong person. There's no single "untag" command — you pick one of three fixes depending on how widespread the problem is.

### 6a. One or two images, and you know who the face actually is

```
dt-aid faces add-image /path/to/photo.jpg <correct-name> [--face-index N]
```

`add-image` finds the wrongly-labeled face in the cache (by embedding similarity > 0.95), **overwrites** its label to `<correct-name>`, appends its vector to the correct person's reference, and rewrites the XMP. The wrong tag is gone immediately. Use `--face-index` if the largest face in the photo isn't the one you want to re-label.

Follow with `dt-aid faces rematch` — it won't revisit this row (it's now labeled), but the richer reference may clean up other false negatives across the library.

### 6b. One image, the face isn't anyone you care about (stranger, background)

Open the image in darktable and remove the wrong `people|<name>` tag from the tagging panel. The parquet still records the wrong label internally, but nothing rewrites the XMP unless you later run `dt-aid scan --force` on that image — at which point the bad reference would put the tag back. If you think that risk is real, fix the reference (6c) instead.

### 6c. Many images wrongly tagged as `<name>` — the reference itself is contaminated

This usually means `build-refs` picked up a stray face, or a cluster that you `relabel`ed had the wrong person mixed in. The fix is to reset that person's reference and re-detect the affected images:

1. **Find the affected images.** The Lua plugin can filter the grid by a name; from the CLI you can query the parquet. A quick one-liner:
   ```
   python -c "import pyarrow.parquet as pq; t = pq.read_table('$HOME/.local/share/dt-aid/faces/embeddings.meta.parquet'); [print(p) for p in t.filter(t['label'].cast('string') == '<name>')['image_path'].to_pylist()]"
   ```

2. **Delete the bad reference file:**
   ```
   rm ~/.local/share/dt-aid/faces/references/<name>.npy
   ```

3. **Rebuild from a clean folder:**
   ```
   dt-aid faces build-refs ./clean-known-faces
   ```
   Only include photos you're confident are `<name>`.

4. **Re-run scan with `--force` on the affected images:**
   ```
   dt-aid scan --faces --force --path /photos/affected
   ```
   `--force` re-detects the faces and re-matches them against the clean reference; the write step drops the stale `people|<name>` tag when it rewrites the XMP. Plain `rematch` will *not* help here — it only touches faces that are still unlabeled.

> **Known gap.** There is no `dt-aid faces unlabel` command, and `rematch` deliberately skips already-labeled faces. That means there's no lightweight "just re-check everyone's labels" pass yet — scope-3 corrections currently cost a `--force` rescan. Tracked as deferred work.

---

## Workflow 7 — opting in to element tagging (objects / scenes / attributes)

**When:** you want tags like `auto|object|dog`, `auto|scene|beach`, `auto|attr|sunset`. This uses a separate model (CLIP) and needs the `[elements]` install extra.

Install once:
```
uv pip install -e '.[elements]'
```

Run on your first scan or any later scan:
```
dt-aid scan --elements                 # elements only
dt-aid scan --faces --elements         # both
```

There is no "rematch" or "cluster" equivalent for elements — they're deterministic per image. If you want to re-tag with a newer label list or a changed threshold, use `--force` to re-process images.

Adjust sensitivity with `DT_AID_ELEMENTS_THRESHOLD=0.25` (higher = fewer tags, more confident).

---

## The recommended cycle

Put it all together:

```
# once, upfront (optional)
dt-aid faces build-refs ./known-faces

# every time you import new photos
dt-aid scan --faces            # add --elements if you use them
dt-aid faces cluster
dt-aid faces rematch           # if you've added any references since last time

# labeling session
dt-aid faces list --min-size 5
dt-aid faces relabel <biggest-cluster-id> <name>
dt-aid faces rematch
# ... repeat relabel + rematch until satisfied ...

# for stubborn faces
dt-aid faces add-image <photo> <name>
dt-aid faces rematch

# tell darktable to pick up the new tags
# (preference toggle, or right-click → reload the selected XMP files)
```

A healthy rhythm is: scan + cluster after each import, labeling session whenever you feel like it, rematch after every reference change.

---

## FAQ

**How do I see the new tags in darktable?**
Either turn on **Preferences → Storage → "look for updated XMP files on startup"** once and restart, or in lighttable select the affected images → right-click → **selected image(s) → reload the selected XMP files**. Managed tags appear under `people|*` and `auto|*` in the tagging panel.

**I named a cluster wrong. How do I undo it?**
Re-run `relabel` with the correct name — it overwrites the previous label. The face rows, references, and XMPs all move to the new name. The old (incorrect) reference file is not deleted automatically; remove it manually from `~/.local/share/dt-aid/faces/references/`.

**A face got tagged as the wrong person.**
See Workflow 6 — the fix depends on scope. One image where you know the correct identity: `dt-aid faces add-image <image> <correct-name>`. One stranger in a background: remove the tag in darktable. Many wrong images with the same name: the reference is contaminated — delete `references/<name>.npy`, rebuild from a clean folder, and `scan --force` the affected images.

**Why is rematch so much faster than scan?**
`scan` has to load every image and run the face detector on it. `rematch` reuses the face embeddings you already cached — it's pure vector math against the reference library. No GPU needed.

**Why do I need to cluster at all? Why doesn't scan just do it?**
Clustering depends on the full population of unmatched faces; running it incrementally during a scan produces unstable cluster IDs that change every run. Making it a separate, manual step means your cluster numbering is stable until *you* decide to re-cluster.

**Can I re-cluster?**
Yes — re-running `dt-aid faces cluster` re-assigns cluster IDs across all currently-unlabeled faces. Labeled faces (`cluster_id = -2`) are never touched. You'd do this if you've added a lot of new unknowns since the last cluster pass.

**My GPU isn't being used.**
Use `--device cuda` explicitly, or set `DT_AID_DEVICE=cuda`. Verify with:
```
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```
If `CUDAExecutionProvider` isn't listed, install the `[gpu]` extra (see `docs/development.md`).

**I want to start over.**
Delete `~/.local/share/dt-aid/` (or just the `faces/` subfolder) and re-scan. Your darktable library is untouched; existing XMP tags stay as-is until the next scan rewrites them.

---

## Command cheat sheet

| Command | When |
|---|---|
| `dt-aid scan --faces [--elements]` | After importing new photos into darktable. |
| `dt-aid faces build-refs <dir>` | First time setting up known people, or starting a person's reference over. |
| `dt-aid faces add-image <image> <name>` | Teaching one photo at a time; enriching an existing person; correcting a wrong tag to the real identity (Workflow 6a). |
| `dt-aid faces cluster` | After a scan that added unmatched faces. |
| `dt-aid faces list [--min-size N]` | Before a labeling session, to see which clusters to tackle. |
| `dt-aid faces relabel <id> <name>` | Naming a cluster. |
| `dt-aid faces rematch` | After *any* reference change (build-refs, add-image, relabel). |

If you're doing all of this from the darktable Lua plugin, the same order applies — the buttons have the same names.
