# dt-aid darktable Lua plugin — install

Adds a "dt-aid" panel to darktable's lighttable view so you can scan, cluster, rematch, and label people without leaving darktable. Requires the `dt-aid` CLI to be installed and working (see the top-level [README](../../README.md)).

## 1. Drop the plugin file in place

Linux / macOS:

```bash
mkdir -p ~/.config/darktable/lua/contrib
cp plugins/darktable-lua/dt-aid.lua ~/.config/darktable/lua/contrib/
```

## 2. Tell darktable to load it

Edit (or create) `~/.config/darktable/luarc` and append:

```
require "contrib/dt-aid"
```

## 3. Restart darktable

You'll see a new **"dt-aid"** module in the lighttable right-hand panel.

## 4. Set the binary path

**Preferences → Lua options → dt-aid: binary path** — set to the full path of your `dt-aid` executable. For a venv-installed build that's typically:

```
/home/sloanb/Projects/darktable-ai-aid/.venv/bin/dt-aid
```

Leave it as `dt-aid` if the binary is on darktable's `$PATH`.

**Preferences → Lua options → dt-aid: inference device** — `auto` / `cpu` / `cuda`. Default `auto`.

## What the panel does

| Button | What it runs |
|---|---|
| **scan (faces)** | `dt-aid scan --faces --device <pref>` — detects faces in any new images, auto-matches against known references. |
| **cluster** | `dt-aid faces cluster` — groups unmatched faces into `people\|unknown\|cluster-NNN` tags. |
| **rematch** | `dt-aid faces rematch` — re-runs the matcher against all unlabeled faces using the current reference library. |
| **tag selected as…** | For each selected image, teaches the chosen name from the image's largest face (`dt-aid faces add-image`), then runs rematch to propagate across the library. |
| **promote cluster of selected** | Reads the `people\|unknown\|cluster-NNN` tag from the first selected image and promotes the whole cluster to the chosen name (`dt-aid faces relabel`). |
| **filter grid to cluster** | Filters the lighttable grid to the selected cluster's images (phase 2 review workflow). |
| **promote selected cluster** | Same as above but acts on the cluster picked in the dropdown rather than the selected image. |
| **refresh names + clusters** | Re-reads `dt-aid faces list` output to repopulate the name and cluster dropdowns. |

## Name workflow

Two ways to pick a name for the `tag selected as…` or `promote cluster` buttons:

1. **Pick an existing person** from the "existing person" dropdown (auto-populated from `~/.local/share/dt-aid/faces/references/`).
2. **Type a new name** in the free-text entry — takes precedence if both are set.

After any relabel / add-image / cluster run, click **refresh names + clusters** to refresh the dropdowns.

## Darktable reload of XMP

The plugin writes XMP sidecars but doesn't force darktable to reload them. Either:

- Turn on **Preferences → Storage → "look for updated XMP files on startup"** once (recommended).
- Or right-click selected images → **selected image(s) → reload the selected XMP files**.

## Troubleshooting

**"dt-aid faces list failed":** the plugin couldn't run `dt-aid`. Check the binary path preference and that the binary is executable.

**Panel doesn't appear after restart:** check `~/.config/darktable/luarc` contains `require "contrib/dt-aid"` and the file is at `~/.config/darktable/lua/contrib/dt-aid.lua`. darktable writes Lua load errors to stderr — start darktable from a terminal and watch for them.

**Nothing happens when I click "scan":** the button is synchronous — the panel appears frozen until the scan completes. For large libraries (30k+ images), running the scan from a terminal is still faster to observe.
