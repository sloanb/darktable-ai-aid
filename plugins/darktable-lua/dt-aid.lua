--[[
  dt-aid — darktable Lua plugin for dt-aid face tagging.

  Phase 1: Scan / Cluster / Rematch buttons, "Tag selected as..." and
  "Promote cluster to..." actions.

  Phase 2: Cluster review panel — pick a cluster, auto-filter the grid,
  promote/reject in one click.

  Install:
    cp dt-aid.lua ~/.config/darktable/lua/contrib/
    Add  require "contrib/dt-aid"  to ~/.config/darktable/luarc
  Then set the dt-aid binary path in Preferences → Lua options.
]]

local dt = require "darktable"
local du = require "lib/dtutils"

du.check_min_api_version("7.0.0", "dt-aid")

local PS = dt.configuration.running_os == "windows" and "\\" or "/"

---------------------------------------------------------------------------
-- Minimal JSON decoder (handles strings, numbers, bools, null, arrays,
-- objects — enough for dt-aid's `--json` output, nothing more).
---------------------------------------------------------------------------
local json = {}
do
  local function skip_ws(s, i)
    while i <= #s and s:sub(i, i):match("[%s]") do i = i + 1 end
    return i
  end
  local parse_value
  local function parse_string(s, i)
    assert(s:sub(i, i) == '"', "expected string at " .. i)
    i = i + 1
    local out = {}
    while i <= #s do
      local c = s:sub(i, i)
      if c == '"' then return table.concat(out), i + 1 end
      if c == "\\" then
        local nxt = s:sub(i + 1, i + 1)
        local map = {['"'] = '"', ["\\"] = "\\", ["/"] = "/", n = "\n", t = "\t", r = "\r", b = "\b", f = "\f"}
        out[#out + 1] = map[nxt] or nxt
        i = i + 2
      else
        out[#out + 1] = c
        i = i + 1
      end
    end
    error("unterminated string")
  end
  local function parse_number(s, i)
    local j = i
    while j <= #s and s:sub(j, j):match("[%-%d%.eE%+]") do j = j + 1 end
    return tonumber(s:sub(i, j - 1)), j
  end
  local function parse_array(s, i)
    assert(s:sub(i, i) == "[", "expected [")
    i = i + 1
    local out, n = {}, 0
    i = skip_ws(s, i)
    if s:sub(i, i) == "]" then return out, i + 1 end
    while true do
      local v
      v, i = parse_value(s, skip_ws(s, i))
      n = n + 1; out[n] = v
      i = skip_ws(s, i)
      local c = s:sub(i, i)
      if c == "]" then return out, i + 1 end
      assert(c == ",", "expected , or ]")
      i = i + 1
    end
  end
  local function parse_object(s, i)
    assert(s:sub(i, i) == "{", "expected {")
    i = i + 1
    local out = {}
    i = skip_ws(s, i)
    if s:sub(i, i) == "}" then return out, i + 1 end
    while true do
      i = skip_ws(s, i)
      local k
      k, i = parse_string(s, i)
      i = skip_ws(s, i)
      assert(s:sub(i, i) == ":", "expected :")
      i = skip_ws(s, i + 1)
      local v
      v, i = parse_value(s, i)
      out[k] = v
      i = skip_ws(s, i)
      local c = s:sub(i, i)
      if c == "}" then return out, i + 1 end
      assert(c == ",", "expected , or }")
      i = i + 1
    end
  end
  parse_value = function(s, i)
    i = skip_ws(s, i)
    local c = s:sub(i, i)
    if c == "{" then return parse_object(s, i) end
    if c == "[" then return parse_array(s, i) end
    if c == '"' then return parse_string(s, i) end
    if c == "t" then return true, i + 4 end
    if c == "f" then return false, i + 5 end
    if c == "n" then return nil, i + 4 end
    return parse_number(s, i)
  end
  function json.decode(s)
    local v, _ = parse_value(s, 1)
    return v
  end
end

---------------------------------------------------------------------------
-- Preferences: dt-aid binary path and device
---------------------------------------------------------------------------
dt.preferences.register("dt-aid", "binary_path", "string",
  "dt-aid: binary path",
  "Path to the dt-aid executable. Leave as 'dt-aid' to use $PATH.",
  "dt-aid")

dt.preferences.register("dt-aid", "device", "enum",
  "dt-aid: inference device",
  "auto picks CUDA if available, else CPU.",
  "auto", "auto", "cpu", "cuda")

local function dtaid_bin()
  local p = dt.preferences.read("dt-aid", "binary_path", "string")
  if not p or p == "" then return "dt-aid" end
  return p
end

local function dtaid_device_flag()
  local d = dt.preferences.read("dt-aid", "device", "enum")
  if d and d ~= "" then return " --device " .. d end
  return ""
end

---------------------------------------------------------------------------
-- Shell helpers
---------------------------------------------------------------------------
local function shell_quote(s)
  return "'" .. tostring(s):gsub("'", "'\\''") .. "'"
end

local function run_capture(cmd)
  local f = io.popen(cmd .. " 2>&1")
  if not f then return nil, "failed to spawn command" end
  local out = f:read("*a") or ""
  local ok, why, code = f:close()
  return out, ok and nil or (why or "error"), code
end

local function run_fire_and_forget(cmd, title)
  dt.print(title .. " running…")
  local out, err = run_capture(cmd)
  if err then
    dt.print(title .. " failed: " .. tostring(err))
    dt.print_log(out)
    return false, out
  end
  dt.print(title .. " done.")
  dt.print_log(out)
  return true, out
end

---------------------------------------------------------------------------
-- Data fetchers
---------------------------------------------------------------------------
local function fetch_list()
  local cmd = dtaid_bin() .. " faces list"
  local out, err = run_capture(cmd)
  if err then
    dt.print("dt-aid faces list failed: " .. tostring(err))
    return nil
  end
  -- Strip any stderr-style lines before the JSON object.
  local json_start = out:find("{")
  if not json_start then return nil end
  local ok, decoded = pcall(json.decode, out:sub(json_start))
  if not ok then
    dt.print("dt-aid faces list: JSON parse error")
    return nil
  end
  return decoded
end

local function person_names(data)
  local out = {}
  if data and data.references then
    for _, r in ipairs(data.references) do out[#out + 1] = r.name end
  end
  table.sort(out)
  return out
end

---------------------------------------------------------------------------
-- UI: name chooser
---------------------------------------------------------------------------
-- Two-field picker: combobox of known names OR text entry for a new one.
local name_combo = dt.new_widget("combobox") {
  label = "existing person",
  "",  -- empty first option
}

local name_entry = dt.new_widget("entry") {
  placeholder = "or type a new name",
  text = "",
}

local function chosen_name()
  local t = name_entry.text
  if t and t ~= "" then return t end
  local selected = name_combo.value
  if selected and selected ~= "" then return selected end
  return nil
end

local function refresh_name_combo()
  local data = fetch_list()
  local names = person_names(data)
  -- reset items
  while #name_combo > 0 do name_combo[#name_combo] = nil end
  name_combo[1] = ""  -- blank selector
  for i, n in ipairs(names) do name_combo[i + 1] = n end
end

---------------------------------------------------------------------------
-- Phase 1 actions
---------------------------------------------------------------------------
local function on_scan_clicked()
  local cmd = dtaid_bin() .. " scan --faces" .. dtaid_device_flag()
  run_fire_and_forget(cmd, "dt-aid scan")
end

local function on_cluster_clicked()
  local cmd = dtaid_bin() .. " faces cluster"
  run_fire_and_forget(cmd, "dt-aid cluster")
  refresh_name_combo()
end

local function on_rematch_clicked()
  local cmd = dtaid_bin() .. " faces rematch"
  run_fire_and_forget(cmd, "dt-aid rematch")
end

local function on_tag_selected_clicked()
  local name = chosen_name()
  if not name then
    dt.print("dt-aid: pick an existing name or type a new one first")
    return
  end
  local selected = dt.gui.selection()
  if #selected == 0 then
    dt.print("dt-aid: select one or more images in lighttable first")
    return
  end
  local ok_count, fail_count = 0, 0
  for _, img in ipairs(selected) do
    local path = img.path .. PS .. img.filename
    local cmd = dtaid_bin() .. " faces add-image --json" .. dtaid_device_flag()
      .. " " .. shell_quote(path) .. " " .. shell_quote(name)
    local out, err = run_capture(cmd)
    if err then
      fail_count = fail_count + 1
    else
      ok_count = ok_count + 1
    end
  end
  dt.print(string.format(
    "dt-aid: tagged %d image(s) as '%s' (%d failed). Rematch queued.",
    ok_count, name, fail_count))
  -- reference library grew — re-sweep the rest of the library
  run_fire_and_forget(dtaid_bin() .. " faces rematch", "dt-aid rematch")
  refresh_name_combo()
end

local function cluster_id_from_tags(image)
  -- find an attached tag like `people|unknown|cluster-NNN` and return NNN
  for _, tag in ipairs(dt.tags.get_tags(image)) do
    local n = tostring(tag.name):match("^people|unknown|cluster%-(%d+)$")
    if n then return tonumber(n) end
  end
  return nil
end

local function on_promote_cluster_clicked()
  local name = chosen_name()
  if not name then
    dt.print("dt-aid: pick/type a name for the cluster first")
    return
  end
  local selected = dt.gui.selection()
  if #selected == 0 then
    dt.print("dt-aid: select an image from the cluster first")
    return
  end
  local cluster_id = cluster_id_from_tags(selected[1])
  if not cluster_id then
    dt.print("dt-aid: selected image has no people|unknown|cluster-NNN tag")
    return
  end
  local cmd = dtaid_bin() .. " faces relabel "
    .. tostring(cluster_id) .. " " .. shell_quote(name)
  run_fire_and_forget(cmd, "dt-aid relabel cluster " .. cluster_id)
  refresh_name_combo()
end

---------------------------------------------------------------------------
-- Phase 2: Cluster review panel
---------------------------------------------------------------------------
local cluster_combo = dt.new_widget("combobox") {
  label = "clusters (largest first)",
  "",
}
local cluster_meta = {}  -- index into cluster_combo -> {cluster_id, face_count, image_count, label}

local function refresh_clusters()
  local data = fetch_list()
  while #cluster_combo > 0 do cluster_combo[#cluster_combo] = nil end
  cluster_meta = {}
  cluster_combo[1] = "select a cluster…"
  if not data or not data.clusters then return end
  for _, c in ipairs(data.clusters) do
    local idx = #cluster_combo + 1
    local label = c.label and (" → " .. c.label) or ""
    cluster_combo[idx] = string.format(
      "cluster %d — %d faces / %d images%s",
      c.cluster_id, c.face_count, c.image_count, label)
    cluster_meta[idx] = c
  end
  dt.print(string.format("dt-aid: %d clusters loaded", #data.clusters))
end

local function selected_cluster()
  for i = 1, #cluster_combo do
    if cluster_combo[i] == cluster_combo.value then
      return cluster_meta[i]
    end
  end
  return nil
end

local function on_filter_to_cluster_clicked()
  local c = selected_cluster()
  if not c then
    dt.print("dt-aid: pick a cluster from the dropdown first")
    return
  end
  -- Use the collect module to filter to this tag.
  local tag_name = string.format("people|unknown|cluster-%03d", c.cluster_id)
  local collection = {
    { operator = 0, value = tag_name,
      data = "", client_data = "",
      item = "DT_COLLECTION_PROP_TAG" }
  }
  -- darktable Lua API: set_collection takes a list of collection rules
  local ok, err = pcall(function()
    dt.gui.libs.collect.filter(collection)
  end)
  if not ok then
    -- Fallback: just tell the user what to type in the collect module.
    dt.print("dt-aid: filter manually to tag '" .. tag_name .. "'")
  end
end

local function on_promote_current_cluster_clicked()
  local c = selected_cluster()
  if not c then
    dt.print("dt-aid: pick a cluster first")
    return
  end
  local name = chosen_name()
  if not name then
    dt.print("dt-aid: pick/type a name for the cluster first")
    return
  end
  local cmd = dtaid_bin() .. " faces relabel "
    .. tostring(c.cluster_id) .. " " .. shell_quote(name)
  run_fire_and_forget(cmd, "dt-aid relabel cluster " .. c.cluster_id)
  refresh_clusters()
  refresh_name_combo()
end

---------------------------------------------------------------------------
-- Assemble the lib widget
---------------------------------------------------------------------------
local btn_scan = dt.new_widget("button") {
  label = "scan (faces)",
  tooltip = "Detect faces on new images; auto-match against known references.",
  clicked_callback = on_scan_clicked,
}

local btn_cluster = dt.new_widget("button") {
  label = "cluster",
  tooltip = "Run HDBSCAN over unmatched faces; writes people|unknown|cluster-NNN tags.",
  clicked_callback = on_cluster_clicked,
}

local btn_rematch = dt.new_widget("button") {
  label = "rematch",
  tooltip = "Re-match all unlabeled faces against the current reference library.",
  clicked_callback = on_rematch_clicked,
}

local btn_refresh = dt.new_widget("button") {
  label = "refresh names + clusters",
  clicked_callback = function()
    refresh_name_combo()
    refresh_clusters()
  end,
}

local btn_tag_selected = dt.new_widget("button") {
  label = "tag selected as…",
  tooltip = "For each selected image, add its largest face as a reference for the chosen name.",
  clicked_callback = on_tag_selected_clicked,
}

local btn_promote_cluster = dt.new_widget("button") {
  label = "promote cluster of selected",
  tooltip = "Read the cluster tag from the selected image and promote it to the chosen name.",
  clicked_callback = on_promote_cluster_clicked,
}

local btn_filter_cluster = dt.new_widget("button") {
  label = "filter grid to cluster",
  clicked_callback = on_filter_to_cluster_clicked,
}

local btn_promote_current = dt.new_widget("button") {
  label = "promote selected cluster",
  clicked_callback = on_promote_current_cluster_clicked,
}

local lib_widget = dt.new_widget("box") {
  orientation = "vertical",
  dt.new_widget("section_label") { label = "scan & match" },
  btn_scan, btn_cluster, btn_rematch,
  dt.new_widget("section_label") { label = "name" },
  name_combo, name_entry,
  dt.new_widget("section_label") { label = "label selected images" },
  btn_tag_selected, btn_promote_cluster,
  dt.new_widget("section_label") { label = "cluster review" },
  cluster_combo, btn_filter_cluster, btn_promote_current,
  dt.new_widget("separator") {},
  btn_refresh,
}

dt.register_lib(
  "dt_aid",              -- plugin id
  "dt-aid",              -- label
  true,                  -- expandable
  false,                 -- resetable
  { [dt.gui.views.lighttable] = { "DT_UI_CONTAINER_PANEL_RIGHT_CENTER", 700 } },
  lib_widget,
  nil, nil
)

-- Populate name/cluster dropdowns on startup.
dt.control.sleep(50)  -- let the lib finish registering
pcall(refresh_name_combo)
pcall(refresh_clusters)
