# How to Use This Plugin

This plugin can be used in a NOMAD Oasis installation.

## Add This Plugin to Your NOMAD installation

Read the [NOMAD plugin documentation](https://nomad-lab.eu/prod/v1/staging/docs/plugins/plugins.html#add-a-plugin-to-your-nomad) for all details on how to deploy the plugin on your NOMAD instance.

## Upload BOSS Results

Upload the BOSS restart file (`.rst`) together with the `boss.out` log from the
same run. The parser creates two entries:

- a raw-file entry for the `.rst` file itself, which only references
- a **BOSS Analysis** entry (an editable ELN), which holds the parsed data and
  the interactive H5Web visualization of the potential-energy-surface fit on
  the Overview tab.

## Parameter Names and Plot Labels

The axis labels of the H5Web plots come from the `parameter_names` field of the
BOSS Analysis entry. They can be set in three ways, in order of precedence:

1. **Edited in the ELN**: change `parameter_names` on the BOSS Analysis entry
   and save. The plot labels and titles update immediately; the fits are not
   recomputed.
2. **Set beforehand via a names file**: upload a file called
   `parameter_names.yml` (or `.yaml`) in the same directory as the `.rst`
   file. It is read whenever the entry has no names yet, e.g. on the first
   processing. The file may contain a plain list

    ```yaml
    - phi
    - psi
    ```

    or a mapping

    ```yaml
    parameter_names:
      - phi
      - psi
    ```

3. **Defaults**: without either of the above, the names default to
   `parameter_0`, `parameter_1`, ...

If the number of names does not match the number of parameters of the BOSS
run, the names are ignored with a warning (and replaced by defaults during the
initial parsing).

!!! note "Reprocessing does not reset names"
    Reprocessing the upload from scratch does **not** change previously set
    parameter names: the BOSS Analysis entry (`*.archive.json`) persists across
    reprocessing, and names stored there — whether edited in the ELN or loaded
    from `parameter_names.yml` — take precedence. This is intended behavior: a
    reprocess never silently undoes user edits. To start over with fresh
    names, delete the generated `*.archive.json` (or the upload) and process
    again.
