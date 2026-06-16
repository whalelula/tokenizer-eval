# Dataset Preparation

This project provides manifest builders for NSynth, Medley-solos-DB, and
MTG-Jamendo. A manifest is a CSV with at least `id` and `audio_path`, plus
dataset-specific metadata columns.

## Medley-solos-DB

Official source: Zenodo record `3464194`.

Metadata-only manifest:

```powershell
prepare-medley-solos-db `
  --root data/medley-solos-db `
  --manifest outputs/medley_solos_manifest.csv `
  --metadata-only
```

Download the official audio tarball as well:

```powershell
prepare-medley-solos-db `
  --root data/medley-solos-db `
  --manifest outputs/medley_solos_manifest.csv `
  --download-audio
```

Useful filters:

```powershell
prepare-medley-solos-db `
  --root data/medley-solos-db `
  --manifest outputs/medley_solos_test_guitar.csv `
  --subsets test `
  --instruments guitar `
  --max-per-instrument 200 `
  --download-audio
```

The downloader writes to `.part` files first and supports resumable retries.
If the archive extraction fails, rerun the command; the partial download is
kept for resume.

## MTG-Jamendo

Official source: `MTG/mtg-jamendo-dataset`.

The CLI downloads split metadata directly from the official GitHub repository.
Audio is distributed separately as large tar shards through MTG mirrors. The
default `audio-subset` is `autotagging_moodtheme`, and the default `audio-type`
is `audio-low`, which is much smaller than the full `raw_30s` audio.

Metadata-only manifest:

```powershell
prepare-mtg-jamendo `
  --root data/mtg-jamendo `
  --task autotagging_moodtheme `
  --split test `
  --split-id 0 `
  --manifest outputs/mtg_jamendo_moodtheme_test.csv `
  --metadata-only
```

Download selected audio shards and unpack them:

```powershell
prepare-mtg-jamendo `
  --root data/mtg-jamendo `
  --task autotagging_moodtheme `
  --split test `
  --audio-subset autotagging_moodtheme `
  --audio-type audio-low `
  --audio-shards 0 1 `
  --download-audio `
  --manifest outputs/mtg_jamendo_moodtheme_test.csv
```

Download all shards only when you have enough disk space:

```powershell
prepare-mtg-jamendo `
  --root data/mtg-jamendo `
  --audio-subset raw_30s `
  --audio-type audio-low `
  --download-audio `
  --metadata-only
```

Useful filters:

```powershell
prepare-mtg-jamendo `
  --root data/mtg-jamendo `
  --tags mood/theme---happy mood/theme---sad `
  --max-items 1000 `
  --metadata-only `
  --manifest outputs/mtg_jamendo_moodtheme_subset.csv
```

MTG-Jamendo tar files are checksum-verified against the official
`data/download/*_sha256_tars.txt` lists before extraction. Corrupted archives
are removed so rerunning the command downloads them again.
