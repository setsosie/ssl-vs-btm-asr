# Languages

Which languages each scale preset trains on, and the numbers behind the choice.
The 3- and 16-language presets come from the design of the ablation and predate
this document; the large preset (`configs/scales/64.yaml`) was selected here.

## The rule

A language is in the large preset when it has

- **at least 50 hours of training audio**, and
- **at least 2 hours of dev audio**, and
- **at least 2 hours of test audio**,

in the official Common Voice 25 partition — the `train.tsv`, `dev.tsv` and
`test.tsv` the loader reads, not the pooled `validated.tsv`. A language under
that line contributes more variance than signal to a multilingual mix and
teaches a merge nothing, which is what the large scale exists to measure.

**Twenty-four of Common Voice 25's 290 locales clear it.** The evaluation
thresholds turn out not to bind: every locale with 50 hours of training audio
also has at least 13.9 hours of dev and of test. The rule is, in practice, the
training threshold.

## Where the numbers come from

Mozilla publishes per-release statistics in
[`common-voice/cv-dataset`](https://github.com/common-voice/cv-dataset). The
file read for this document is

| | |
|---|---|
| Path | `datasets/scripted-speech/cv-corpus-25.0-2026-03-09.json` |
| Raw URL | `https://raw.githubusercontent.com/common-voice/cv-dataset/main/datasets/scripted-speech/cv-corpus-25.0-2026-03-09.json` |
| Blob SHA | `ac5fe102ab012f3fba02dd2eec1b65ef670826d5` |
| Repository HEAD when read | `f99d8239d2796131b73ac99f92ee7cb4443bf3ba` (2026-06-16) |

Common Voice 25.0 is the release this repository pins. Release 26.0 exists
(2026-06-12) and would move every number below.

Fetch it and regenerate this file:

```bash
gh api repos/common-voice/cv-dataset/contents/datasets/scripted-speech/cv-corpus-25.0-2026-03-09.json --jq '.content' | base64 -d > cv25.json
python scripts/select_languages.py --release cv25.json --preset-out configs/scales/64.yaml --table-out docs/languages.md
```

The prose above the language list in a preset is written by hand; the script
regenerates the list itself and the table at the end of this file. A test
asserts that the locales marked `yes` below are exactly the locales the preset
holds, so the two cannot drift apart.

### What the fields mean

Quoting the release's own
[documentation](https://github.com/common-voice/cv-dataset/tree/main/datasets/scripted-speech):

> We use the Corpora Creator tool to parse through metadata to generate train,
> dev, and test sets. The Corpora Creator eliminates duplication in clips and
> maximizes for speaker diversity.

> Each train/dev/test set is generated non-deterministically, meaning they will
> vary from release to release even for minor updates.

> `validated` -- clips with two or more validations where `up_votes` >
> `down_votes`

> Note that total clips in these sets will most probably not add up to the total
> validated clips because of this limitation.

That last line is the single most important fact here. The official `train`
split is heavily deduplicated — roughly one clip per sentence — so it is far
smaller than the validated pool. Japanese has 299,767 validated clips and 19,695
train clips; Russian has 251.9 validated hours and 38.7 train hours. A language
can look large in Common Voice's headline hours and still be small to train on.

### How a split's hours are computed

The release publishes a clip count per bucket and **one** mean clip duration per
locale (`avgDurationSecs`). It publishes no per-split duration, so

```
split hours = buckets.<split> x avgDurationSecs / 3600
```

is the best the document supports, and every hour figure in this file is that
product. It assumes the mean clip length of a split matches the mean of the
locale. Because Corpora Creator selects for speaker diversity rather than for
duration there is no reason to expect a systematic skew, but it is an
approximation, not a measurement. `svb data-stats` measures the real durations
from `clip_durations.tsv` once a corpus is actually on disk; that is the number
to trust, and it is the one to check the selection against before a large run.

`validated h` below is the release's own `validHrs`, not a product.

## Why the large preset is 32 languages and not 64

Twenty-four locales qualify. Eight more are in the preset because the
16-language preset commits to them and dropping them would stop the smaller
tiers from being subsets of the larger one. 24 + 8 = **32**.

Padding to 64 is not available at any defensible threshold. Holding the
evaluation thresholds at 2 hours and sweeping the training threshold:

| min train hours | locales |
|---:|---:|
| 50 (the rule) | 24 |
| 40 | 25 |
| 30 | 32 |
| 20 | 37 |
| 10 | 44 |
| 8 | 52 |
| 6 | 61 |
| **5** | **67** |

Reaching 64 languages needs the training threshold at about 5 hours — a tenth of
the rule, and less audio than a single speaker contributes in some of these
corpora. The scale tier is therefore reported as what it is.

One lever would change this, and it is a protocol decision rather than a data
one: training on `validated.tsv` minus the dev and test splits, instead of on
`train.tsv`. That pool is much larger — 48 locales have at least 54 validated
hours — but it is not the partition the loader reads, it reintroduces the
duplicate clips Corpora Creator removed, and it needs its own leakage argument.
Nothing in this repository does it today.

## Three rules that override the arithmetic

**Held-out transfer languages are excluded however large.** Malayalam, Marathi,
Telugu and Gujarati are held out for the transfer experiment
(`configs/scales/heldout.yaml`, sourced from OpenSLR). Three of them also exist
in Common Voice (`ml`, `mr`, `te`); training on the Common Voice half would make
the held-out number measure transfer to a language the pipeline had already
supervised. None of them comes close to the threshold anyway — Marathi, the
largest, has 3.8 train hours — so the guard costs nothing today and is there for
the release where it does not. Odia (`or`, 3.3 train hours) is **not** on that
list, because it is not in the held-out set; if the pending Odia decision in
`docs/data.md` ever adds it, this exclusion list has to grow with it.

**A language the smaller presets commit to is kept however small.** The eight
carried languages and what they actually have at Common Voice 25:

| locale | language | train h | dev h | test h | validated h |
|---|---|---:|---:|---:|---:|
| tr | Turkish | 43.6 | 12.6 | 12.6 | 129.2 |
| ru | Russian | 38.7 | 14.8 | 14.8 | 251.9 |
| uk | Ukrainian | 35.6 | 13.4 | 13.4 | 101.6 |
| ar | Arabic | 33.4 | 11.8 | 12.2 | 91.9 |
| pl | Polish | 32.3 | 12.8 | 12.8 | 176.6 |
| ja | Japanese | 24.4 | 11.2 | 11.2 | 371.9 |
| hi | Hindi | 6.9 | 3.9 | 4.7 | 15.6 |
| fi | Finnish | 2.7 | 2.3 | 2.3 | 15.8 |

Eight of the sixteen. This is worth saying plainly in any write-up: half the
16-language preset is trained on under 50 hours, Hindi on under 7 and Finnish on
under 3. Finnish and Hindi are small enough that a per-language result for them
is a small-sample result, and Hindi is in the 3-language preset too. Neither was
changed here — the smaller presets are part of the design that was already
committed to — but the numbers belong beside every result they produce.

**Where one language appears under several locales, only the largest stands for
it.** Common Voice carries regional and orthographic variants (`zh-CN` /
`zh-HK` / `zh-TW`, `rm-sursilv` / `rm-vallader`, and others); keeping two of them
would weight that language twice in a mix whose point is breadth across
languages. The rule is implemented and tested, and at this threshold it changes
nothing: no variant locale qualifies.

Esperanto (`eo`, 244.6 train hours) is kept. It is a constructed language rather
than one with a speech community of native speakers, which is worth a sentence
in a write-up, but it is a real Common Voice language with real recorded speech
and the rule has no clause that would exclude it.

## Writing system and the primary metric

`word_boundary: false` makes character error rate the primary metric for a
language, because whitespace tokenization of a transcript in a script that does
not separate words yields one token per sentence and word error rate over it
means nothing.

It is derived from the script the language is written in, not from a list of
locale codes. Each language's script is the ISO 15924 code CLDR resolves its
locale to, read from
[`unicode-org/cldr`](https://github.com/unicode-org/cldr) at
`common/supplemental/likelySubtags.xml` (blob
`11be52e4d0bce1e7e048b5ddaba2a6563c1f0d85`); a script in `Hani`, `Hans`, `Hant`,
`Jpan`, `Kore`, `Thai`, `Laoo`, `Khmr`, `Mymr` or `Tibt` sets `word_boundary`
false. CLDR has no `mhr` entry, so Meadow Mari takes the script of the Mari
macrolanguage `chm` (`chm_Cyrl_RU`).

Of the 32, **only Japanese** (`Jpan`) is written without spaces. Uzbek resolves
to `uz_Latn_UZ`, so the Latin orthography, not the Cyrillic one.

## Typological spread

The point of the large tier is not the 32 largest corpora, which would be almost
entirely Indo-European. The selected set covers **11 top-level families across
20 branches** and **7 scripts**:

| | |
|---|---|
| Families | Indo-European 14, Turkic 4, Atlantic-Congo 3, Uralic 3, Afro-Asiatic 2, Kartvelian 1, Dravidian 1, Northwest Caucasian 1, Japonic 1, isolate (Basque) 1, constructed (Esperanto) 1 |
| Scripts | Latin 19, Cyrillic 6, Arabic 3, Georgian 1, Tamil 1, Japanese 1, Devanagari 1 |

Indo-European is still 14 of 32, because Common Voice is what it is. The
languages that carry the merge-versus-typology analysis are the far ones —
Georgian, Tamil, Abkhaz, Kabyle, Uyghur, Pashto, Meadow Mari, Kinyarwanda,
Luganda, Basque — and all of them are here on the strength of the rule, not by
being reached for.

Family labels are the conventional genealogical ones and nothing computes on
them. The script codes are the load-bearing field, and those are CLDR-resolved.

## Every locale considered

All 290 locales of Common Voice 25, ordered by training hours. Family and script
are filled in only for selected locales: curating them for all 290 would be 290
claims nothing in this repository checks.

Hours are `clip count x avgDurationSecs / 3600` as described above, except
`validated h`, which is the release's own `validHrs`.

| locale | language | family | script | train h | dev h | test h | validated h | avg clip s | included | reason |
|---|---|---|---|---|---|---|---|---|---|---|
| ca | Catalan | Indo-European (Romance) | Latn | 1758.3 | 23.7 | 23.7 | 3359.8 | 5.194 | yes | meets the rule (train >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| en | English | Indo-European (Germanic) | Latn | 1679.0 | 24.0 | 24.0 | 2751.6 | 5.266 | yes | meets the rule (train >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| rw | Kinyarwanda | Atlantic-Congo (Bantu) | Latn | 1395.1 | 22.2 | 22.5 | 2001.6 | 5.007 | yes | meets the rule (train >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| de | German | Indo-European (Germanic) | Latn | 908.4 | 23.7 | 23.7 | 1388.8 | 5.265 | yes | meets the rule (train >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| fr | French | Indo-European (Romance) | Latn | 858.1 | 22.7 | 22.7 | 1095.9 | 5.036 | yes | meets the rule (train >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| es | Spanish | Indo-European (Romance) | Latn | 485.7 | 21.6 | 21.6 | 593.3 | 4.880 | yes | meets the rule (train >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| be | Belarusian | Indo-European (Slavic) | Cyrl | 462.9 | 21.1 | 21.1 | 1816.0 | 4.793 | yes | meets the rule (train >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| it | Italian | Indo-European (Romance) | Latn | 261.9 | 22.9 | 22.9 | 362.9 | 5.433 | yes | meets the rule (train >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| gl | Galician | Indo-European (Romance) | Latn | 256.2 | 21.1 | 21.2 | 308.0 | 4.988 | yes | meets the rule (train >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| eo | Esperanto | Constructed | Latn | 244.6 | 25.2 | 25.2 | 1441.1 | 6.078 | yes | meets the rule (train >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| mhr | Meadow Mari | Uralic (Mari) | Cyrl | 239.5 | 18.8 | 19.5 | 280.9 | 4.622 | yes | meets the rule (train >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| ps | Pashto | Indo-European (Iranian) | Arab | 239.0 | 16.9 | 16.9 | 3041.9 | 3.943 | yes | meets the rule (train >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| ug | Uyghur | Turkic (Karluk) | Arab | 209.4 | 24.2 | 24.2 | 450.6 | 5.922 | yes | meets the rule (train >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| eu | Basque | Isolate | Latn | 203.1 | 22.4 | 22.4 | 472.4 | 5.438 | yes | meets the rule (train >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| ab | Abkhaz | Northwest Caucasian | Cyrl | 148.8 | 21.6 | 21.7 | 207.4 | 5.505 | yes | meets the rule (train >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| ba | Bashkir | Turkic (Kipchak) | Cyrl | 146.5 | 17.9 | 17.9 | 258.8 | 4.427 | yes | meets the rule (train >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| kab | Kabyle | Afro-Asiatic (Berber) | Latn | 141.6 | 13.9 | 13.9 | 570.7 | 3.342 | yes | meets the rule (train >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| lg | Luganda | Atlantic-Congo (Bantu) | Latn | 114.2 | 21.5 | 21.5 | 436.8 | 5.784 | yes | meets the rule (train >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| hu | Hungarian | Uralic (Ugric) | Latn | 91.4 | 19.9 | 20.0 | 132.6 | 5.541 | yes | meets the rule (train >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| ka | Georgian | Kartvelian | Geor | 89.7 | 18.6 | 18.7 | 167.5 | 5.121 | yes | meets the rule (train >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| ta | Tamil | Dravidian | Taml | 79.9 | 20.9 | 21.0 | 234.8 | 6.182 | yes | meets the rule (train >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| sw | Swahili | Atlantic-Congo (Bantu) | Latn | 68.1 | 17.9 | 17.9 | 392.1 | 5.246 | yes | meets the rule (train >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| uz | Uzbek | Turkic (Karluk) | Latn | 56.5 | 14.2 | 14.3 | 101.0 | 4.159 | yes | meets the rule (train >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| nl | Dutch | Indo-European (Germanic) | Latn | 56.3 | 14.9 | 14.9 | 126.2 | 4.371 | yes | meets the rule (train >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| tr | Turkish | Turkic (Oghuz) | Latn | 43.6 | 12.6 | 12.6 | 129.2 | 3.850 | yes | below the rule (train 43.6 h < 50.0 h); kept because the smaller presets commit to it |
| ru | Russian | Indo-European (Slavic) | Cyrl | 38.7 | 14.8 | 14.8 | 251.9 | 5.179 | yes | below the rule (train 38.7 h < 50.0 h); kept because the smaller presets commit to it |
| th | — | — | — | 38.4 | 12.9 | 12.9 | 173.3 | 4.193 | no | train 38.4 h < 50.0 h |
| zh-CN | — | — | — | 37.3 | 13.4 | 13.4 | 239.1 | 4.539 | no | train 37.3 h < 50.0 h |
| uk | Ukrainian | Indo-European (Slavic) | Cyrl | 35.6 | 13.4 | 13.4 | 101.6 | 4.649 | yes | below the rule (train 35.6 h < 50.0 h); kept because the smaller presets commit to it |
| ar | Arabic | Afro-Asiatic (Semitic) | Arab | 33.4 | 11.8 | 12.1 | 91.9 | 4.162 | yes | below the rule (train 33.4 h < 50.0 h); kept because the smaller presets commit to it |
| fa | — | — | — | 33.0 | 11.7 | 11.7 | 372.9 | 3.942 | no | train 33.0 h < 50.0 h |
| pl | Polish | Indo-European (Slavic) | Latn | 32.3 | 12.8 | 12.8 | 176.6 | 4.572 | yes | below the rule (train 32.3 h < 50.0 h); kept because the smaller presets commit to it |
| cs | — | — | — | 27.6 | 11.7 | 11.7 | 81.1 | 4.460 | no | train 27.6 h < 50.0 h |
| pt | — | — | — | 26.9 | 11.2 | 11.2 | 187.3 | 4.188 | no | train 26.9 h < 50.0 h |
| bn | — | — | — | 26.2 | 11.4 | 11.4 | 54.3 | 4.371 | no | train 26.2 h < 50.0 h |
| ja | Japanese | Japonic | Jpan | 24.4 | 11.2 | 11.2 | 371.9 | 4.467 | yes | below the rule (train 24.4 h < 50.0 h); kept because the smaller presets commit to it |
| kbd | — | — | — | 24.0 | 13.1 | 13.1 | 272.2 | 6.223 | no | train 24.0 h < 50.0 h |
| lv | — | — | — | 19.7 | 10.5 | 10.5 | 265.4 | 4.810 | no | train 19.7 h < 50.0 h |
| mrj | — | — | — | 16.7 | 8.5 | 8.3 | 33.7 | 4.193 | no | train 16.7 h < 50.0 h |
| hy-AM | — | — | — | 15.7 | 8.9 | 9.3 | 34.3 | 5.383 | no | train 15.7 h < 50.0 h |
| kln | — | — | — | 13.9 | 8.1 | 7.6 | 40.6 | 4.530 | no | train 13.9 h < 50.0 h |
| lt | — | — | — | 12.2 | 7.9 | 8.0 | 28.4 | 5.104 | no | train 12.2 h < 50.0 h |
| sk | — | — | — | 11.6 | 6.6 | 7.1 | 56.6 | 4.296 | no | train 11.6 h < 50.0 h |
| cy | — | — | — | 11.0 | 7.4 | 7.4 | 124.1 | 4.916 | no | train 11.0 h < 50.0 h |
| zh-HK | — | — | — | 9.7 | 6.5 | 6.5 | 108.5 | 4.151 | no | train 9.7 h < 50.0 h |
| sv-SE | — | — | — | 9.2 | 6.1 | 6.2 | 47.7 | 4.024 | no | train 9.2 h < 50.0 h |
| tt | — | — | — | 9.0 | 4.7 | 5.4 | 32.3 | 3.835 | no | train 9.0 h < 50.0 h |
| ckb | — | — | — | 9.0 | 6.1 | 6.1 | 137.0 | 4.095 | no | train 9.0 h < 50.0 h |
| ur | — | — | — | 8.6 | 6.0 | 6.0 | 79.7 | 4.230 | no | train 8.6 h < 50.0 h |
| nan-tw | — | — | — | 8.5 | 4.4 | 4.7 | 21.8 | 2.651 | no | train 8.5 h < 50.0 h |
| oru | — | — | — | 8.4 | 3.5 | 3.0 | 21.2 | 7.306 | no | train 8.4 h < 50.0 h |
| yue | — | — | — | 8.2 | 5.6 | 5.6 | 210.7 | 3.961 | no | train 8.2 h < 50.0 h |
| tli | — | — | — | 8.1 | 0.0 | 1.8 | 9.9 | 12.604 | no | train 8.1 h < 50.0 h |
| bg | — | — | — | 7.6 | 4.5 | 5.2 | 17.3 | 5.475 | no | train 7.6 h < 50.0 h |
| kw | — | — | — | 7.1 | 0.0 | 2.7 | 12.4 | 4.121 | no | train 7.1 h < 50.0 h |
| zh-TW | — | — | — | 6.9 | 4.8 | 4.8 | 79.7 | 3.362 | no | train 6.9 h < 50.0 h |
| hi | Hindi | Indo-European (Indo-Aryan) | Deva | 6.9 | 3.9 | 4.7 | 15.6 | 5.045 | yes | below the rule (train 6.9 h < 50.0 h); kept because the smaller presets commit to it |
| ipk | — | — | — | 6.8 | 0.0 | 0.4 | 7.2 | 7.938 | no | train 6.8 h < 50.0 h |
| et | — | — | — | 6.5 | 5.4 | 5.4 | 51.7 | 6.717 | no | train 6.5 h < 50.0 h |
| kmr | — | — | — | 6.5 | 4.8 | 4.8 | 75.7 | 4.161 | no | train 6.5 h < 50.0 h |
| esu | — | — | — | 6.5 | 0.0 | 0.5 | 7.6 | 3.800 | no | train 6.5 h < 50.0 h |
| ltg | — | — | — | 6.3 | 4.9 | 4.9 | 30.1 | 4.775 | no | train 6.3 h < 50.0 h |
| lzz | — | — | — | 6.2 | 4.4 | 4.3 | 26.1 | 4.450 | no | train 6.2 h < 50.0 h |
| ady | — | — | — | 6.2 | 4.9 | 4.9 | 70.5 | 5.217 | no | train 6.2 h < 50.0 h |
| luo | — | — | — | 6.1 | 4.1 | 4.1 | 27.5 | 4.873 | no | train 6.1 h < 50.0 h |
| ro | — | — | — | 5.8 | 4.4 | 4.4 | 22.3 | 4.046 | no | train 5.8 h < 50.0 h |
| ia | — | — | — | 5.7 | 2.2 | 2.2 | 14.3 | 4.202 | no | train 5.7 h < 50.0 h |
| id | — | — | — | 5.5 | 3.8 | 4.1 | 33.6 | 3.991 | no | train 5.5 h < 50.0 h |
| dml | — | — | — | 5.3 | 1.8 | 1.0 | 10.2 | 6.019 | no | train 5.3 h < 50.0 h |
| an | — | — | — | 5.3 | 3.6 | 3.7 | 16.9 | 4.574 | no | train 5.3 h < 50.0 h |
| fy-NL | — | — | — | 5.3 | 4.3 | 4.3 | 70.4 | 4.852 | no | train 5.3 h < 50.0 h |
| bgp | — | — | — | 5.3 | 2.1 | 2.8 | 11.5 | 5.418 | no | train 5.3 h < 50.0 h |
| gwc | — | — | — | 5.1 | 1.2 | 1.4 | 11.6 | 5.651 | no | train 5.1 h < 50.0 h |
| gwt | — | — | — | 5.0 | 0.0 | 0.3 | 12.2 | 5.722 | no | train 5.0 h < 50.0 h |
| khw | — | — | — | 5.0 | 3.0 | 2.9 | 16.0 | 6.777 | no | train 5.0 h < 50.0 h |
| wbl | — | — | — | 4.8 | 2.1 | 2.1 | 12.1 | 6.753 | no | train 4.8 h < 50.0 h |
| trw | — | — | — | 4.3 | 3.1 | 2.9 | 16.5 | 5.373 | no | train 4.3 h < 50.0 h |
| da | — | — | — | 4.1 | 3.1 | 3.1 | 13.0 | 4.072 | no | train 4.1 h < 50.0 h |
| sah | — | — | — | 4.1 | 3.1 | 3.2 | 16.3 | 6.336 | no | train 4.1 h < 50.0 h |
| br | — | — | — | 4.0 | 3.2 | 3.2 | 30.7 | 3.256 | no | train 4.0 h < 50.0 h |
| gv | — | — | — | 3.9 | 1.9 | 0.8 | 10.1 | 5.782 | no | train 3.9 h < 50.0 h |
| mr | — | — | — | 3.8 | 3.0 | 3.1 | 18.9 | 6.200 | no | held-out transfer language (marathi); never trained on |
| dv | — | — | — | 3.8 | 3.2 | 3.2 | 37.8 | 5.120 | no | train 3.8 h < 50.0 h |
| sq | — | — | — | 3.8 | 2.5 | 2.7 | 9.0 | 5.088 | no | train 3.8 h < 50.0 h |
| bft | — | — | — | 3.6 | 3.0 | 3.0 | 16.6 | 5.928 | no | train 3.6 h < 50.0 h |
| plk | — | — | — | 3.5 | 1.7 | 0.9 | 12.6 | 5.026 | no | train 3.5 h < 50.0 h |
| mvy | — | — | — | 3.4 | 2.6 | 2.8 | 22.1 | 4.791 | no | train 3.4 h < 50.0 h |
| brh | — | — | — | 3.3 | 1.0 | 1.7 | 9.9 | 7.019 | no | train 3.3 h < 50.0 h |
| or | — | — | — | 3.3 | 1.0 | 0.7 | 6.3 | 5.539 | no | train 3.3 h < 50.0 h |
| tig | — | — | — | 3.3 | 2.7 | 2.7 | 10.7 | 5.984 | no | train 3.3 h < 50.0 h |
| dar | — | — | — | 3.2 | 2.3 | 2.2 | 14.5 | 5.703 | no | train 3.2 h < 50.0 h |
| tok | — | — | — | 3.2 | 2.7 | 2.7 | 15.6 | 4.310 | no | train 3.2 h < 50.0 h |
| dmk | — | — | — | 3.1 | 0.0 | 0.9 | 10.2 | 3.445 | no | train 3.1 h < 50.0 h |
| xhe | — | — | — | 3.0 | 0.0 | 1.2 | 9.6 | 3.032 | no | train 3.0 h < 50.0 h |
| mn | — | — | — | 3.0 | 2.6 | 2.6 | 45.8 | 4.892 | no | train 3.0 h < 50.0 h |
| gju | — | — | — | 2.9 | 0.0 | 0.6 | 10.1 | 3.274 | no | train 2.9 h < 50.0 h |
| rm-sursilv | — | — | — | 2.9 | 2.4 | 2.5 | 7.9 | 5.379 | no | train 2.9 h < 50.0 h |
| yaq | — | — | — | 2.8 | 0.2 | 1.2 | 10.2 | 5.311 | no | train 2.8 h < 50.0 h |
| mk | — | — | — | 2.8 | 2.4 | 2.4 | 25.0 | 4.807 | no | train 2.8 h < 50.0 h |
| phl | — | — | — | 2.7 | 1.9 | 1.9 | 21.3 | 4.921 | no | train 2.7 h < 50.0 h |
| fi | Finnish | Uralic (Finnic) | Latn | 2.7 | 2.3 | 2.3 | 15.8 | 4.629 | yes | below the rule (train 2.7 h < 50.0 h); kept because the smaller presets commit to it |
| nyu | — | — | — | 2.6 | 0.0 | 0.6 | 9.2 | 9.059 | no | train 2.6 h < 50.0 h |
| mt | — | — | — | 2.5 | 2.1 | 2.2 | 8.7 | 4.747 | no | train 2.5 h < 50.0 h |
| lij | — | — | — | 2.5 | 1.0 | 1.5 | 5.0 | 3.870 | no | train 2.5 h < 50.0 h |
| he | — | — | — | 2.4 | 0.5 | 1.2 | 5.3 | 4.551 | no | train 2.4 h < 50.0 h |
| yo | — | — | — | 2.4 | 1.6 | 1.8 | 5.8 | 6.040 | no | train 2.4 h < 50.0 h |
| dav | — | — | — | 2.4 | 1.4 | 1.1 | 9.3 | 4.059 | no | train 2.4 h < 50.0 h |
| ha | — | — | — | 2.3 | 0.8 | 0.9 | 4.2 | 4.351 | no | train 2.3 h < 50.0 h |
| bas | — | — | — | 2.3 | 1.4 | 1.7 | 12.1 | 3.909 | no | train 2.3 h < 50.0 h |
| sr | — | — | — | 2.3 | 1.7 | 1.8 | 7.6 | 3.261 | no | train 2.3 h < 50.0 h |
| ky | — | — | — | 2.3 | 2.0 | 2.0 | 38.9 | 4.553 | no | train 2.3 h < 50.0 h |
| el | — | — | — | 2.2 | 2.0 | 2.0 | 19.9 | 4.152 | no | train 2.2 h < 50.0 h |
| gn | — | — | — | 2.2 | 0.8 | 1.4 | 5.2 | 4.597 | no | train 2.2 h < 50.0 h |
| vi | — | — | — | 2.1 | 1.5 | 1.6 | 7.3 | 3.997 | no | train 2.1 h < 50.0 h |
| bsh | — | — | — | 2.1 | 0.8 | 1.0 | 9.9 | 5.264 | no | train 2.1 h < 50.0 h |
| cv | — | — | — | 2.0 | 1.7 | 1.8 | 24.5 | 5.042 | no | train 2.0 h < 50.0 h |
| tay | — | — | — | 2.0 | 0.7 | 1.3 | 11.5 | 5.572 | no | train 2.0 h < 50.0 h |
| myv | — | — | — | 2.0 | 0.4 | 0.8 | 3.2 | 5.784 | no | train 2.0 h < 50.0 h |
| dag | — | — | — | 1.9 | 1.7 | 1.7 | 16.2 | 4.294 | no | train 1.9 h < 50.0 h |
| ssi | — | — | — | 1.9 | 0.0 | 0.2 | 10.5 | 3.721 | no | train 1.9 h < 50.0 h |
| ggg | — | — | — | 1.9 | 0.0 | 0.4 | 7.4 | 3.988 | no | train 1.9 h < 50.0 h |
| skr | — | — | — | 1.8 | 1.3 | 1.2 | 4.3 | 4.164 | no | train 1.8 h < 50.0 h |
| mki | — | — | — | 1.8 | 0.0 | 0.0 | 9.9 | 3.194 | no | train 1.8 h < 50.0 h |
| kxp | — | — | — | 1.8 | 0.0 | 0.2 | 11.0 | 3.403 | no | train 1.8 h < 50.0 h |
| sbn | — | — | — | 1.7 | 0.0 | 0.2 | 10.7 | 3.524 | no | train 1.7 h < 50.0 h |
| mve | — | — | — | 1.7 | 0.7 | 0.2 | 10.1 | 4.739 | no | train 1.7 h < 50.0 h |
| dru | — | — | — | 1.7 | 1.5 | 1.5 | 10.4 | 5.676 | no | train 1.7 h < 50.0 h |
| hsb | — | — | — | 1.7 | 0.7 | 1.0 | 3.4 | 7.503 | no | train 1.7 h < 50.0 h |
| odk | — | — | — | 1.7 | 0.8 | 1.1 | 11.2 | 6.369 | no | train 1.7 h < 50.0 h |
| sl | — | — | — | 1.6 | 1.5 | 1.5 | 17.3 | 3.986 | no | train 1.6 h < 50.0 h |
| lrk | — | — | — | 1.6 | 0.0 | 0.3 | 11.1 | 3.480 | no | train 1.6 h < 50.0 h |
| scl | — | — | — | 1.6 | 1.0 | 1.1 | 10.0 | 4.072 | no | train 1.6 h < 50.0 h |
| mdd | — | — | — | 1.6 | 0.0 | 0.1 | 10.0 | 6.498 | no | train 1.6 h < 50.0 h |
| pua | — | — | — | 1.6 | 0.7 | 0.7 | 10.2 | 4.885 | no | train 1.6 h < 50.0 h |
| as | — | — | — | 1.6 | 0.8 | 0.7 | 3.0 | 5.857 | no | train 1.6 h < 50.0 h |
| gig | — | — | — | 1.6 | 0.0 | 0.1 | 10.1 | 2.992 | no | train 1.6 h < 50.0 h |
| nb-NO | — | — | — | 1.5 | 0.5 | 0.4 | 2.3 | 4.193 | no | train 1.5 h < 50.0 h |
| bnn | — | — | — | 1.5 | 1.4 | 1.4 | 10.3 | 5.090 | no | train 1.5 h < 50.0 h |
| kls | — | — | — | 1.5 | 1.3 | 1.3 | 10.0 | 3.699 | no | train 1.5 h < 50.0 h |
| ml | — | — | — | 1.5 | 1.1 | 1.0 | 4.1 | 4.242 | no | held-out transfer language (malayalam); never trained on |
| ydg | — | — | — | 1.4 | 0.0 | 0.4 | 10.6 | 3.596 | no | train 1.4 h < 50.0 h |
| tvu | — | — | — | 1.4 | 1.0 | 0.9 | 10.2 | 7.009 | no | train 1.4 h < 50.0 h |
| bsk | — | — | — | 1.4 | 0.4 | 0.9 | 10.2 | 4.262 | no | train 1.4 h < 50.0 h |
| xka | — | — | — | 1.4 | 0.0 | 0.4 | 9.8 | 3.159 | no | train 1.4 h < 50.0 h |
| trv | — | — | — | 1.4 | 0.9 | 0.8 | 9.9 | 5.611 | no | train 1.4 h < 50.0 h |
| kvx | — | — | — | 1.3 | 1.0 | 0.7 | 11.0 | 5.322 | no | train 1.3 h < 50.0 h |
| fue | — | — | — | 1.3 | 0.0 | 0.1 | 10.6 | 5.332 | no | train 1.3 h < 50.0 h |
| tn | — | — | — | 1.3 | 0.4 | 0.4 | 4.2 | 4.370 | no | train 1.3 h < 50.0 h |
| cux | — | — | — | 1.3 | 0.7 | 0.6 | 10.3 | 4.101 | no | train 1.3 h < 50.0 h |
| ajg | — | — | — | 1.2 | 0.4 | 0.5 | 12.7 | 2.389 | no | train 1.2 h < 50.0 h |
| sc | — | — | — | 1.2 | 0.7 | 0.9 | 3.0 | 4.705 | no | train 1.2 h < 50.0 h |
| nla | — | — | — | 1.1 | 0.3 | 0.4 | 8.9 | 6.930 | no | train 1.1 h < 50.0 h |
| tk | — | — | — | 1.1 | 0.8 | 0.8 | 3.1 | 5.509 | no | train 1.1 h < 50.0 h |
| var | — | — | — | 1.1 | 0.8 | 0.9 | 10.1 | 5.307 | no | train 1.1 h < 50.0 h |
| eko | — | — | — | 1.1 | 0.7 | 0.9 | 8.3 | 7.541 | no | train 1.1 h < 50.0 h |
| qws | — | — | — | 1.1 | 0.0 | 0.1 | 10.3 | 4.164 | no | train 1.1 h < 50.0 h |
| pa-IN | — | — | — | 1.1 | 0.7 | 0.7 | 2.4 | 4.806 | no | train 1.1 h < 50.0 h |
| haz | — | — | — | 1.0 | 0.1 | 0.6 | 10.5 | 4.556 | no | train 1.0 h < 50.0 h |
| pwn | — | — | — | 1.0 | 1.0 | 1.0 | 14.6 | 4.873 | no | train 1.0 h < 50.0 h |
| nnh | — | — | — | 1.0 | 0.5 | 0.7 | 18.9 | 8.916 | no | train 1.0 h < 50.0 h |
| mgg | — | — | — | 1.0 | 0.6 | 0.4 | 10.2 | 7.580 | no | train 1.0 h < 50.0 h |
| gjk | — | — | — | 1.0 | 0.7 | 0.8 | 10.8 | 4.528 | no | train 1.0 h < 50.0 h |
| ush | — | — | — | 1.0 | 0.3 | 0.6 | 6.6 | 6.148 | no | train 1.0 h < 50.0 h |
| mcx | — | — | — | 1.0 | 0.2 | 0.5 | 10.1 | 6.611 | no | train 1.0 h < 50.0 h |
| bbl | — | — | — | 1.0 | 0.9 | 0.9 | 11.2 | 8.787 | no | train 1.0 h < 50.0 h |
| ga-IE | — | — | — | 1.0 | 0.9 | 0.9 | 14.2 | 3.831 | no | train 1.0 h < 50.0 h |
| ko | — | — | — | 1.0 | 0.7 | 0.8 | 2.5 | 5.211 | no | train 1.0 h < 50.0 h |
| am | — | — | — | 1.0 | 0.4 | 0.5 | 1.9 | 6.345 | no | train 1.0 h < 50.0 h |
| qxp | — | — | — | 1.0 | 0.9 | 0.9 | 31.2 | 4.937 | no | train 1.0 h < 50.0 h |
| hno | — | — | — | 1.0 | 0.9 | 0.8 | 10.2 | 4.009 | no | train 1.0 h < 50.0 h |
| mau | — | — | — | 1.0 | 0.4 | 0.5 | 10.4 | 6.216 | no | train 1.0 h < 50.0 h |
| ig | — | — | — | 1.0 | 0.9 | 0.9 | 2.7 | 5.428 | no | train 1.0 h < 50.0 h |
| qur | — | — | — | 0.9 | 0.0 | 0.0 | 10.0 | 3.470 | no | train 0.9 h < 50.0 h |
| sva | — | — | — | 0.9 | 0.8 | 0.8 | 15.7 | 5.921 | no | train 0.9 h < 50.0 h |
| ibb | — | — | — | 0.9 | 0.8 | 0.8 | 7.8 | 9.002 | no | train 0.9 h < 50.0 h |
| tui | — | — | — | 0.9 | 0.8 | 0.8 | 9.7 | 4.634 | no | train 0.9 h < 50.0 h |
| kk | — | — | — | 0.9 | 0.8 | 0.8 | 2.5 | 4.946 | no | train 0.9 h < 50.0 h |
| sei | — | — | — | 0.9 | 0.5 | 0.6 | 10.1 | 4.535 | no | train 0.9 h < 50.0 h |
| zza | — | — | — | 0.9 | 0.5 | 0.5 | 1.9 | 4.032 | no | train 0.9 h < 50.0 h |
| rm-vallader | — | — | — | 0.9 | 0.8 | 0.8 | 2.5 | 5.832 | no | train 0.9 h < 50.0 h |
| qxw | — | — | — | 0.9 | 0.2 | 0.3 | 11.7 | 5.250 | no | train 0.9 h < 50.0 h |
| zgh | — | — | — | 0.9 | 0.3 | 0.2 | 1.4 | 3.559 | no | train 0.9 h < 50.0 h |
| gej | — | — | — | 0.9 | 0.6 | 0.6 | 11.1 | 2.444 | no | train 0.9 h < 50.0 h |
| ksf | — | — | — | 0.9 | 0.7 | 0.7 | 17.2 | 8.354 | no | train 0.9 h < 50.0 h |
| lss | — | — | — | 0.9 | 0.5 | 0.6 | 9.9 | 3.463 | no | train 0.9 h < 50.0 h |
| nmg | — | — | — | 0.9 | 0.5 | 0.5 | 10.4 | 6.401 | no | train 0.9 h < 50.0 h |
| mhk | — | — | — | 0.8 | 0.5 | 0.3 | 11.3 | 6.149 | no | train 0.8 h < 50.0 h |
| bri | — | — | — | 0.8 | 0.2 | 0.4 | 10.4 | 4.241 | no | train 0.8 h < 50.0 h |
| bnm | — | — | — | 0.8 | 0.6 | 0.7 | 15.3 | 7.173 | no | train 0.8 h < 50.0 h |
| phr | — | — | — | 0.8 | 0.7 | 0.7 | 14.0 | 3.963 | no | train 0.8 h < 50.0 h |
| mug | — | — | — | 0.8 | 0.6 | 0.6 | 5.4 | 7.273 | no | train 0.8 h < 50.0 h |
| nlv | — | — | — | 0.8 | 0.4 | 0.5 | 11.6 | 6.272 | no | train 0.8 h < 50.0 h |
| qwa | — | — | — | 0.8 | 0.1 | 0.4 | 9.9 | 4.951 | no | train 0.8 h < 50.0 h |
| cnh | — | — | — | 0.8 | 0.7 | 0.7 | 2.4 | 3.517 | no | train 0.8 h < 50.0 h |
| qxu | — | — | — | 0.8 | 0.0 | 0.4 | 10.1 | 4.186 | no | train 0.8 h < 50.0 h |
| yav | — | — | — | 0.8 | 0.5 | 0.6 | 8.6 | 6.655 | no | train 0.8 h < 50.0 h |
| fmp | — | — | — | 0.8 | 0.7 | 0.7 | 11.4 | 7.865 | no | train 0.8 h < 50.0 h |
| fub | — | — | — | 0.8 | 0.5 | 0.6 | 12.9 | 6.068 | no | train 0.8 h < 50.0 h |
| ebr | — | — | — | 0.7 | 0.0 | 0.3 | 1.8 | 4.167 | no | train 0.7 h < 50.0 h |
| qup | — | — | — | 0.7 | 0.5 | 0.4 | 11.9 | 5.932 | no | train 0.7 h < 50.0 h |
| qxt | — | — | — | 0.7 | 0.1 | 0.4 | 10.3 | 4.290 | no | train 0.7 h < 50.0 h |
| nn-NO | — | — | — | 0.7 | 0.4 | 0.5 | 1.6 | 4.441 | no | train 0.7 h < 50.0 h |
| mse | — | — | — | 0.7 | 0.6 | 0.5 | 7.7 | 6.412 | no | train 0.7 h < 50.0 h |
| cut | — | — | — | 0.7 | 0.5 | 0.6 | 10.1 | 6.631 | no | train 0.7 h < 50.0 h |
| nmz | — | — | — | 0.7 | 0.7 | 0.7 | 11.2 | 2.947 | no | train 0.7 h < 50.0 h |
| hux | — | — | — | 0.7 | 0.0 | 0.4 | 10.0 | 3.876 | no | train 0.7 h < 50.0 h |
| bbj | — | — | — | 0.7 | 0.5 | 0.5 | 12.3 | 6.110 | no | train 0.7 h < 50.0 h |
| jgo | — | — | — | 0.7 | 0.6 | 0.6 | 11.3 | 6.695 | no | train 0.7 h < 50.0 h |
| bkh | — | — | — | 0.7 | 0.5 | 0.5 | 10.0 | 6.400 | no | train 0.7 h < 50.0 h |
| jqr | — | — | — | 0.7 | 0.5 | 0.4 | 9.9 | 5.898 | no | train 0.7 h < 50.0 h |
| bag | — | — | — | 0.7 | 0.4 | 0.5 | 11.0 | 5.636 | no | train 0.7 h < 50.0 h |
| dua | — | — | — | 0.7 | 0.6 | 0.6 | 12.5 | 6.424 | no | train 0.7 h < 50.0 h |
| gid | — | — | — | 0.6 | 0.6 | 0.6 | 9.9 | 7.000 | no | train 0.6 h < 50.0 h |
| bci | — | — | — | 0.6 | 0.5 | 0.6 | 11.3 | 7.210 | no | train 0.6 h < 50.0 h |
| cpy | — | — | — | 0.6 | 0.2 | 0.4 | 10.0 | 4.421 | no | train 0.6 h < 50.0 h |
| os | — | — | — | 0.6 | 0.4 | 0.3 | 1.4 | 5.523 | no | train 0.6 h < 50.0 h |
| qux | — | — | — | 0.6 | 0.4 | 0.5 | 9.8 | 5.721 | no | train 0.6 h < 50.0 h |
| cjk | — | — | — | 0.6 | 0.6 | 0.6 | 12.0 | 5.874 | no | train 0.6 h < 50.0 h |
| giz | — | — | — | 0.6 | 0.5 | 0.4 | 10.1 | 5.572 | no | train 0.6 h < 50.0 h |
| mbo | — | — | — | 0.6 | 0.5 | 0.4 | 10.9 | 5.513 | no | train 0.6 h < 50.0 h |
| qvj | — | — | — | 0.6 | 0.6 | 0.6 | 10.8 | 5.961 | no | train 0.6 h < 50.0 h |
| zoc | — | — | — | 0.6 | 0.5 | 0.5 | 10.1 | 4.083 | no | train 0.6 h < 50.0 h |
| udl | — | — | — | 0.6 | 0.4 | 0.5 | 9.5 | 5.352 | no | train 0.6 h < 50.0 h |
| byv | — | — | — | 0.6 | 0.6 | 0.6 | 13.2 | 6.178 | no | train 0.6 h < 50.0 h |
| hem | — | — | — | 0.6 | 0.6 | 0.6 | 9.9 | 5.999 | no | train 0.6 h < 50.0 h |
| xmf | — | — | — | 0.6 | 0.6 | 0.6 | 11.6 | 6.156 | no | train 0.6 h < 50.0 h |
| qva | — | — | — | 0.6 | 0.2 | 0.4 | 9.8 | 4.287 | no | train 0.6 h < 50.0 h |
| abb | — | — | — | 0.6 | 0.4 | 0.4 | 11.2 | 5.109 | no | train 0.6 h < 50.0 h |
| ewo | — | — | — | 0.6 | 0.5 | 0.5 | 13.6 | 6.472 | no | train 0.6 h < 50.0 h |
| lua | — | — | — | 0.6 | 0.5 | 0.5 | 8.9 | 6.664 | no | train 0.6 h < 50.0 h |
| bce | — | — | — | 0.5 | 0.5 | 0.5 | 10.0 | 5.892 | no | train 0.5 h < 50.0 h |
| ast | — | — | — | 0.5 | 0.1 | 0.3 | 1.0 | 4.397 | no | train 0.5 h < 50.0 h |
| prq | — | — | — | 0.5 | 0.4 | 0.4 | 9.7 | 4.726 | no | train 0.5 h < 50.0 h |
| qxa | — | — | — | 0.5 | 0.3 | 0.4 | 10.1 | 4.343 | no | train 0.5 h < 50.0 h |
| tar | — | — | — | 0.5 | 0.5 | 0.5 | 10.0 | 4.549 | no | train 0.5 h < 50.0 h |
| pcm | — | — | — | 0.5 | 0.5 | 0.5 | 12.5 | 5.808 | no | train 0.5 h < 50.0 h |
| beb | — | — | — | 0.5 | 0.5 | 0.5 | 10.0 | 5.347 | no | train 0.5 h < 50.0 h |
| yi | — | — | — | 0.5 | 0.5 | 0.5 | 2.0 | 3.975 | no | train 0.5 h < 50.0 h |
| mxu | — | — | — | 0.5 | 0.5 | 0.5 | 12.3 | 5.733 | no | train 0.5 h < 50.0 h |
| rof | — | — | — | 0.5 | 0.5 | 0.5 | 10.4 | 3.905 | no | train 0.5 h < 50.0 h |
| gya | — | — | — | 0.5 | 0.5 | 0.5 | 9.8 | 5.078 | no | train 0.5 h < 50.0 h |
| bax | — | — | — | 0.5 | 0.4 | 0.5 | 10.6 | 4.917 | no | train 0.5 h < 50.0 h |
| btv | — | — | — | 0.5 | 0.3 | 0.4 | 10.3 | 4.057 | no | train 0.5 h < 50.0 h |
| qvl | — | — | — | 0.5 | 0.2 | 0.4 | 10.0 | 3.960 | no | train 0.5 h < 50.0 h |
| mcf | — | — | — | 0.5 | 0.0 | 0.2 | 10.2 | 2.656 | no | train 0.5 h < 50.0 h |
| qvi | — | — | — | 0.5 | 0.4 | 0.5 | 11.4 | 4.427 | no | train 0.5 h < 50.0 h |
| bba | — | — | — | 0.5 | 0.4 | 0.4 | 10.6 | 6.068 | no | train 0.5 h < 50.0 h |
| bkm | — | — | — | 0.5 | 0.5 | 0.5 | 11.4 | 5.425 | no | train 0.5 h < 50.0 h |
| mua | — | — | — | 0.5 | 0.3 | 0.4 | 9.7 | 4.006 | no | train 0.5 h < 50.0 h |
| tg | — | — | — | 0.5 | 0.2 | 0.2 | 0.8 | 5.027 | no | train 0.5 h < 50.0 h |
| bum | — | — | — | 0.4 | 0.3 | 0.4 | 10.0 | 4.657 | no | train 0.4 h < 50.0 h |
| ncx | — | — | — | 0.4 | 0.4 | 0.4 | 10.7 | 4.457 | no | train 0.4 h < 50.0 h |
| fan | — | — | — | 0.4 | 0.4 | 0.4 | 9.4 | 4.399 | no | train 0.4 h < 50.0 h |
| szy | — | — | — | 0.4 | 0.4 | 0.4 | 13.7 | 5.397 | no | train 0.4 h < 50.0 h |
| sat | — | — | — | 0.4 | 0.1 | 0.2 | 0.7 | 4.481 | no | train 0.4 h < 50.0 h |
| oc | — | — | — | 0.4 | 0.4 | 0.4 | 2.7 | 4.871 | no | train 0.4 h < 50.0 h |
| ne-NP | — | — | — | 0.4 | 0.4 | 0.3 | 1.3 | 4.111 | no | train 0.4 h < 50.0 h |
| bfd | — | — | — | 0.4 | 0.4 | 0.4 | 10.1 | 5.609 | no | train 0.4 h < 50.0 h |
| wes | — | — | — | 0.4 | 0.4 | 0.4 | 10.3 | 4.118 | no | train 0.4 h < 50.0 h |
| mcn | — | — | — | 0.4 | 0.4 | 0.4 | 10.1 | 4.214 | no | train 0.4 h < 50.0 h |
| qus | — | — | — | 0.4 | 0.3 | 0.3 | 10.8 | 3.749 | no | train 0.4 h < 50.0 h |
| kdh | — | — | — | 0.3 | 0.2 | 0.2 | 9.2 | 2.448 | no | train 0.3 h < 50.0 h |
| az | — | — | — | 0.3 | 0.1 | 0.2 | 0.7 | 5.452 | no | train 0.3 h < 50.0 h |
| eto | — | — | — | 0.3 | 0.3 | 0.3 | 9.4 | 3.225 | no | train 0.3 h < 50.0 h |
| af | — | — | — | 0.3 | 0.2 | 0.2 | 0.8 | 6.080 | no | train 0.3 h < 50.0 h |
| sd | — | — | — | 0.3 | 0.0 | 0.0 | 0.4 | 4.070 | no | train 0.3 h < 50.0 h |
| tw | — | — | — | 0.3 | 0.0 | 0.0 | 0.3 | 4.385 | no | train 0.3 h < 50.0 h |
| mdf | — | — | — | 0.3 | 0.1 | 0.2 | 0.5 | 5.250 | no | train 0.3 h < 50.0 h |
| lo | — | — | — | 0.2 | 0.1 | 0.1 | 0.3 | 6.532 | no | train 0.2 h < 50.0 h |
| dyu | — | — | — | 0.2 | 0.1 | 0.1 | 0.4 | 6.309 | no | train 0.2 h < 50.0 h |
| is | — | — | — | 0.1 | 0.0 | 0.1 | 0.2 | 6.425 | no | train 0.1 h < 50.0 h |
| te | — | — | — | 0.1 | 0.1 | 0.1 | 0.8 | 4.195 | no | held-out transfer language (telugu); never trained on |
| vot | — | — | — | 0.1 | 0.0 | 0.0 | 0.1 | 2.412 | no | train 0.1 h < 50.0 h |
| ti | — | — | — | 0.1 | 0.0 | 0.0 | 0.1 | 5.194 | no | train 0.1 h < 50.0 h |
| gsw | — | — | — | 0.0 | 0.0 | 0.0 | 0.6 | 5.716 | no | train 0.0 h < 50.0 h |
| quy | — | — | — | 0.0 | 0.0 | 0.0 | 0.1 | 4.994 | no | train 0.0 h < 50.0 h |
| nhi | — | — | — | 0.0 | 0.0 | 0.0 | 0.1 | 5.081 | no | train 0.0 h < 50.0 h |
| rup | — | — | — | 0.0 | 0.0 | 0.0 | 0.1 | 7.021 | no | train 0.0 h < 50.0 h |
| ms | — | — | — | 0.0 | 0.0 | 0.0 | 0.1 | 6.176 | no | train 0.0 h < 50.0 h |
| zu | — | — | — | 0.0 | 0.0 | 0.0 | 0.0 | 5.533 | no | train 0.0 h < 50.0 h |
| nso | — | — | — | 0.0 | 0.0 | 0.0 | 0.0 | 4.630 | no | train 0.0 h < 50.0 h |
| ht | — | — | — | 0.0 | 0.0 | 0.0 | 0.0 | 5.566 | no | train 0.0 h < 50.0 h |
| xh | — | — | — | 0.0 | 0.0 | 0.0 | 0.0 | 5.990 | no | train 0.0 h < 50.0 h |
| st | — | — | — | 0.0 | 0.0 | 0.0 | 0.0 | 3.223 | no | train 0.0 h < 50.0 h |
| dsb | — | — | — | 0.0 | 0.0 | 0.0 | 0.0 | 8.057 | no | train 0.0 h < 50.0 h |
| hr | — | — | — | 0.0 | 0.0 | 0.0 | 0.0 | 4.311 | no | train 0.0 h < 50.0 h |
| nr | — | — | — | 0.0 | 0.0 | 0.0 | 0.0 | 8.671 | no | train 0.0 h < 50.0 h |
| ss | — | — | — | 0.0 | 0.0 | 0.0 | 0.0 | 7.440 | no | train 0.0 h < 50.0 h |
| ts | — | — | — | 0.0 | 0.0 | 0.0 | 0.0 | 8.928 | no | train 0.0 h < 50.0 h |
| ve | — | — | — | 0.0 | 0.0 | 0.0 | 0.0 | 5.733 | no | train 0.0 h < 50.0 h |
