# Languages

Which languages each scale preset trains on, and the numbers behind the choice.
The 3- and 16-language presets come from the design of the ablation and predate
this document; the large preset (`configs/scales/64.yaml`) was selected here.

## The rule

A language is in the large preset when it has

- **at least 50 hours of training audio**, and
- **at least 2 hours of dev audio**, and
- **at least 2 hours of test audio**.

Training audio means **validated minus the evaluation splits** — what a run
actually reads — not the official `train.tsv`. See below for why those are very
different numbers.

**Forty-four of Common Voice 25's 290 locales clear it**, and the preset carries
two more that the smaller presets commit to. The evaluation thresholds do not
bind: every locale over the training threshold has at least 4.4 hours of dev and
of test.

## Two ways to count training audio

Common Voice ships `validated.tsv` — every clip with two or more validations and
more up-votes than down-votes — alongside the `train`/`dev`/`test` partition. The
partition is **not** a partition of `validated`. Quoting the release's own
[documentation](https://github.com/common-voice/cv-dataset/tree/main/datasets/scripted-speech):

> We use the Corpora Creator tool to parse through metadata to generate train,
> dev, and test sets. The Corpora Creator eliminates duplication in clips and
> maximizes for speaker diversity.

> Each train/dev/test set is generated non-deterministically, meaning they will
> vary from release to release even for minor updates.

> Note that total clips in these sets will most probably not add up to the total
> validated clips because of this limitation.

> `validated` -- clips with two or more validations where `up_votes` >
> `down_votes`

The deduplication is the whole story. Corpora Creator splits a frame holding at
most one clip per sentence, so `train.tsv` is roughly one recording per sentence
and is about a third of the validated audio across the release. Japanese has
299,767 validated clips and 19,695 train clips; Russian has 251.9 validated
hours and 38.7 train hours. Counted on `train.tsv`, only 24 locales reach 50
hours and half the 16-language preset falls below the line. Counted on validated
minus dev and test, 44 do.

This repository trains on the second. `configs/base.yaml` sets
`train.cv_train_source: validated_minus_eval`; setting it to `train` reproduces a
run made against the official split, and every run records which it used.

### The speaker guard

Validated minus dev and test is a set difference on *clips*, and that is not
enough. In Corpora Creator (`src/corporacreator/corpus.py`, blob
`da6233c3d76d13627e569763cd6cdcfb32540b80`) the split label is assigned one whole
`client_id` at a time, so train, dev and test are speaker-disjoint from each
other — but the labels are assigned over the deduplicated frame, and the test
split is then truncated with `.head(test_size)`. Both leave clips in
`validated.tsv` that are in no split at all, and some of them belong to the dev
and test speakers. Subtracting dev and test by clip path alone would train the
model on the voices it is about to be scored against.

So the loader also drops every validated clip whose `client_id` appears in dev or
test, and `svb data-stats` reports per language how many clips, how many
speakers and how many hours that removed. The release documentation promises
nothing about this, which is why the guard is a filter and not an assertion. A
release that ships no `client_id` column is refused rather than trained on
without it.

The same fact makes the wider source a superset of the narrower one: no training
speaker is a dev or test speaker, so the guard never removes a row the official
split had kept. That is asserted in the tests rather than assumed.

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
python scripts/select_languages.py --release cv25.json --table-out docs/languages.md
```

Writing over an existing preset needs `--preset-out` *and* `--write-preset`, so
the script can be run to see what a release would change without changing it.

### How a split's hours are computed

The release publishes a clip count per bucket, the validated hours (`validHrs`)
and **one** mean clip duration per locale (`avgDurationSecs`). It publishes no
per-split duration, so

```
split hours    = buckets.<split> x avgDurationSecs / 3600
trainable hours = validHrs - dev hours - test hours
```

Every hour figure below is one of those. `validated h` is the release's own
`validHrs`; `train h` is the official split, kept for comparison against
published Common Voice tables; `trainable h` is what the rule is applied to.

`trainable h` is an **upper bound**. The release document carries no speaker
information, so it cannot subtract what the speaker guard removes. The measured
figure comes from `svb data-stats` against a corpus on disk, which reads the
split files themselves and reports the guard's cost per language. Check the
selection against that before committing a large run.

## Why the large preset is not 64 languages

Forty-four locales qualify. Two more — Finnish and Hindi — are in the preset
because the 16-language preset commits to them and dropping them would stop the
smaller tiers from being subsets of the larger one.

Holding the evaluation thresholds at 2 hours and sweeping the training
threshold:

| min trainable hours | locales selected |
|---:|---:|
| 100 | 34 |
| 80 | 37 |
| **50 (the rule)** | **46** |
| 40 | 49 |
| 30 | 53 |
| 20 | 59 |
| **16** | **64** |
| 10 | 76 |

A 64-language tier is now reachable at a 16-hour training threshold. That is a
real option rather than the 5-hour threshold the official-split count would have
needed, and it is a decision about what the paper wants its largest scale to
mean, not a data question. This file does not make it.

## Pending preset additions

`configs/scales/64.yaml` still holds the 32 languages selected under the old
`train.tsv` statistic. It has not been regenerated, because a wider-corpus
option is being surveyed separately and regenerating twice would churn the
preset. Every language in the preset today is still selected by the rule, so the
committed preset is a subset of what is marked included below. These fourteen
are selected and not yet in it:

```yaml
pending_additions: [fa, kbd, lv, zh-CN, yue, pt, th, ckb, cy, ur, kmr, fy-NL, ady, cs]
```

Four of them — Chinese, Cantonese, Thai and the Japanese already in the preset —
are written without spaces, so a regeneration would add three more languages
reported on character error rate as their primary metric.

## Three rules that override the arithmetic

**Held-out transfer languages are excluded however large.** Malayalam, Marathi,
Telugu and Gujarati are held out for the transfer experiment
(`configs/scales/heldout.yaml`, sourced from OpenSLR). Three of them also exist
in Common Voice (`ml`, `mr`, `te`); training on the Common Voice half would make
the held-out number measure transfer to a language the pipeline had already
supervised. None comes close to the threshold — Marathi, the largest, has 18.9
validated hours — so the guard costs nothing today and is there for the release
where it does not. Odia (`or`) is **not** on that list, because it is not in the
held-out set; if the pending Odia decision in `docs/data.md` ever adds it, this
exclusion list has to grow with it.

**A language the smaller presets commit to is kept however small.** Under the
official split, eight of the sixteen fell below 50 hours. On the training set a
run actually reads, two do:

| locale | language | trainable h | train h | dev h | test h | validated h |
|---|---|---:|---:|---:|---:|---:|
| fi | Finnish | 11.1 | 2.7 | 2.3 | 2.3 | 15.8 |
| hi | Hindi | 7.0 | 6.9 | 3.9 | 4.7 | 15.6 |

Both are small enough that a per-language result for them is a small-sample
result, and Hindi is in the 3-language preset too. Neither was changed here —
the smaller presets are part of a design already committed to — but the numbers
belong beside every result they produce. The other six recovered: Turkish 104.0,
Russian 222.4, Ukrainian 74.8, Arabic 67.9, Polish 151.0 and Japanese 349.5
trainable hours.

**Where one language appears under several locales, only the largest stands for
it.** Keeping two variants would weight that language twice in a mix whose point
is breadth across languages. Under the official split this rule changed nothing;
on the wider pool it fires — `zh-HK` (95.6 h) and `zh-TW` (70.1 h) both clear the
threshold and are dropped in favour of `zh-CN` (212.3 h).

Esperanto (`eo`, 1390.7 trainable hours) is kept. It is a constructed language
rather than one with a speech community of native speakers, which is worth a
sentence in a write-up, but it is a real Common Voice language with real
recorded speech and the rule has no clause that would exclude it.

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
macrolanguage `chm` (`chm_Cyrl_RU`), and none for `kmr`, so Northern Kurdish
takes CLDR's `ku` (`ku_Latn_TR`).

Of the 46 selected, four are written without spaces: Japanese (`Jpan`), Chinese
(`Hans`), Cantonese (`Hant`) and Thai (`Thai`). Uzbek resolves to `uz_Latn_UZ`,
so the Latin orthography, not the Cyrillic one.

The same script decides which text-normalization policy a language gets, since
the reason a policy exists is a property of the writing system. Each preset
states the policy on the language's own line; the assignments and their sources
are in [`normalization.md`](normalization.md).

| Script | Languages in the presets | Policy |
|---|---|---|
| `Latn` (European) | en de fr es it nl pl fi ca eo eu hu gl | `whisper-basic` |
| `Latn` (Turkic) | tr | `turkic-tr` |
| `Latn` (other) | sw rw lg kab uz | `latin-marks` |
| `Cyrl` | ru uk be ab ba mhr | `whisper-basic` |
| `Geor` | ka | `whisper-basic` |
| `Deva` `Taml` `Mlym` `Telu` `Gujr` | hi ta, held-out ml mr te gu | `indic-vistaar` |
| `Arab` | ar / ps / ug | `arabic-ouaal` / `perso-arabic` / `uyghur-ug` |
| `Jpan` | ja | `ja-cer` |

Turkish is the one European-script language not on `whisper-basic`, and the
Arabic-script three each take a different policy: the family's conventions fold
letters in opposite directions, so there is no single Arabic-script answer.

## Typological spread

The point of the large tier is not the largest corpora, which would be almost
entirely Indo-European. The selected set covers **13 top-level families across
24 branches** and **10 scripts**:

| | |
|---|---|
| Families | Indo-European 23, Turkic 4, Atlantic-Congo 3, Northwest Caucasian 3, Uralic 3, Afro-Asiatic 2, Sino-Tibetan 2, Kartvelian 1, Dravidian 1, Kra-Dai 1, Japonic 1, isolate (Basque) 1, constructed (Esperanto) 1 |
| Scripts | Latin 25, Cyrillic 8, Arabic 6, Japanese 1, Han (Simplified) 1, Han (Traditional) 1, Tamil 1, Thai 1, Georgian 1, Devanagari 1 |

Indo-European is 23 of 46, because Common Voice is what it is. The languages
that carry the merge-versus-typology analysis are the far ones — Georgian,
Tamil, Abkhaz, Kabardian, Adyghe, Kabyle, Uyghur, Pashto, Meadow Mari,
Kinyarwanda, Luganda, Basque, Thai, Cantonese — and all of them are here on the
strength of the rule, not by being reached for.

Family labels are the conventional genealogical ones and nothing computes on
them. The script codes are the load-bearing field, and those are CLDR-resolved.

## Every locale considered

All 290 locales of Common Voice 25, ordered by trainable hours. Family and
script are filled in only for selected locales: curating them for all 290 would
be 290 claims nothing in this repository checks.

| locale | language | family | script | trainable h | train h | dev h | test h | validated h | avg clip s | included | reason |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ca | Catalan | Indo-European (Romance) | Latn | 3312.4 | 1758.3 | 23.7 | 23.7 | 3359.8 | 5.194 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| ps | Pashto | Indo-European (Iranian) | Arab | 3008.0 | 239.0 | 16.9 | 16.9 | 3041.9 | 3.943 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| en | English | Indo-European (Germanic) | Latn | 2703.6 | 1679.0 | 24.0 | 24.0 | 2751.6 | 5.266 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| rw | Kinyarwanda | Atlantic-Congo (Bantu) | Latn | 1956.8 | 1395.1 | 22.2 | 22.5 | 2001.6 | 5.007 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| be | Belarusian | Indo-European (Slavic) | Cyrl | 1773.7 | 462.9 | 21.1 | 21.1 | 1816.0 | 4.793 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| eo | Esperanto | Constructed | Latn | 1390.7 | 244.6 | 25.2 | 25.2 | 1441.1 | 6.078 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| de | German | Indo-European (Germanic) | Latn | 1341.4 | 908.4 | 23.7 | 23.7 | 1388.8 | 5.265 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| fr | French | Indo-European (Romance) | Latn | 1050.5 | 858.1 | 22.7 | 22.7 | 1095.9 | 5.036 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| es | Spanish | Indo-European (Romance) | Latn | 550.2 | 485.7 | 21.6 | 21.6 | 593.3 | 4.880 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| kab | Kabyle | Afro-Asiatic (Berber) | Latn | 542.9 | 141.6 | 13.9 | 13.9 | 570.7 | 3.342 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| eu | Basque | Isolate | Latn | 427.7 | 203.1 | 22.4 | 22.4 | 472.4 | 5.438 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| ug | Uyghur | Turkic (Karluk) | Arab | 402.2 | 209.4 | 24.2 | 24.2 | 450.6 | 5.922 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| lg | Luganda | Atlantic-Congo (Bantu) | Latn | 393.9 | 114.2 | 21.5 | 21.5 | 436.8 | 5.784 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| sw | Swahili | Atlantic-Congo (Bantu) | Latn | 356.4 | 68.1 | 17.9 | 17.9 | 392.1 | 5.246 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| ja | Japanese | Japonic | Jpan | 349.5 | 24.4 | 11.2 | 11.2 | 371.9 | 4.467 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| fa | Persian | Indo-European (Iranian) | Arab | 349.5 | 33.0 | 11.7 | 11.7 | 372.9 | 3.942 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| it | Italian | Indo-European (Romance) | Latn | 317.1 | 261.9 | 22.9 | 22.9 | 362.9 | 5.433 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| gl | Galician | Indo-European (Romance) | Latn | 265.7 | 256.2 | 21.1 | 21.2 | 308.0 | 4.988 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| kbd | Kabardian | Northwest Caucasian | Cyrl | 246.0 | 24.0 | 13.1 | 13.1 | 272.2 | 6.223 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| lv | Latvian | Indo-European (Baltic) | Latn | 244.5 | 19.7 | 10.5 | 10.5 | 265.4 | 4.810 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| mhr | Meadow Mari | Uralic (Mari) | Cyrl | 242.6 | 239.5 | 18.8 | 19.5 | 280.9 | 4.622 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| ba | Bashkir | Turkic (Kipchak) | Cyrl | 223.0 | 146.5 | 17.9 | 17.9 | 258.8 | 4.427 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| ru | Russian | Indo-European (Slavic) | Cyrl | 222.4 | 38.7 | 14.8 | 14.8 | 251.9 | 5.179 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| zh-CN | Chinese (Mandarin) | Sino-Tibetan (Sinitic) | Hans | 212.3 | 37.3 | 13.4 | 13.4 | 239.1 | 4.539 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| yue | Cantonese | Sino-Tibetan (Sinitic) | Hant | 199.4 | 8.2 | 5.6 | 5.6 | 210.7 | 3.961 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| ta | Tamil | Dravidian | Taml | 192.9 | 79.9 | 20.9 | 21.0 | 234.8 | 6.182 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| pt | Portuguese | Indo-European (Romance) | Latn | 164.8 | 26.9 | 11.2 | 11.2 | 187.3 | 4.188 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| ab | Abkhaz | Northwest Caucasian | Cyrl | 164.1 | 148.8 | 21.6 | 21.7 | 207.4 | 5.505 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| pl | Polish | Indo-European (Slavic) | Latn | 151.0 | 32.3 | 12.8 | 12.8 | 176.6 | 4.572 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| th | Thai | Kra-Dai (Tai) | Thai | 147.5 | 38.4 | 12.9 | 12.9 | 173.3 | 4.193 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| ka | Georgian | Kartvelian | Geor | 130.3 | 89.7 | 18.6 | 18.7 | 167.5 | 5.121 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| ckb | Central Kurdish | Indo-European (Iranian) | Arab | 124.8 | 9.0 | 6.1 | 6.1 | 137.0 | 4.095 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| cy | Welsh | Indo-European (Celtic) | Latn | 109.3 | 11.0 | 7.4 | 7.4 | 124.1 | 4.916 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| tr | Turkish | Turkic (Oghuz) | Latn | 104.0 | 43.6 | 12.6 | 12.6 | 129.2 | 3.850 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| nl | Dutch | Indo-European (Germanic) | Latn | 96.5 | 56.3 | 14.9 | 14.9 | 126.2 | 4.371 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| zh-HK | — | — | — | 95.6 | 9.7 | 6.5 | 6.5 | 108.5 | 4.151 | no | same language as zh-CN, which has more trainable audio |
| hu | Hungarian | Uralic (Ugric) | Latn | 92.6 | 91.4 | 19.9 | 20.0 | 132.6 | 5.541 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| uk | Ukrainian | Indo-European (Slavic) | Cyrl | 74.8 | 35.6 | 13.4 | 13.4 | 101.6 | 4.649 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| uz | Uzbek | Turkic (Karluk) | Latn | 72.5 | 56.5 | 14.2 | 14.3 | 101.0 | 4.159 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| zh-TW | — | — | — | 70.1 | 6.9 | 4.8 | 4.8 | 79.7 | 3.362 | no | same language as zh-CN, which has more trainable audio |
| ar | Arabic | Afro-Asiatic (Semitic) | Arab | 67.9 | 33.4 | 11.8 | 12.1 | 91.9 | 4.162 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| ur | Urdu | Indo-European (Indo-Aryan) | Arab | 67.8 | 8.6 | 6.0 | 6.0 | 79.7 | 4.230 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| kmr | Northern Kurdish | Indo-European (Iranian) | Latn | 66.0 | 6.5 | 4.8 | 4.8 | 75.7 | 4.161 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| fy-NL | West Frisian | Indo-European (Germanic) | Latn | 61.9 | 5.3 | 4.3 | 4.3 | 70.4 | 4.852 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| ady | Adyghe | Northwest Caucasian | Cyrl | 60.7 | 6.2 | 4.9 | 4.9 | 70.5 | 5.217 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| cs | Czech | Indo-European (Slavic) | Latn | 57.6 | 27.6 | 11.7 | 11.7 | 81.1 | 4.460 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| sk | — | — | — | 42.9 | 11.6 | 6.6 | 7.1 | 56.6 | 4.296 | no | trainable 42.9 h < 50.0 h |
| et | — | — | — | 40.9 | 6.5 | 5.4 | 5.4 | 51.7 | 6.717 | no | trainable 40.9 h < 50.0 h |
| mn | — | — | — | 40.6 | 3.0 | 2.6 | 2.6 | 45.8 | 4.892 | no | trainable 40.6 h < 50.0 h |
| sv-SE | — | — | — | 35.4 | 9.2 | 6.1 | 6.2 | 47.7 | 4.024 | no | trainable 35.4 h < 50.0 h |
| ky | — | — | — | 34.8 | 2.3 | 2.0 | 2.0 | 38.9 | 4.553 | no | trainable 34.8 h < 50.0 h |
| bn | — | — | — | 31.5 | 26.2 | 11.4 | 11.4 | 54.3 | 4.371 | no | trainable 31.5 h < 50.0 h |
| dv | — | — | — | 31.4 | 3.8 | 3.2 | 3.2 | 37.8 | 5.120 | no | trainable 31.4 h < 50.0 h |
| qxp | — | — | — | 29.3 | 1.0 | 0.9 | 0.9 | 31.2 | 4.937 | no | trainable 29.3 h < 50.0 h |
| id | — | — | — | 25.7 | 5.5 | 3.8 | 4.1 | 33.6 | 3.991 | no | trainable 25.7 h < 50.0 h |
| kln | — | — | — | 24.9 | 13.9 | 8.1 | 7.6 | 40.6 | 4.530 | no | trainable 24.9 h < 50.0 h |
| br | — | — | — | 24.4 | 4.0 | 3.2 | 3.2 | 30.7 | 3.256 | no | trainable 24.4 h < 50.0 h |
| tt | — | — | — | 22.1 | 9.0 | 4.7 | 5.4 | 32.3 | 3.835 | no | trainable 22.1 h < 50.0 h |
| cv | — | — | — | 21.0 | 2.0 | 1.7 | 1.8 | 24.5 | 5.042 | no | trainable 21.0 h < 50.0 h |
| ltg | — | — | — | 20.4 | 6.3 | 4.9 | 4.9 | 30.1 | 4.775 | no | trainable 20.4 h < 50.0 h |
| mk | — | — | — | 20.1 | 2.8 | 2.4 | 2.4 | 25.0 | 4.807 | no | trainable 20.1 h < 50.0 h |
| luo | — | — | — | 19.3 | 6.1 | 4.1 | 4.1 | 27.5 | 4.873 | no | trainable 19.3 h < 50.0 h |
| nnh | — | — | — | 17.7 | 1.0 | 0.5 | 0.7 | 18.9 | 8.916 | no | trainable 17.7 h < 50.0 h |
| phl | — | — | — | 17.5 | 2.7 | 1.9 | 1.9 | 21.3 | 4.921 | no | trainable 17.5 h < 50.0 h |
| lzz | — | — | — | 17.4 | 6.2 | 4.4 | 4.3 | 26.1 | 4.450 | no | trainable 17.4 h < 50.0 h |
| mrj | — | — | — | 16.9 | 16.7 | 8.5 | 8.3 | 33.7 | 4.193 | no | trainable 16.9 h < 50.0 h |
| mvy | — | — | — | 16.7 | 3.4 | 2.6 | 2.8 | 22.1 | 4.791 | no | trainable 16.7 h < 50.0 h |
| hy-AM | — | — | — | 16.0 | 15.7 | 8.9 | 9.3 | 34.3 | 5.383 | no | trainable 16.0 h < 50.0 h |
| el | — | — | — | 16.0 | 2.2 | 2.0 | 2.0 | 19.9 | 4.152 | no | trainable 16.0 h < 50.0 h |
| ksf | — | — | — | 15.7 | 0.9 | 0.7 | 0.7 | 17.2 | 8.354 | no | trainable 15.7 h < 50.0 h |
| oru | — | — | — | 14.8 | 8.4 | 3.5 | 3.0 | 21.2 | 7.306 | no | trainable 14.8 h < 50.0 h |
| sl | — | — | — | 14.4 | 1.6 | 1.5 | 1.5 | 17.3 | 3.986 | no | trainable 14.4 h < 50.0 h |
| sva | — | — | — | 14.1 | 0.9 | 0.8 | 0.8 | 15.7 | 5.921 | no | trainable 14.1 h < 50.0 h |
| bnm | — | — | — | 14.1 | 0.8 | 0.6 | 0.7 | 15.3 | 7.173 | no | trainable 14.1 h < 50.0 h |
| ro | — | — | — | 13.4 | 5.8 | 4.4 | 4.4 | 22.3 | 4.046 | no | trainable 13.4 h < 50.0 h |
| szy | — | — | — | 12.9 | 0.4 | 0.4 | 0.4 | 13.7 | 5.397 | no | trainable 12.9 h < 50.0 h |
| mr | — | — | — | 12.8 | 3.8 | 3.0 | 3.1 | 18.9 | 6.200 | no | held-out transfer language (marathi); never trained on |
| dag | — | — | — | 12.7 | 1.9 | 1.7 | 1.7 | 16.2 | 4.294 | no | trainable 12.7 h < 50.0 h |
| pwn | — | — | — | 12.7 | 1.0 | 1.0 | 1.0 | 14.6 | 4.873 | no | trainable 12.7 h < 50.0 h |
| nan-tw | — | — | — | 12.6 | 8.5 | 4.4 | 4.7 | 21.8 | 2.651 | no | trainable 12.6 h < 50.0 h |
| ewo | — | — | — | 12.6 | 0.6 | 0.5 | 0.5 | 13.6 | 6.472 | no | trainable 12.6 h < 50.0 h |
| lt | — | — | — | 12.5 | 12.2 | 7.9 | 8.0 | 28.4 | 5.104 | no | trainable 12.5 h < 50.0 h |
| phr | — | — | — | 12.5 | 0.8 | 0.7 | 0.7 | 14.0 | 3.963 | no | trainable 12.5 h < 50.0 h |
| ga-IE | — | — | — | 12.4 | 1.0 | 0.9 | 0.9 | 14.2 | 3.831 | no | trainable 12.4 h < 50.0 h |
| byv | — | — | — | 12.1 | 0.6 | 0.6 | 0.6 | 13.2 | 6.178 | no | trainable 12.1 h < 50.0 h |
| fub | — | — | — | 11.9 | 0.8 | 0.5 | 0.6 | 12.9 | 6.068 | no | trainable 11.9 h < 50.0 h |
| gwt | — | — | — | 11.9 | 5.0 | 0.0 | 0.3 | 12.2 | 5.722 | no | trainable 11.9 h < 50.0 h |
| ajg | — | — | — | 11.8 | 1.2 | 0.4 | 0.5 | 12.7 | 2.389 | no | trainable 11.8 h < 50.0 h |
| pcm | — | — | — | 11.4 | 0.5 | 0.5 | 0.5 | 12.5 | 5.808 | no | trainable 11.4 h < 50.0 h |
| dua | — | — | — | 11.4 | 0.7 | 0.6 | 0.6 | 12.5 | 6.424 | no | trainable 11.4 h < 50.0 h |
| mxu | — | — | — | 11.3 | 0.5 | 0.5 | 0.5 | 12.3 | 5.733 | no | trainable 11.3 h < 50.0 h |
| bbj | — | — | — | 11.3 | 0.7 | 0.5 | 0.5 | 12.3 | 6.110 | no | trainable 11.3 h < 50.0 h |
| qxw | — | — | — | 11.1 | 0.9 | 0.2 | 0.3 | 11.7 | 5.250 | no | trainable 11.1 h < 50.0 h |
| fi | Finnish | Uralic (Finnic) | Latn | 11.1 | 2.7 | 2.3 | 2.3 | 15.8 | 4.629 | yes | below the rule (trainable 11.1 h < 50.0 h); kept because the smaller presets commit to it |
| qup | — | — | — | 11.0 | 0.7 | 0.5 | 0.4 | 11.9 | 5.932 | no | trainable 11.0 h < 50.0 h |
| kxp | — | — | — | 10.8 | 1.8 | 0.0 | 0.2 | 11.0 | 3.403 | no | trainable 10.8 h < 50.0 h |
| lrk | — | — | — | 10.7 | 1.6 | 0.0 | 0.3 | 11.1 | 3.480 | no | trainable 10.7 h < 50.0 h |
| cjk | — | — | — | 10.7 | 0.6 | 0.6 | 0.6 | 12.0 | 5.874 | no | trainable 10.7 h < 50.0 h |
| nlv | — | — | — | 10.7 | 0.8 | 0.4 | 0.5 | 11.6 | 6.272 | no | trainable 10.7 h < 50.0 h |
| bft | — | — | — | 10.6 | 3.6 | 3.0 | 3.0 | 16.6 | 5.928 | no | trainable 10.6 h < 50.0 h |
| bkm | — | — | — | 10.5 | 0.5 | 0.5 | 0.5 | 11.4 | 5.425 | no | trainable 10.5 h < 50.0 h |
| sbn | — | — | — | 10.5 | 1.7 | 0.0 | 0.2 | 10.7 | 3.524 | no | trainable 10.5 h < 50.0 h |
| qvi | — | — | — | 10.5 | 0.5 | 0.4 | 0.5 | 11.4 | 4.427 | no | trainable 10.5 h < 50.0 h |
| trw | — | — | — | 10.5 | 4.3 | 3.1 | 2.9 | 16.5 | 5.373 | no | trainable 10.5 h < 50.0 h |
| xmf | — | — | — | 10.5 | 0.6 | 0.6 | 0.6 | 11.6 | 6.156 | no | trainable 10.5 h < 50.0 h |
| fue | — | — | — | 10.4 | 1.3 | 0.0 | 0.1 | 10.6 | 5.332 | no | trainable 10.4 h < 50.0 h |
| mhk | — | — | — | 10.4 | 0.8 | 0.5 | 0.3 | 11.3 | 6.149 | no | trainable 10.4 h < 50.0 h |
| ssi | — | — | — | 10.4 | 1.9 | 0.0 | 0.2 | 10.5 | 3.721 | no | trainable 10.4 h < 50.0 h |
| abb | — | — | — | 10.3 | 0.6 | 0.4 | 0.4 | 11.2 | 5.109 | no | trainable 10.3 h < 50.0 h |
| ydg | — | — | — | 10.3 | 1.4 | 0.0 | 0.4 | 10.6 | 3.596 | no | trainable 10.3 h < 50.0 h |
| qws | — | — | — | 10.2 | 1.1 | 0.0 | 0.1 | 10.3 | 4.164 | no | trainable 10.2 h < 50.0 h |
| bci | — | — | — | 10.1 | 0.6 | 0.5 | 0.6 | 11.3 | 7.210 | no | trainable 10.1 h < 50.0 h |
| tok | — | — | — | 10.1 | 3.2 | 2.7 | 2.7 | 15.6 | 4.310 | no | trainable 10.1 h < 50.0 h |
| khw | — | — | — | 10.1 | 5.0 | 3.0 | 2.9 | 16.0 | 6.777 | no | trainable 10.1 h < 50.0 h |
| bag | — | — | — | 10.1 | 0.7 | 0.4 | 0.5 | 11.0 | 5.636 | no | trainable 10.1 h < 50.0 h |
| qus | — | — | — | 10.1 | 0.4 | 0.3 | 0.3 | 10.8 | 3.749 | no | trainable 10.1 h < 50.0 h |
| sah | — | — | — | 10.1 | 4.1 | 3.1 | 3.2 | 16.3 | 6.336 | no | trainable 10.1 h < 50.0 h |
| dar | — | — | — | 10.1 | 3.2 | 2.3 | 2.2 | 14.5 | 5.703 | no | trainable 10.1 h < 50.0 h |
| jgo | — | — | — | 10.1 | 0.7 | 0.6 | 0.6 | 11.3 | 6.695 | no | trainable 10.1 h < 50.0 h |
| mcf | — | — | — | 10.0 | 0.5 | 0.0 | 0.2 | 10.2 | 2.656 | no | trainable 10.0 h < 50.0 h |
| qur | — | — | — | 10.0 | 0.9 | 0.0 | 0.0 | 10.0 | 3.470 | no | trainable 10.0 h < 50.0 h |
| mbo | — | — | — | 10.0 | 0.6 | 0.5 | 0.4 | 10.9 | 5.513 | no | trainable 10.0 h < 50.0 h |
| gig | — | — | — | 10.0 | 1.6 | 0.0 | 0.1 | 10.1 | 2.992 | no | trainable 10.0 h < 50.0 h |
| fmp | — | — | — | 9.9 | 0.8 | 0.7 | 0.7 | 11.4 | 7.865 | no | trainable 9.9 h < 50.0 h |
| plk | — | — | — | 9.9 | 3.5 | 1.7 | 0.9 | 12.6 | 5.026 | no | trainable 9.9 h < 50.0 h |
| mdd | — | — | — | 9.9 | 1.6 | 0.0 | 0.1 | 10.0 | 6.498 | no | trainable 9.9 h < 50.0 h |
| nmz | — | — | — | 9.9 | 0.7 | 0.7 | 0.7 | 11.2 | 2.947 | no | trainable 9.9 h < 50.0 h |
| mki | — | — | — | 9.9 | 1.8 | 0.0 | 0.0 | 9.9 | 3.194 | no | trainable 9.9 h < 50.0 h |
| gej | — | — | — | 9.9 | 0.9 | 0.6 | 0.6 | 11.1 | 2.444 | no | trainable 9.9 h < 50.0 h |
| ncx | — | — | — | 9.9 | 0.4 | 0.4 | 0.4 | 10.7 | 4.457 | no | trainable 9.9 h < 50.0 h |
| haz | — | — | — | 9.8 | 1.0 | 0.1 | 0.6 | 10.5 | 4.556 | no | trainable 9.8 h < 50.0 h |
| qxt | — | — | — | 9.8 | 0.7 | 0.1 | 0.4 | 10.3 | 4.290 | no | trainable 9.8 h < 50.0 h |
| ia | — | — | — | 9.8 | 5.7 | 2.2 | 2.2 | 14.3 | 4.202 | no | trainable 9.8 h < 50.0 h |
| bri | — | — | — | 9.8 | 0.8 | 0.2 | 0.4 | 10.4 | 4.241 | no | trainable 9.8 h < 50.0 h |
| bba | — | — | — | 9.7 | 0.5 | 0.4 | 0.4 | 10.6 | 6.068 | no | trainable 9.7 h < 50.0 h |
| qxu | — | — | — | 9.7 | 0.8 | 0.0 | 0.4 | 10.1 | 4.186 | no | trainable 9.7 h < 50.0 h |
| bax | — | — | — | 9.7 | 0.5 | 0.4 | 0.5 | 10.6 | 4.917 | no | trainable 9.7 h < 50.0 h |
| qvj | — | — | — | 9.7 | 0.6 | 0.6 | 0.6 | 10.8 | 5.961 | no | trainable 9.7 h < 50.0 h |
| btv | — | — | — | 9.6 | 0.5 | 0.3 | 0.4 | 10.3 | 4.057 | no | trainable 9.6 h < 50.0 h |
| kw | — | — | — | 9.6 | 7.1 | 0.0 | 2.7 | 12.4 | 4.121 | no | trainable 9.6 h < 50.0 h |
| hux | — | — | — | 9.6 | 0.7 | 0.0 | 0.4 | 10.0 | 3.876 | no | trainable 9.6 h < 50.0 h |
| wes | — | — | — | 9.6 | 0.4 | 0.4 | 0.4 | 10.3 | 4.118 | no | trainable 9.6 h < 50.0 h |
| gju | — | — | — | 9.5 | 2.9 | 0.0 | 0.6 | 10.1 | 3.274 | no | trainable 9.5 h < 50.0 h |
| tay | — | — | — | 9.5 | 2.0 | 0.7 | 1.3 | 11.5 | 5.572 | no | trainable 9.5 h < 50.0 h |
| an | — | — | — | 9.5 | 5.3 | 3.6 | 3.7 | 16.9 | 4.574 | no | trainable 9.5 h < 50.0 h |
| xka | — | — | — | 9.5 | 1.4 | 0.0 | 0.4 | 9.8 | 3.159 | no | trainable 9.5 h < 50.0 h |
| nmg | — | — | — | 9.5 | 0.9 | 0.5 | 0.5 | 10.4 | 6.401 | no | trainable 9.5 h < 50.0 h |
| mau | — | — | — | 9.4 | 1.0 | 0.4 | 0.5 | 10.4 | 6.216 | no | trainable 9.4 h < 50.0 h |
| cpy | — | — | — | 9.4 | 0.6 | 0.2 | 0.4 | 10.0 | 4.421 | no | trainable 9.4 h < 50.0 h |
| mcx | — | — | — | 9.4 | 1.0 | 0.2 | 0.5 | 10.1 | 6.611 | no | trainable 9.4 h < 50.0 h |
| qxa | — | — | — | 9.4 | 0.5 | 0.3 | 0.4 | 10.1 | 4.343 | no | trainable 9.4 h < 50.0 h |
| kvx | — | — | — | 9.4 | 1.3 | 1.0 | 0.7 | 11.0 | 5.322 | no | trainable 9.4 h < 50.0 h |
| qvl | — | — | — | 9.4 | 0.5 | 0.2 | 0.4 | 10.0 | 3.960 | no | trainable 9.4 h < 50.0 h |
| rof | — | — | — | 9.4 | 0.5 | 0.5 | 0.5 | 10.4 | 3.905 | no | trainable 9.4 h < 50.0 h |
| qwa | — | — | — | 9.4 | 0.8 | 0.1 | 0.4 | 9.9 | 4.951 | no | trainable 9.4 h < 50.0 h |
| mcn | — | — | — | 9.4 | 0.4 | 0.4 | 0.4 | 10.1 | 4.214 | no | trainable 9.4 h < 50.0 h |
| bbl | — | — | — | 9.3 | 1.0 | 0.9 | 0.9 | 11.2 | 8.787 | no | trainable 9.3 h < 50.0 h |
| bum | — | — | — | 9.3 | 0.4 | 0.3 | 0.4 | 10.0 | 4.657 | no | trainable 9.3 h < 50.0 h |
| dmk | — | — | — | 9.3 | 3.1 | 0.0 | 0.9 | 10.2 | 3.445 | no | trainable 9.3 h < 50.0 h |
| odk | — | — | — | 9.3 | 1.7 | 0.8 | 1.1 | 11.2 | 6.369 | no | trainable 9.3 h < 50.0 h |
| bfd | — | — | — | 9.3 | 0.4 | 0.4 | 0.4 | 10.1 | 5.609 | no | trainable 9.3 h < 50.0 h |
| gjk | — | — | — | 9.2 | 1.0 | 0.7 | 0.8 | 10.8 | 4.528 | no | trainable 9.2 h < 50.0 h |
| qva | — | — | — | 9.2 | 0.6 | 0.2 | 0.4 | 9.8 | 4.287 | no | trainable 9.2 h < 50.0 h |
| mgg | — | — | — | 9.2 | 1.0 | 0.6 | 0.4 | 10.2 | 7.580 | no | trainable 9.2 h < 50.0 h |
| giz | — | — | — | 9.1 | 0.6 | 0.5 | 0.4 | 10.1 | 5.572 | no | trainable 9.1 h < 50.0 h |
| mve | — | — | — | 9.1 | 1.7 | 0.7 | 0.2 | 10.1 | 4.739 | no | trainable 9.1 h < 50.0 h |
| beb | — | — | — | 9.1 | 0.5 | 0.5 | 0.5 | 10.0 | 5.347 | no | trainable 9.1 h < 50.0 h |
| mua | — | — | — | 9.0 | 0.5 | 0.3 | 0.4 | 9.7 | 4.006 | no | trainable 9.0 h < 50.0 h |
| prq | — | — | — | 9.0 | 0.5 | 0.4 | 0.4 | 9.7 | 4.726 | no | trainable 9.0 h < 50.0 h |
| zoc | — | — | — | 9.0 | 0.6 | 0.5 | 0.5 | 10.1 | 4.083 | no | trainable 9.0 h < 50.0 h |
| gwc | — | — | — | 9.0 | 5.1 | 1.2 | 1.4 | 11.6 | 5.651 | no | trainable 9.0 h < 50.0 h |
| bas | — | — | — | 9.0 | 2.3 | 1.4 | 1.7 | 12.1 | 3.909 | no | trainable 9.0 h < 50.0 h |
| sei | — | — | — | 9.0 | 0.9 | 0.5 | 0.6 | 10.1 | 4.535 | no | trainable 9.0 h < 50.0 h |
| cux | — | — | — | 9.0 | 1.3 | 0.7 | 0.6 | 10.3 | 4.101 | no | trainable 9.0 h < 50.0 h |
| bce | — | — | — | 9.0 | 0.5 | 0.5 | 0.5 | 10.0 | 5.892 | no | trainable 9.0 h < 50.0 h |
| cut | — | — | — | 9.0 | 0.7 | 0.5 | 0.6 | 10.1 | 6.631 | no | trainable 9.0 h < 50.0 h |
| tar | — | — | — | 8.9 | 0.5 | 0.5 | 0.5 | 10.0 | 4.549 | no | trainable 8.9 h < 50.0 h |
| jqr | — | — | — | 8.9 | 0.7 | 0.5 | 0.4 | 9.9 | 5.898 | no | trainable 8.9 h < 50.0 h |
| bsk | — | — | — | 8.9 | 1.4 | 0.4 | 0.9 | 10.2 | 4.262 | no | trainable 8.9 h < 50.0 h |
| eto | — | — | — | 8.9 | 0.3 | 0.3 | 0.3 | 9.4 | 3.225 | no | trainable 8.9 h < 50.0 h |
| bkh | — | — | — | 8.9 | 0.7 | 0.5 | 0.5 | 10.0 | 6.400 | no | trainable 8.9 h < 50.0 h |
| qux | — | — | — | 8.9 | 0.6 | 0.4 | 0.5 | 9.8 | 5.721 | no | trainable 8.9 h < 50.0 h |
| lss | — | — | — | 8.9 | 0.9 | 0.5 | 0.6 | 9.9 | 3.463 | no | trainable 8.9 h < 50.0 h |
| yaq | — | — | — | 8.8 | 2.8 | 0.2 | 1.2 | 10.2 | 5.311 | no | trainable 8.8 h < 50.0 h |
| hem | — | — | — | 8.8 | 0.6 | 0.6 | 0.6 | 9.9 | 5.999 | no | trainable 8.8 h < 50.0 h |
| pua | — | — | — | 8.8 | 1.6 | 0.7 | 0.7 | 10.2 | 4.885 | no | trainable 8.8 h < 50.0 h |
| gid | — | — | — | 8.8 | 0.6 | 0.6 | 0.6 | 9.9 | 7.000 | no | trainable 8.8 h < 50.0 h |
| gya | — | — | — | 8.8 | 0.5 | 0.5 | 0.5 | 9.8 | 5.078 | no | trainable 8.8 h < 50.0 h |
| kdh | — | — | — | 8.7 | 0.3 | 0.2 | 0.2 | 9.2 | 2.448 | no | trainable 8.7 h < 50.0 h |
| fan | — | — | — | 8.6 | 0.4 | 0.4 | 0.4 | 9.4 | 4.399 | no | trainable 8.6 h < 50.0 h |
| udl | — | — | — | 8.6 | 0.6 | 0.4 | 0.5 | 9.5 | 5.352 | no | trainable 8.6 h < 50.0 h |
| nyu | — | — | — | 8.5 | 2.6 | 0.0 | 0.6 | 9.2 | 9.059 | no | trainable 8.5 h < 50.0 h |
| hno | — | — | — | 8.5 | 1.0 | 0.9 | 0.8 | 10.2 | 4.009 | no | trainable 8.5 h < 50.0 h |
| xhe | — | — | — | 8.4 | 3.0 | 0.0 | 1.2 | 9.6 | 3.032 | no | trainable 8.4 h < 50.0 h |
| tvu | — | — | — | 8.4 | 1.4 | 1.0 | 0.9 | 10.2 | 7.009 | no | trainable 8.4 h < 50.0 h |
| var | — | — | — | 8.4 | 1.1 | 0.8 | 0.9 | 10.1 | 5.307 | no | trainable 8.4 h < 50.0 h |
| trv | — | — | — | 8.3 | 1.4 | 0.9 | 0.8 | 9.9 | 5.611 | no | trainable 8.3 h < 50.0 h |
| nla | — | — | — | 8.2 | 1.1 | 0.3 | 0.4 | 8.9 | 6.930 | no | trainable 8.2 h < 50.0 h |
| tui | — | — | — | 8.1 | 0.9 | 0.8 | 0.8 | 9.7 | 4.634 | no | trainable 8.1 h < 50.0 h |
| bsh | — | — | — | 8.1 | 2.1 | 0.8 | 1.0 | 9.9 | 5.264 | no | trainable 8.1 h < 50.0 h |
| tli | — | — | — | 8.1 | 8.1 | 0.0 | 1.8 | 9.9 | 12.604 | no | trainable 8.1 h < 50.0 h |
| scl | — | — | — | 7.9 | 1.6 | 1.0 | 1.1 | 10.0 | 4.072 | no | trainable 7.9 h < 50.0 h |
| wbl | — | — | — | 7.9 | 4.8 | 2.1 | 2.1 | 12.1 | 6.753 | no | trainable 7.9 h < 50.0 h |
| lua | — | — | — | 7.8 | 0.6 | 0.5 | 0.5 | 8.9 | 6.664 | no | trainable 7.8 h < 50.0 h |
| bg | — | — | — | 7.6 | 7.6 | 4.5 | 5.2 | 17.3 | 5.475 | no | trainable 7.6 h < 50.0 h |
| bnn | — | — | — | 7.6 | 1.5 | 1.4 | 1.4 | 10.3 | 5.090 | no | trainable 7.6 h < 50.0 h |
| yav | — | — | — | 7.5 | 0.8 | 0.5 | 0.6 | 8.6 | 6.655 | no | trainable 7.5 h < 50.0 h |
| kls | — | — | — | 7.5 | 1.5 | 1.3 | 1.3 | 10.0 | 3.699 | no | trainable 7.5 h < 50.0 h |
| gv | — | — | — | 7.4 | 3.9 | 1.9 | 0.8 | 10.1 | 5.782 | no | trainable 7.4 h < 50.0 h |
| dru | — | — | — | 7.4 | 1.7 | 1.5 | 1.5 | 10.4 | 5.676 | no | trainable 7.4 h < 50.0 h |
| dml | — | — | — | 7.3 | 5.3 | 1.8 | 1.0 | 10.2 | 6.019 | no | trainable 7.3 h < 50.0 h |
| brh | — | — | — | 7.2 | 3.3 | 1.0 | 1.7 | 9.9 | 7.019 | no | trainable 7.2 h < 50.0 h |
| esu | — | — | — | 7.1 | 6.5 | 0.0 | 0.5 | 7.6 | 3.800 | no | trainable 7.1 h < 50.0 h |
| ggg | — | — | — | 7.0 | 1.9 | 0.0 | 0.4 | 7.4 | 3.988 | no | trainable 7.0 h < 50.0 h |
| hi | Hindi | Indo-European (Indo-Aryan) | Deva | 7.0 | 6.9 | 3.9 | 4.7 | 15.6 | 5.045 | yes | below the rule (trainable 7.0 h < 50.0 h); kept because the smaller presets commit to it |
| ipk | — | — | — | 6.8 | 6.8 | 0.0 | 0.4 | 7.2 | 7.938 | no | trainable 6.8 h < 50.0 h |
| da | — | — | — | 6.8 | 4.1 | 3.1 | 3.1 | 13.0 | 4.072 | no | trainable 6.8 h < 50.0 h |
| dav | — | — | — | 6.7 | 2.4 | 1.4 | 1.1 | 9.3 | 4.059 | no | trainable 6.7 h < 50.0 h |
| eko | — | — | — | 6.7 | 1.1 | 0.7 | 0.9 | 8.3 | 7.541 | no | trainable 6.7 h < 50.0 h |
| mse | — | — | — | 6.6 | 0.7 | 0.6 | 0.5 | 7.7 | 6.412 | no | trainable 6.6 h < 50.0 h |
| bgp | — | — | — | 6.6 | 5.3 | 2.1 | 2.8 | 11.5 | 5.418 | no | trainable 6.6 h < 50.0 h |
| ibb | — | — | — | 6.2 | 0.9 | 0.8 | 0.8 | 7.8 | 9.002 | no | trainable 6.2 h < 50.0 h |
| ush | — | — | — | 5.6 | 1.0 | 0.3 | 0.6 | 6.6 | 6.148 | no | trainable 5.6 h < 50.0 h |
| tig | — | — | — | 5.3 | 3.3 | 2.7 | 2.7 | 10.7 | 5.984 | no | trainable 5.3 h < 50.0 h |
| or | — | — | — | 4.6 | 3.3 | 1.0 | 0.7 | 6.3 | 5.539 | no | trainable 4.6 h < 50.0 h |
| mt | — | — | — | 4.4 | 2.5 | 2.1 | 2.2 | 8.7 | 4.747 | no | trainable 4.4 h < 50.0 h |
| mug | — | — | — | 4.2 | 0.8 | 0.6 | 0.6 | 5.4 | 7.273 | no | trainable 4.2 h < 50.0 h |
| sr | — | — | — | 4.2 | 2.3 | 1.7 | 1.8 | 7.6 | 3.261 | no | trainable 4.2 h < 50.0 h |
| vi | — | — | — | 4.2 | 2.1 | 1.5 | 1.6 | 7.3 | 3.997 | no | trainable 4.2 h < 50.0 h |
| sq | — | — | — | 3.8 | 3.8 | 2.5 | 2.7 | 9.0 | 5.088 | no | trainable 3.8 h < 50.0 h |
| he | — | — | — | 3.6 | 2.4 | 0.5 | 1.2 | 5.3 | 4.551 | no | trainable 3.6 h < 50.0 h |
| tn | — | — | — | 3.4 | 1.3 | 0.4 | 0.4 | 4.2 | 4.370 | no | trainable 3.4 h < 50.0 h |
| rm-sursilv | — | — | — | 3.1 | 2.9 | 2.4 | 2.5 | 7.9 | 5.379 | no | trainable 3.1 h < 50.0 h |
| gn | — | — | — | 2.9 | 2.2 | 0.8 | 1.4 | 5.2 | 4.597 | no | trainable 2.9 h < 50.0 h |
| lij | — | — | — | 2.6 | 2.5 | 1.0 | 1.5 | 5.0 | 3.870 | no | trainable 2.6 h < 50.0 h |
| ha | — | — | — | 2.5 | 2.3 | 0.8 | 0.9 | 4.2 | 4.351 | no | trainable 2.5 h < 50.0 h |
| yo | — | — | — | 2.4 | 2.4 | 1.6 | 1.8 | 5.8 | 6.040 | no | trainable 2.4 h < 50.0 h |
| myv | — | — | — | 2.0 | 2.0 | 0.4 | 0.8 | 3.2 | 5.784 | no | trainable 2.0 h < 50.0 h |
| ml | — | — | — | 1.9 | 1.5 | 1.1 | 1.0 | 4.1 | 4.242 | no | held-out transfer language (malayalam); never trained on |
| oc | — | — | — | 1.9 | 0.4 | 0.4 | 0.4 | 2.7 | 4.871 | no | trainable 1.9 h < 50.0 h |
| skr | — | — | — | 1.8 | 1.8 | 1.3 | 1.2 | 4.3 | 4.164 | no | trainable 1.8 h < 50.0 h |
| hsb | — | — | — | 1.7 | 1.7 | 0.7 | 1.0 | 3.4 | 7.503 | no | trainable 1.7 h < 50.0 h |
| as | — | — | — | 1.5 | 1.6 | 0.8 | 0.7 | 3.0 | 5.857 | no | trainable 1.5 h < 50.0 h |
| ebr | — | — | — | 1.5 | 0.7 | 0.0 | 0.3 | 1.8 | 4.167 | no | trainable 1.5 h < 50.0 h |
| nb-NO | — | — | — | 1.5 | 1.5 | 0.5 | 0.4 | 2.3 | 4.193 | no | trainable 1.5 h < 50.0 h |
| tk | — | — | — | 1.5 | 1.1 | 0.8 | 0.8 | 3.1 | 5.509 | no | trainable 1.5 h < 50.0 h |
| sc | — | — | — | 1.4 | 1.2 | 0.7 | 0.9 | 3.0 | 4.705 | no | trainable 1.4 h < 50.0 h |
| pa-IN | — | — | — | 1.1 | 1.1 | 0.7 | 0.7 | 2.4 | 4.806 | no | trainable 1.1 h < 50.0 h |
| yi | — | — | — | 1.0 | 0.5 | 0.5 | 0.5 | 2.0 | 3.975 | no | trainable 1.0 h < 50.0 h |
| ko | — | — | — | 1.0 | 1.0 | 0.7 | 0.8 | 2.5 | 5.211 | no | trainable 1.0 h < 50.0 h |
| ig | — | — | — | 1.0 | 1.0 | 0.9 | 0.9 | 2.7 | 5.428 | no | trainable 1.0 h < 50.0 h |
| am | — | — | — | 1.0 | 1.0 | 0.4 | 0.5 | 1.9 | 6.345 | no | trainable 1.0 h < 50.0 h |
| kk | — | — | — | 0.9 | 0.9 | 0.8 | 0.8 | 2.5 | 4.946 | no | trainable 0.9 h < 50.0 h |
| rm-vallader | — | — | — | 0.9 | 0.9 | 0.8 | 0.8 | 2.5 | 5.832 | no | trainable 0.9 h < 50.0 h |
| cnh | — | — | — | 0.9 | 0.8 | 0.7 | 0.7 | 2.4 | 3.517 | no | trainable 0.9 h < 50.0 h |
| zza | — | — | — | 0.9 | 0.9 | 0.5 | 0.5 | 1.9 | 4.032 | no | trainable 0.9 h < 50.0 h |
| zgh | — | — | — | 0.9 | 0.9 | 0.3 | 0.2 | 1.4 | 3.559 | no | trainable 0.9 h < 50.0 h |
| nn-NO | — | — | — | 0.7 | 0.7 | 0.4 | 0.5 | 1.6 | 4.441 | no | trainable 0.7 h < 50.0 h |
| os | — | — | — | 0.7 | 0.6 | 0.4 | 0.3 | 1.4 | 5.523 | no | trainable 0.7 h < 50.0 h |
| ne-NP | — | — | — | 0.7 | 0.4 | 0.4 | 0.3 | 1.3 | 4.111 | no | trainable 0.7 h < 50.0 h |
| te | — | — | — | 0.6 | 0.1 | 0.1 | 0.1 | 0.8 | 4.195 | no | held-out transfer language (telugu); never trained on |
| ast | — | — | — | 0.6 | 0.5 | 0.1 | 0.3 | 1.0 | 4.397 | no | trainable 0.6 h < 50.0 h |
| gsw | — | — | — | 0.5 | 0.0 | 0.0 | 0.0 | 0.6 | 5.716 | no | trainable 0.5 h < 50.0 h |
| tg | — | — | — | 0.4 | 0.5 | 0.2 | 0.2 | 0.8 | 5.027 | no | trainable 0.4 h < 50.0 h |
| sat | — | — | — | 0.4 | 0.4 | 0.1 | 0.2 | 0.7 | 4.481 | no | trainable 0.4 h < 50.0 h |
| az | — | — | — | 0.3 | 0.3 | 0.1 | 0.2 | 0.7 | 5.452 | no | trainable 0.3 h < 50.0 h |
| af | — | — | — | 0.3 | 0.3 | 0.2 | 0.2 | 0.8 | 6.080 | no | trainable 0.3 h < 50.0 h |
| sd | — | — | — | 0.3 | 0.3 | 0.0 | 0.0 | 0.4 | 4.070 | no | trainable 0.3 h < 50.0 h |
| mdf | — | — | — | 0.3 | 0.3 | 0.1 | 0.2 | 0.5 | 5.250 | no | trainable 0.3 h < 50.0 h |
| tw | — | — | — | 0.3 | 0.3 | 0.0 | 0.0 | 0.3 | 4.385 | no | trainable 0.3 h < 50.0 h |
| lo | — | — | — | 0.2 | 0.2 | 0.1 | 0.1 | 0.3 | 6.532 | no | trainable 0.2 h < 50.0 h |
| dyu | — | — | — | 0.2 | 0.2 | 0.1 | 0.1 | 0.4 | 6.309 | no | trainable 0.2 h < 50.0 h |
| is | — | — | — | 0.1 | 0.1 | 0.0 | 0.1 | 0.2 | 6.425 | no | trainable 0.1 h < 50.0 h |
| vot | — | — | — | 0.1 | 0.1 | 0.0 | 0.0 | 0.1 | 2.412 | no | trainable 0.1 h < 50.0 h |
| ti | — | — | — | 0.0 | 0.1 | 0.0 | 0.0 | 0.1 | 5.194 | no | trainable 0.0 h < 50.0 h |
| quy | — | — | — | 0.0 | 0.0 | 0.0 | 0.0 | 0.1 | 4.994 | no | trainable 0.0 h < 50.0 h |
| ms | — | — | — | 0.0 | 0.0 | 0.0 | 0.0 | 0.1 | 6.176 | no | trainable 0.0 h < 50.0 h |
| nhi | — | — | — | 0.0 | 0.0 | 0.0 | 0.0 | 0.1 | 5.081 | no | trainable 0.0 h < 50.0 h |
| rup | — | — | — | 0.0 | 0.0 | 0.0 | 0.0 | 0.1 | 7.021 | no | trainable 0.0 h < 50.0 h |
| ht | — | — | — | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 5.566 | no | trainable 0.0 h < 50.0 h |
| zu | — | — | — | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 5.533 | no | trainable 0.0 h < 50.0 h |
| nso | — | — | — | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 4.630 | no | trainable 0.0 h < 50.0 h |
| xh | — | — | — | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 5.990 | no | trainable 0.0 h < 50.0 h |
| dsb | — | — | — | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 8.057 | no | trainable 0.0 h < 50.0 h |
| hr | — | — | — | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 4.311 | no | trainable 0.0 h < 50.0 h |
| nr | — | — | — | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 8.671 | no | trainable 0.0 h < 50.0 h |
| ss | — | — | — | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 7.440 | no | trainable 0.0 h < 50.0 h |
| st | — | — | — | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 3.223 | no | trainable 0.0 h < 50.0 h |
| ts | — | — | — | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 8.928 | no | trainable 0.0 h < 50.0 h |
| ve | — | — | — | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 5.733 | no | trainable 0.0 h < 50.0 h |
