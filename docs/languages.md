# Languages

The large preset is **64 languages**: 46 from Common Voice 25 and 18 from other
public corpora. This is the evidence for each one, and for everything left out.

## The rule

A language is in when it has at least **50 hours of training audio**, at least
**2 hours of dev** and at least **2 hours of test**.

Training audio means what a run actually reads, which differs by source:

| Source | Training hours | Dev and test |
|---|---|---|
| Common Voice | `validHrs` less dev and test — validated minus the evaluation splits | the official splits |
| a corpus that ships a full partition | its published train split | its published dev and test |
| a corpus that ships none, or an unusable one | the total less a tenth each | a tenth each, which the loader derives |

That last row is the loader's own arithmetic stated up front rather than left
for a reader to work out: `svb.data.splits` gives dev and test a tenth each,
speaker-disjointly, so a corpus over the training threshold clears the
evaluation thresholds by construction.

**Fifty-seven of the 64 meet the rule outright.** Seven are below it and say so
here rather than quietly passing; they are listed under *Carried* below.

## Where the numbers come from

Common Voice figures come from Mozilla's own release statistics:

| | |
|---|---|
| Path | `datasets/scripted-speech/cv-corpus-25.0-2026-03-09.json` |
| Repository | [`common-voice/cv-dataset`](https://github.com/common-voice/cv-dataset) |
| Blob SHA | `ac5fe102ab012f3fba02dd2eec1b65ef670826d5` |
| Repository HEAD when read | `f99d8239d2796131b73ac99f92ee7cb4443bf3ba` (2026-06-16) |

A Common Voice split's hours are its clip count times the release's mean clip
duration, because the release publishes no per-split duration. The trainable
figure is an **upper bound**: the release carries no speaker information, so it
cannot subtract what the loader's speaker guard removes. `svb data-stats`
measures the real figure against a corpus on disk.

Every other corpus has a row in [`../configs/corpora.yaml`](../configs/corpora.yaml)
carrying its verbatim licence, source page, downloads, formats and hours, each
read on that source page. Where a page publishes no value the field is null and
named under `unverified` — four corpora have an unstated sample rate and two an
unstated transcript format, and those are checks to make at ingest, not values
to assume.

Regenerate this file and the preset:

```bash
gh api repos/common-voice/cv-dataset/contents/datasets/scripted-speech/cv-corpus-25.0-2026-03-09.json --jq '.content' | base64 -d > cv25.json
python scripts/select_languages.py --release cv25.json --table-out docs/languages.md --preset-out configs/scales/64.yaml --write-preset
```

Regenerating carries existing `normalizer:` values forward, so policy work done
elsewhere is not silently dropped.

## What the 18 additions are

| Language | Code | Corpus | Trainable h | Licence | Family · script |
|---|---|---|---:|---|---|
| Croatian | hr | ParlaSpeech-HR v1.0 | 1452.8 | CC BY-SA 4.0 | Indo-European (Slavic) · Latn |
| Estonian | et | TalTech Estonian 1.0 | 1334.0 | CC BY-SA 4.0 DEED | Uralic (Finnic) · Latn |
| Sundanese | su | SLR36 | 266.4 | CC BY-SA 4.0 | Austronesian · Latn |
| Kazakh | kk | KSC (SLR102) | 265.6 | CC BY 4.0 | Turkic (Kipchak) · Cyrl |
| Kannada | kn | IISc-MILE (SLR126) | 280.0 | CC BY 2.0 | Dravidian · Knda |
| Javanese | jv | SLR35 | 236.8 | CC BY-SA 4.0 | Austronesian · Latn |
| Bengali | bn | SLR53 | 183.2 | CC BY-SA 4.0 | Indo-European (Indo-Aryan) · Beng |
| Sinhala | si | SLR52 | 179.2 | CC BY-SA 4.0 | Indo-European (Indo-Aryan) · Sinh |
| Nepali | ne | SLR54 | 132.0 | CC BY-SA 4.0 | Indo-European (Indo-Aryan) · Deva |
| Icelandic | is | Samrómur 21.05 (SLR112) | 116.0 | CC BY 4.0 | Indo-European (Germanic) · Latn |
| Tibetan | bo | TIBMD@MUC (SLR124) | 67.5 | CC BY-SA 4.0 | Sino-Tibetan · Tibt · **no word boundary** |
| Armenian | hy | SLR160 | 56.0 | CC BY 4.0 | Indo-European (Armenian) · Armn |
| Sepedi | nso | NCHLT | 50.9 | CC BY 3.0 | Atlantic-Congo (Sotho-Tswana) · Latn |
| Xitsonga | ts | NCHLT | 49.8 | CC BY 3.0 | Atlantic-Congo (Tswa-Ronga) · Latn |
| Tshivenda | ve | NCHLT | 49.6 | CC BY 3.0 | Atlantic-Congo (Venda) · Latn |
| isiXhosa | xh | NCHLT | 49.4 | CC BY 3.0 | Atlantic-Congo (Nguni) · Latn |
| isiZulu | zu | NCHLT | 48.5 | CC BY 3.0 | Atlantic-Congo (Nguni) · Latn |
| Korean | ko | Zeroth (SLR40) | 42.2 | CC BY 4.0 | Koreanic · Kore |

Bengali is in Common Voice too, at 31.5 trainable hours — below the rule. SLR53
is what puts the language in the preset, so it is read from there and the
Common Voice half is not used.

**Access.** Every one of these serves anonymously. The NCHLT corpora were the
open question: SADiLaR's resources page mentions accepting terms of use, but its
DSpace bitstream endpoint answers an unauthenticated client, which was tested
rather than assumed. Corpora behind a gate, a form or an email request are not
here at any size.

## Carried: the seven below the rule

| Code | Language | Trainable h | Why it is in anyway |
|---|---|---:|---|
| ts | Xitsonga | 49.8 | NCHLT ships a train split just under the bar against a corpus of about 56 h |
| ve | Tshivenda | 49.6 | as above |
| xh | isiXhosa | 49.4 | as above |
| zu | isiZulu | 48.5 | as above |
| ko | Korean | 42.2 | the only openly-licensed Korean corpus; its shipped 1.2 h test is under the evaluation bar, so all 52.8 h are repartitioned |
| fi | Finnish | 11.1 | the 16-language preset commits to it |
| hi | Hindi | 7.0 | the 3- and 16-language presets commit to it |

Finnish and Hindi are small enough that a per-language result for either is a
small-sample result. Both could have been repaired from other corpora and were
not: Hindi's SLR103 links its licence to a viewer rather than naming it and is
8 kHz M4A, and the Finnish Parliament corpus carries a composite CLARIN tag with
a NoDerivatives term. Neither meets the licence rule, so both stay Common
Voice-only with the caveat attached.

## Domain diversity is narrower than 64 languages sounds

This belongs in any write-up. The additions are almost entirely **read prompts**
(the five Google crowdsourced sets, the five NCHLT languages, Samrómur, Kannada,
Armenian, Tibetan) or **parliamentary speech** (ParlaSpeech-HR, and TalTech is
long-form spontaneous broadcast and meeting audio). NCHLT's prompts are scripted
and its published test side is an 8-speaker suite. So the mix gains a great deal
of typological range and rather little domain range, and a result that improves
on read speech should not be reported as improving on speech.

Licence bookkeeping is a second caveat: the corpora mix CC0, CC BY 2.0, 3.0 and
4.0 and CC BY-SA 4.0. ShareAlike over a training corpus is not settled law with
respect to model weights, and the paper should take a position rather than say
nothing.

## Normalization policies

Every one of the 64 names its policy on its own line in the preset, and the
policy column in the table below is generated rather than written by hand —
a regeneration that dropped it is how the assignments were lost once already.
The rules behind each policy and the sources they follow are in
[`normalization.md`](normalization.md).

| Policy | Languages | |
|---|---|---|
| `whisper-basic` | ab ady ba be ca cs cy de en eo es et eu fi fr fy-NL gl hr hu is it ka kbd kk lv mhr nl pl pt ru uk | 31 |
| `latin-marks` | jv kab kmr lg nso rw su sw ts uz ve xh zu | 13 |
| `indic-vistaar` | bn hi kn ne si ta, and held-out gu ml mr te | 6 + 4 |
| `perso-arabic` | ckb fa ps ur | 4 |
| `han-mer` | yue zh-CN | 2 |
| `arabic-ouaal` `armenian-hy` `ja-cer` `ko-kspon` `thai-cer` `tibetan-syllable` `turkic-tr` `uyghur-ug` | ar, hy, ja, ko, th, bo, tr, ug | 1 each |

The Latin script splits three ways and the Arabic script three ways, because
neither can decide a policy on its own: European Latin uses Whisper's normalizer
and the rest need the mark-preserving one, while the Arabic-script conventions
fold letters in opposite directions.

Six of the eighteen bring a script the text layer had never seen — Tibetan,
Sinhala, Kannada, Bengali, Armenian and Korean — which for a character-CTC model
also grows the output layer. Three of them needed a policy that did not exist:
Armenian writes punctuation inside the word, Korean must never see the
compatibility form, and Tibetan is an abugida written without spaces whose
corpus states no evaluation convention at all, so its policy is reasoned from
the orthography and flagged unverified.

## Writing system and the primary metric

`word_boundary: false` makes character error rate the primary metric, because
whitespace tokenization of a script that does not separate words yields one
token per sentence. It is derived from the script the language is written in,
read from [`unicode-org/cldr`](https://github.com/unicode-org/cldr) at
`common/supplemental/likelySubtags.xml` (blob
`11be52e4d0bce1e7e048b5ddaba2a6563c1f0d85`).

Five of the 64 are written without spaces: Japanese, Chinese, Cantonese, Thai
and now **Tibetan**. Korean is **not** among them — Hangul is written with
spaces between words, so word error rate means the same for it as for a
Latin-script language. CLDR has no entry for `mhr` or `kmr`, so Meadow Mari
takes the script of the Mari macrolanguage `chm` and Northern Kurdish takes
CLDR's `ku`.

## Typological spread

**15 top-level families across 33 branches, and 16 scripts:**

| | |
|---|---|
| Families | Indo-European 29, Atlantic-Congo 8, Turkic 5, Uralic 4, Northwest Caucasian 3, Sino-Tibetan 3, Afro-Asiatic 2, Dravidian 2, Austronesian 2, Japonic 1, Kra-Dai 1, Kartvelian 1, Koreanic 1, isolate (Basque) 1, constructed (Esperanto) 1 |
| Scripts | Latin 35, Cyrillic 9, Arabic 6, Devanagari 2, Japanese 1, Kannada 1, Han (Simplified) 1, Han (Traditional) 1, Tamil 1, Bengali 1, Sinhala 1, Thai 1, Georgian 1, Tibetan 1, Armenian 1, Hangul 1 |

The 18 additions bring five families the Common Voice half does not have at all
— Austronesian, Koreanic, Sino-Tibetan at real branch depth, Dravidian beyond
Tamil, and Atlantic-Congo beyond the three Bantu languages Common Voice
supplies — and six scripts. Indo-European is still 29 of 64, because Common
Voice is what it is.

## Reserve

Verified and qualifying, not selected. If a row above falls over on inspection,
these are the substitutes, in rough order of size:

Swedish (RixVox, 5383 h, CC BY 4.0), Serbian (ParlaSpeech-RS, 896 h — its
licence tag differs between CLARIN and the Hugging Face mirror and would need
resolving first), Slovenian (ARTUR 1.0, 884 h, audio and transcripts under
separate handles), Khmer (Digital-Divide-Data, 727 h but **12 speakers**, which
is why it was not selected), Afrikaans, Sesotho, Setswana, isiNdebele and
Siswati (NCHLT, ~56 h each, same access path as the five chosen), Punjabi and
Sanskrit (Kathbath, 136.9 and 115.5 h, CC0, but m4a needing an ffmpeg decode),
Norwegian (NPSC, 140 h, CC0 audio), Faroese (BLARK 1.0, 100 h, 48 kHz so it
needs resampling), Romanian (VoxPopuli, 89 h, CC0).

## Rejected, and why

Grouped by the criterion each fails. None is counted anywhere above.

**Gated.** IndicVoices and Shrutilipi (AI4Bharat contact-info gate),
NaijaVoices (gate and CC BY-NC-SA), ivrit.ai, ReazonSpeech, MDCC Cantonese
(signed licence by email), Bud500, WenetSpeech (application form), AISHELL-2
(institutional-email form), LaboroTVSpeech, FT Speech Danish (Google Form),
LIEPA-2 Lithuanian (both channels), LATE Latvian, Kaggle Bengali.AI, CSJ (paid),
AI-Hub Korean, LDC and ELRA.

**Licence.** MAGICDATA, Primewords and ST-CMDS (CC BY-NC-ND); Multilingual TEDx
(the NoDerivatives term, not the NonCommercial one); MASRI Maltese; SWARA and
RSC Romanian; CMU Wilderness, whose audio is governed by bible.is terms that
permit personal non-commercial use only.

**Transcripts.** VoxLingua107 (language-ID labels, no transcripts), Greek
Podcast Corpus and both `mesolitica` Malay sets (Whisper or Google-STT
pseudo-labels), WenetSpeech's high-label subset (OCR/ASR-derived).

**Too small.** FLEURS (~12 h per language, which rules out all 102),
MediaSpeech (10 h), THCHS-30, VIVOS, JSUT, Gowajee Thai, CantoMap, NICT-Tib1
(superseded by TIBMD@MUC), ArmSpeech (superseded by SLR160), Bashkir `AigizK`
(and mostly synthetic), Lwazi (telephone bandwidth), BembaSpeech, Kallaama
Wolof, and every OpenSLR "high quality TTS data" set (SLR41–44, 63–66, 78–80),
which are TTS corpora rather than ASR training sets.

**Structurally unusable.** BibleTTS (one speaker per language, so no
speaker-disjoint split exists, and scripture-only), HKCanCor (samples only),
`aidatatang_200zh` (retracted by its owner), MBSpeech Mongolian and ManaTTS
Persian (single speaker).

**Held out by rule.** Malayalam, Marathi, Telugu and Gujarati are the transfer
set and Odia is the pending fifth. None may appear as training data under any
corpus, and because five of the corpora above come from the same crowdsourced
programme as the held-out four, that is enforced by a test rather than by care.

**No qualifying corpus found**, after looking: Azerbaijani, Sakha, Mongolian,
Macedonian, Bosnian, Montenegrin, Manx, Breton, Lao, Indonesian, Malay
(human-transcribed), Tagalog.

## Every locale and corpus considered

All 290 Common Voice locales plus the 18 corpus languages, ordered by trainable
hours. Family and script are filled in only for selected languages: curating
them for all 308 would be 308 claims nothing in this repository checks.

| locale | language | family | script | policy | corpus | trainable h | train h | dev h | test h | validated h | included | reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ca | Catalan | Indo-European (Romance) | Latn | whisper-basic | common_voice_25 | 3312.4 | 1758.3 | 23.7 | 23.7 | 3359.8 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| ps | Pashto | Indo-European (Iranian) | Arab | perso-arabic | common_voice_25 | 3008.0 | 239.0 | 16.9 | 16.9 | 3041.9 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| en | English | Indo-European (Germanic) | Latn | whisper-basic | common_voice_25 | 2703.6 | 1679.0 | 24.0 | 24.0 | 2751.6 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| rw | Kinyarwanda | Atlantic-Congo (Bantu) | Latn | latin-marks | common_voice_25 | 1956.8 | 1395.1 | 22.2 | 22.5 | 2001.6 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| be | Belarusian | Indo-European (Slavic) | Cyrl | whisper-basic | common_voice_25 | 1773.7 | 462.9 | 21.1 | 21.1 | 1816.0 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| hr | Croatian | Indo-European (Slavic) | Latn | whisper-basic | parlaspeech_hr | 1452.8 | 0.0 | 181.6 | 181.6 | 1816.0 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| eo | Esperanto | Constructed | Latn | whisper-basic | common_voice_25 | 1390.7 | 244.6 | 25.2 | 25.2 | 1441.1 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| de | German | Indo-European (Germanic) | Latn | whisper-basic | common_voice_25 | 1341.4 | 908.4 | 23.7 | 23.7 | 1388.8 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| et | Estonian | Uralic (Finnic) | Latn | whisper-basic | taltech_estonian | 1334.0 | 1334.0 | 21.0 | 23.0 | 0.0 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| fr | French | Indo-European (Romance) | Latn | whisper-basic | common_voice_25 | 1050.5 | 858.1 | 22.7 | 22.7 | 1095.9 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| es | Spanish | Indo-European (Romance) | Latn | whisper-basic | common_voice_25 | 550.2 | 485.7 | 21.6 | 21.6 | 593.3 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| kab | Kabyle | Afro-Asiatic (Berber) | Latn | latin-marks | common_voice_25 | 542.9 | 141.6 | 13.9 | 13.9 | 570.7 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| eu | Basque | Isolate | Latn | whisper-basic | common_voice_25 | 427.7 | 203.1 | 22.4 | 22.4 | 472.4 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| ug | Uyghur | Turkic (Karluk) | Arab | uyghur-ug | common_voice_25 | 402.2 | 209.4 | 24.2 | 24.2 | 450.6 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| lg | Luganda | Atlantic-Congo (Bantu) | Latn | latin-marks | common_voice_25 | 393.9 | 114.2 | 21.5 | 21.5 | 436.8 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| sw | Swahili | Atlantic-Congo (Bantu) | Latn | latin-marks | common_voice_25 | 356.4 | 68.1 | 17.9 | 17.9 | 392.1 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| ja | Japanese | Japonic | Jpan | ja-cer | common_voice_25 | 349.5 | 24.4 | 11.2 | 11.2 | 371.9 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| fa | Persian | Indo-European (Iranian) | Arab | perso-arabic | common_voice_25 | 349.5 | 33.0 | 11.7 | 11.7 | 372.9 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| it | Italian | Indo-European (Romance) | Latn | whisper-basic | common_voice_25 | 317.1 | 261.9 | 22.9 | 22.9 | 362.9 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| kn | Kannada | Dravidian | Knda | indic-vistaar | slr126_kannada | 280.0 | 0.0 | 35.0 | 35.0 | 350.0 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| su | Sundanese | Austronesian (Malayo-Polynesian) | Latn | latin-marks | slr36_sundanese | 266.4 | 0.0 | 33.3 | 33.3 | 333.0 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| gl | Galician | Indo-European (Romance) | Latn | whisper-basic | common_voice_25 | 265.7 | 256.2 | 21.1 | 21.2 | 308.0 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| kk | Kazakh | Turkic (Kipchak) | Cyrl | whisper-basic | slr102_ksc_kazakh | 265.6 | 0.0 | 33.2 | 33.2 | 332.0 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| kbd | Kabardian | Northwest Caucasian | Cyrl | whisper-basic | common_voice_25 | 246.0 | 24.0 | 13.1 | 13.1 | 272.2 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| lv | Latvian | Indo-European (Baltic) | Latn | whisper-basic | common_voice_25 | 244.5 | 19.7 | 10.5 | 10.5 | 265.4 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| mhr | Meadow Mari | Uralic (Mari) | Cyrl | whisper-basic | common_voice_25 | 242.6 | 239.5 | 18.8 | 19.5 | 280.9 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| jv | Javanese | Austronesian (Malayo-Polynesian) | Latn | latin-marks | slr35_javanese | 236.8 | 0.0 | 29.6 | 29.6 | 296.0 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| ba | Bashkir | Turkic (Kipchak) | Cyrl | whisper-basic | common_voice_25 | 223.0 | 146.5 | 17.9 | 17.9 | 258.8 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| ru | Russian | Indo-European (Slavic) | Cyrl | whisper-basic | common_voice_25 | 222.4 | 38.7 | 14.8 | 14.8 | 251.9 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| zh-CN | Chinese (Mandarin) | Sino-Tibetan (Sinitic) | Hans | han-mer | common_voice_25 | 212.3 | 37.3 | 13.4 | 13.4 | 239.1 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| yue | Cantonese | Sino-Tibetan (Sinitic) | Hant | han-mer | common_voice_25 | 199.4 | 8.2 | 5.6 | 5.6 | 210.7 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| ta | Tamil | Dravidian | Taml | indic-vistaar | common_voice_25 | 192.9 | 79.9 | 20.9 | 21.0 | 234.8 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| bn | Bengali | Indo-European (Indo-Aryan) | Beng | indic-vistaar | slr53_bengali | 183.2 | 0.0 | 22.9 | 22.9 | 229.0 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| si | Sinhala | Indo-European (Indo-Aryan) | Sinh | indic-vistaar | slr52_sinhala | 179.2 | 0.0 | 22.4 | 22.4 | 224.0 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| pt | Portuguese | Indo-European (Romance) | Latn | whisper-basic | common_voice_25 | 164.8 | 26.9 | 11.2 | 11.2 | 187.3 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| ab | Abkhaz | Northwest Caucasian | Cyrl | whisper-basic | common_voice_25 | 164.1 | 148.8 | 21.6 | 21.7 | 207.4 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| pl | Polish | Indo-European (Slavic) | Latn | whisper-basic | common_voice_25 | 151.0 | 32.3 | 12.8 | 12.8 | 176.6 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| th | Thai | Kra-Dai (Tai) | Thai | thai-cer | common_voice_25 | 147.5 | 38.4 | 12.9 | 12.9 | 173.3 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| ne | Nepali | Indo-European (Indo-Aryan) | Deva | indic-vistaar | slr54_nepali | 132.0 | 0.0 | 16.5 | 16.5 | 165.0 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| ka | Georgian | Kartvelian | Geor | whisper-basic | common_voice_25 | 130.3 | 89.7 | 18.6 | 18.7 | 167.5 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| ckb | Central Kurdish | Indo-European (Iranian) | Arab | perso-arabic | common_voice_25 | 124.8 | 9.0 | 6.1 | 6.1 | 137.0 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| is | Icelandic | Indo-European (Germanic) | Latn | whisper-basic | slr112_samromur | 116.0 | 0.0 | 14.5 | 14.5 | 145.0 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| cy | Welsh | Indo-European (Celtic) | Latn | whisper-basic | common_voice_25 | 109.3 | 11.0 | 7.4 | 7.4 | 124.1 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| tr | Turkish | Turkic (Oghuz) | Latn | turkic-tr | common_voice_25 | 104.0 | 43.6 | 12.6 | 12.6 | 129.2 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| nl | Dutch | Indo-European (Germanic) | Latn | whisper-basic | common_voice_25 | 96.5 | 56.3 | 14.9 | 14.9 | 126.2 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| zh-HK | — | — | — | — | common_voice_25 | 95.6 | 9.7 | 6.5 | 6.5 | 108.5 | no | same language as zh-CN, which has more trainable audio |
| hu | Hungarian | Uralic (Ugric) | Latn | whisper-basic | common_voice_25 | 92.6 | 91.4 | 19.9 | 20.0 | 132.6 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| uk | Ukrainian | Indo-European (Slavic) | Cyrl | whisper-basic | common_voice_25 | 74.8 | 35.6 | 13.4 | 13.4 | 101.6 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| uz | Uzbek | Turkic (Karluk) | Latn | latin-marks | common_voice_25 | 72.5 | 56.5 | 14.2 | 14.3 | 101.0 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| zh-TW | — | — | — | — | common_voice_25 | 70.1 | 6.9 | 4.8 | 4.8 | 79.7 | no | same language as zh-CN, which has more trainable audio |
| ar | Arabic | Afro-Asiatic (Semitic) | Arab | arabic-ouaal | common_voice_25 | 67.9 | 33.4 | 11.8 | 12.1 | 91.9 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| ur | Urdu | Indo-European (Indo-Aryan) | Arab | perso-arabic | common_voice_25 | 67.8 | 8.6 | 6.0 | 6.0 | 79.7 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| bo | Tibetan | Sino-Tibetan (Bodish) | Tibt | tibetan-syllable | slr124_tibetan | 67.5 | 0.0 | 8.4 | 8.4 | 84.3 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| kmr | Northern Kurdish | Indo-European (Iranian) | Latn | latin-marks | common_voice_25 | 66.0 | 6.5 | 4.8 | 4.8 | 75.7 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| fy-NL | West Frisian | Indo-European (Germanic) | Latn | whisper-basic | common_voice_25 | 61.9 | 5.3 | 4.3 | 4.3 | 70.4 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| ady | Adyghe | Northwest Caucasian | Cyrl | whisper-basic | common_voice_25 | 60.7 | 6.2 | 4.9 | 4.9 | 70.5 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| cs | Czech | Indo-European (Slavic) | Latn | whisper-basic | common_voice_25 | 57.6 | 27.6 | 11.7 | 11.7 | 81.1 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| hy | Armenian | Indo-European (Armenian) | Armn | armenian-hy | slr160_armenian | 56.0 | 0.0 | 7.0 | 7.0 | 70.0 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| nso | Sepedi | Atlantic-Congo (Sotho-Tswana) | Latn | latin-marks | nchlt_sepedi | 50.9 | 50.9 | 2.5 | 2.9 | 0.0 | yes | meets the rule (trainable >= 50.0 h, dev >= 2.0 h, test >= 2.0 h) |
| ts | Xitsonga | Atlantic-Congo (Tswa-Ronga) | Latn | latin-marks | nchlt_xitsonga | 49.8 | 49.8 | 2.9 | 3.6 | 0.0 | yes | below the rule (trainable 49.8 h < 50.0 h); kept for typological coverage; its NCHLT corpus totals about 56 h and the shortfall is in the shipped train split, not in the corpus |
| ve | Tshivenda | Atlantic-Congo (Venda) | Latn | latin-marks | nchlt_tshivenda | 49.6 | 49.6 | 3.6 | 3.1 | 0.0 | yes | below the rule (trainable 49.6 h < 50.0 h); kept for typological coverage; its NCHLT corpus totals about 56 h and the shortfall is in the shipped train split, not in the corpus |
| xh | isiXhosa | Atlantic-Congo (Nguni) | Latn | latin-marks | nchlt_xhosa | 49.4 | 49.4 | 3.8 | 3.1 | 0.0 | yes | below the rule (trainable 49.4 h < 50.0 h); kept for typological coverage; its NCHLT corpus totals about 56 h and the shortfall is in the shipped train split, not in the corpus |
| zu | isiZulu | Atlantic-Congo (Nguni) | Latn | latin-marks | nchlt_zulu | 48.5 | 48.5 | 3.7 | 4.0 | 0.0 | yes | below the rule (trainable 48.5 h < 50.0 h); kept for typological coverage; its NCHLT corpus totals about 56 h and the shortfall is in the shipped train split, not in the corpus |
| sk | — | — | — | — | common_voice_25 | 42.9 | 11.6 | 6.6 | 7.1 | 56.6 | no | trainable 42.9 h < 50.0 h |
| ko | Korean | Koreanic | Kore | ko-kspon | slr40_zeroth_korean | 42.2 | 51.6 | 5.3 | 5.3 | 52.8 | yes | below the rule (trainable 42.2 h < 50.0 h); kept because it is the only open Korean corpus, and Koreanic and Hangul are in the preset only through it; the shipped 1.2 h test is under the evaluation bar, so all 52.8 h are repartitioned |
| et | — | — | — | — | common_voice_25 | 40.9 | 6.5 | 5.4 | 5.4 | 51.7 | no | read from taltech_estonian instead, which has more trainable audio |
| mn | — | — | — | — | common_voice_25 | 40.6 | 3.0 | 2.6 | 2.6 | 45.8 | no | trainable 40.6 h < 50.0 h |
| sv-SE | — | — | — | — | common_voice_25 | 35.4 | 9.2 | 6.1 | 6.2 | 47.7 | no | trainable 35.4 h < 50.0 h |
| ky | — | — | — | — | common_voice_25 | 34.8 | 2.3 | 2.0 | 2.0 | 38.9 | no | trainable 34.8 h < 50.0 h |
| bn | — | — | — | — | common_voice_25 | 31.5 | 26.2 | 11.4 | 11.4 | 54.3 | no | read from slr53_bengali instead, which has more trainable audio |
| dv | — | — | — | — | common_voice_25 | 31.4 | 3.8 | 3.2 | 3.2 | 37.8 | no | trainable 31.4 h < 50.0 h |
| qxp | — | — | — | — | common_voice_25 | 29.3 | 1.0 | 0.9 | 0.9 | 31.2 | no | trainable 29.3 h < 50.0 h |
| id | — | — | — | — | common_voice_25 | 25.7 | 5.5 | 3.8 | 4.1 | 33.6 | no | trainable 25.7 h < 50.0 h |
| kln | — | — | — | — | common_voice_25 | 24.9 | 13.9 | 8.1 | 7.6 | 40.6 | no | trainable 24.9 h < 50.0 h |
| br | — | — | — | — | common_voice_25 | 24.4 | 4.0 | 3.2 | 3.2 | 30.7 | no | trainable 24.4 h < 50.0 h |
| tt | — | — | — | — | common_voice_25 | 22.1 | 9.0 | 4.7 | 5.4 | 32.3 | no | trainable 22.1 h < 50.0 h |
| cv | — | — | — | — | common_voice_25 | 21.0 | 2.0 | 1.7 | 1.8 | 24.5 | no | trainable 21.0 h < 50.0 h |
| ltg | — | — | — | — | common_voice_25 | 20.4 | 6.3 | 4.9 | 4.9 | 30.1 | no | trainable 20.4 h < 50.0 h |
| mk | — | — | — | — | common_voice_25 | 20.1 | 2.8 | 2.4 | 2.4 | 25.0 | no | trainable 20.1 h < 50.0 h |
| luo | — | — | — | — | common_voice_25 | 19.3 | 6.1 | 4.1 | 4.1 | 27.5 | no | trainable 19.3 h < 50.0 h |
| nnh | — | — | — | — | common_voice_25 | 17.7 | 1.0 | 0.5 | 0.7 | 18.9 | no | trainable 17.7 h < 50.0 h |
| phl | — | — | — | — | common_voice_25 | 17.5 | 2.7 | 1.9 | 1.9 | 21.3 | no | trainable 17.5 h < 50.0 h |
| lzz | — | — | — | — | common_voice_25 | 17.4 | 6.2 | 4.4 | 4.3 | 26.1 | no | trainable 17.4 h < 50.0 h |
| mrj | — | — | — | — | common_voice_25 | 16.9 | 16.7 | 8.5 | 8.3 | 33.7 | no | trainable 16.9 h < 50.0 h |
| mvy | — | — | — | — | common_voice_25 | 16.7 | 3.4 | 2.6 | 2.8 | 22.1 | no | trainable 16.7 h < 50.0 h |
| hy-AM | — | — | — | — | common_voice_25 | 16.0 | 15.7 | 8.9 | 9.3 | 34.3 | no | trainable 16.0 h < 50.0 h |
| el | — | — | — | — | common_voice_25 | 16.0 | 2.2 | 2.0 | 2.0 | 19.9 | no | trainable 16.0 h < 50.0 h |
| ksf | — | — | — | — | common_voice_25 | 15.7 | 0.9 | 0.7 | 0.7 | 17.2 | no | trainable 15.7 h < 50.0 h |
| oru | — | — | — | — | common_voice_25 | 14.8 | 8.4 | 3.5 | 3.0 | 21.2 | no | trainable 14.8 h < 50.0 h |
| sl | — | — | — | — | common_voice_25 | 14.4 | 1.6 | 1.5 | 1.5 | 17.3 | no | trainable 14.4 h < 50.0 h |
| sva | — | — | — | — | common_voice_25 | 14.1 | 0.9 | 0.8 | 0.8 | 15.7 | no | trainable 14.1 h < 50.0 h |
| bnm | — | — | — | — | common_voice_25 | 14.1 | 0.8 | 0.6 | 0.7 | 15.3 | no | trainable 14.1 h < 50.0 h |
| ro | — | — | — | — | common_voice_25 | 13.4 | 5.8 | 4.4 | 4.4 | 22.3 | no | trainable 13.4 h < 50.0 h |
| szy | — | — | — | — | common_voice_25 | 12.9 | 0.4 | 0.4 | 0.4 | 13.7 | no | trainable 12.9 h < 50.0 h |
| mr | — | — | — | — | common_voice_25 | 12.8 | 3.8 | 3.0 | 3.1 | 18.9 | no | held-out transfer language (marathi); never trained on |
| dag | — | — | — | — | common_voice_25 | 12.7 | 1.9 | 1.7 | 1.7 | 16.2 | no | trainable 12.7 h < 50.0 h |
| pwn | — | — | — | — | common_voice_25 | 12.7 | 1.0 | 1.0 | 1.0 | 14.6 | no | trainable 12.7 h < 50.0 h |
| nan-tw | — | — | — | — | common_voice_25 | 12.6 | 8.5 | 4.4 | 4.7 | 21.8 | no | trainable 12.6 h < 50.0 h |
| ewo | — | — | — | — | common_voice_25 | 12.6 | 0.6 | 0.5 | 0.5 | 13.6 | no | trainable 12.6 h < 50.0 h |
| lt | — | — | — | — | common_voice_25 | 12.5 | 12.2 | 7.9 | 8.0 | 28.4 | no | trainable 12.5 h < 50.0 h |
| phr | — | — | — | — | common_voice_25 | 12.5 | 0.8 | 0.7 | 0.7 | 14.0 | no | trainable 12.5 h < 50.0 h |
| ga-IE | — | — | — | — | common_voice_25 | 12.4 | 1.0 | 0.9 | 0.9 | 14.2 | no | trainable 12.4 h < 50.0 h |
| byv | — | — | — | — | common_voice_25 | 12.1 | 0.6 | 0.6 | 0.6 | 13.2 | no | trainable 12.1 h < 50.0 h |
| fub | — | — | — | — | common_voice_25 | 11.9 | 0.8 | 0.5 | 0.6 | 12.9 | no | trainable 11.9 h < 50.0 h |
| gwt | — | — | — | — | common_voice_25 | 11.9 | 5.0 | 0.0 | 0.3 | 12.2 | no | trainable 11.9 h < 50.0 h |
| ajg | — | — | — | — | common_voice_25 | 11.8 | 1.2 | 0.4 | 0.5 | 12.7 | no | trainable 11.8 h < 50.0 h |
| pcm | — | — | — | — | common_voice_25 | 11.4 | 0.5 | 0.5 | 0.5 | 12.5 | no | trainable 11.4 h < 50.0 h |
| dua | — | — | — | — | common_voice_25 | 11.4 | 0.7 | 0.6 | 0.6 | 12.5 | no | trainable 11.4 h < 50.0 h |
| mxu | — | — | — | — | common_voice_25 | 11.3 | 0.5 | 0.5 | 0.5 | 12.3 | no | trainable 11.3 h < 50.0 h |
| bbj | — | — | — | — | common_voice_25 | 11.3 | 0.7 | 0.5 | 0.5 | 12.3 | no | trainable 11.3 h < 50.0 h |
| qxw | — | — | — | — | common_voice_25 | 11.1 | 0.9 | 0.2 | 0.3 | 11.7 | no | trainable 11.1 h < 50.0 h |
| fi | Finnish | Uralic (Finnic) | Latn | whisper-basic | common_voice_25 | 11.1 | 2.7 | 2.3 | 2.3 | 15.8 | yes | below the rule (trainable 11.1 h < 50.0 h); kept because the smaller presets commit to it |
| qup | — | — | — | — | common_voice_25 | 11.0 | 0.7 | 0.5 | 0.4 | 11.9 | no | trainable 11.0 h < 50.0 h |
| kxp | — | — | — | — | common_voice_25 | 10.8 | 1.8 | 0.0 | 0.2 | 11.0 | no | trainable 10.8 h < 50.0 h |
| lrk | — | — | — | — | common_voice_25 | 10.7 | 1.6 | 0.0 | 0.3 | 11.1 | no | trainable 10.7 h < 50.0 h |
| cjk | — | — | — | — | common_voice_25 | 10.7 | 0.6 | 0.6 | 0.6 | 12.0 | no | trainable 10.7 h < 50.0 h |
| nlv | — | — | — | — | common_voice_25 | 10.7 | 0.8 | 0.4 | 0.5 | 11.6 | no | trainable 10.7 h < 50.0 h |
| bft | — | — | — | — | common_voice_25 | 10.6 | 3.6 | 3.0 | 3.0 | 16.6 | no | trainable 10.6 h < 50.0 h |
| bkm | — | — | — | — | common_voice_25 | 10.5 | 0.5 | 0.5 | 0.5 | 11.4 | no | trainable 10.5 h < 50.0 h |
| sbn | — | — | — | — | common_voice_25 | 10.5 | 1.7 | 0.0 | 0.2 | 10.7 | no | trainable 10.5 h < 50.0 h |
| qvi | — | — | — | — | common_voice_25 | 10.5 | 0.5 | 0.4 | 0.5 | 11.4 | no | trainable 10.5 h < 50.0 h |
| trw | — | — | — | — | common_voice_25 | 10.5 | 4.3 | 3.1 | 2.9 | 16.5 | no | trainable 10.5 h < 50.0 h |
| xmf | — | — | — | — | common_voice_25 | 10.5 | 0.6 | 0.6 | 0.6 | 11.6 | no | trainable 10.5 h < 50.0 h |
| fue | — | — | — | — | common_voice_25 | 10.4 | 1.3 | 0.0 | 0.1 | 10.6 | no | trainable 10.4 h < 50.0 h |
| mhk | — | — | — | — | common_voice_25 | 10.4 | 0.8 | 0.5 | 0.3 | 11.3 | no | trainable 10.4 h < 50.0 h |
| ssi | — | — | — | — | common_voice_25 | 10.4 | 1.9 | 0.0 | 0.2 | 10.5 | no | trainable 10.4 h < 50.0 h |
| abb | — | — | — | — | common_voice_25 | 10.3 | 0.6 | 0.4 | 0.4 | 11.2 | no | trainable 10.3 h < 50.0 h |
| ydg | — | — | — | — | common_voice_25 | 10.3 | 1.4 | 0.0 | 0.4 | 10.6 | no | trainable 10.3 h < 50.0 h |
| qws | — | — | — | — | common_voice_25 | 10.2 | 1.1 | 0.0 | 0.1 | 10.3 | no | trainable 10.2 h < 50.0 h |
| bci | — | — | — | — | common_voice_25 | 10.1 | 0.6 | 0.5 | 0.6 | 11.3 | no | trainable 10.1 h < 50.0 h |
| tok | — | — | — | — | common_voice_25 | 10.1 | 3.2 | 2.7 | 2.7 | 15.6 | no | trainable 10.1 h < 50.0 h |
| khw | — | — | — | — | common_voice_25 | 10.1 | 5.0 | 3.0 | 2.9 | 16.0 | no | trainable 10.1 h < 50.0 h |
| bag | — | — | — | — | common_voice_25 | 10.1 | 0.7 | 0.4 | 0.5 | 11.0 | no | trainable 10.1 h < 50.0 h |
| qus | — | — | — | — | common_voice_25 | 10.1 | 0.4 | 0.3 | 0.3 | 10.8 | no | trainable 10.1 h < 50.0 h |
| sah | — | — | — | — | common_voice_25 | 10.1 | 4.1 | 3.1 | 3.2 | 16.3 | no | trainable 10.1 h < 50.0 h |
| dar | — | — | — | — | common_voice_25 | 10.1 | 3.2 | 2.3 | 2.2 | 14.5 | no | trainable 10.1 h < 50.0 h |
| jgo | — | — | — | — | common_voice_25 | 10.1 | 0.7 | 0.6 | 0.6 | 11.3 | no | trainable 10.1 h < 50.0 h |
| mcf | — | — | — | — | common_voice_25 | 10.0 | 0.5 | 0.0 | 0.2 | 10.2 | no | trainable 10.0 h < 50.0 h |
| qur | — | — | — | — | common_voice_25 | 10.0 | 0.9 | 0.0 | 0.0 | 10.0 | no | trainable 10.0 h < 50.0 h |
| mbo | — | — | — | — | common_voice_25 | 10.0 | 0.6 | 0.5 | 0.4 | 10.9 | no | trainable 10.0 h < 50.0 h |
| gig | — | — | — | — | common_voice_25 | 10.0 | 1.6 | 0.0 | 0.1 | 10.1 | no | trainable 10.0 h < 50.0 h |
| fmp | — | — | — | — | common_voice_25 | 9.9 | 0.8 | 0.7 | 0.7 | 11.4 | no | trainable 9.9 h < 50.0 h |
| plk | — | — | — | — | common_voice_25 | 9.9 | 3.5 | 1.7 | 0.9 | 12.6 | no | trainable 9.9 h < 50.0 h |
| mdd | — | — | — | — | common_voice_25 | 9.9 | 1.6 | 0.0 | 0.1 | 10.0 | no | trainable 9.9 h < 50.0 h |
| nmz | — | — | — | — | common_voice_25 | 9.9 | 0.7 | 0.7 | 0.7 | 11.2 | no | trainable 9.9 h < 50.0 h |
| mki | — | — | — | — | common_voice_25 | 9.9 | 1.8 | 0.0 | 0.0 | 9.9 | no | trainable 9.9 h < 50.0 h |
| gej | — | — | — | — | common_voice_25 | 9.9 | 0.9 | 0.6 | 0.6 | 11.1 | no | trainable 9.9 h < 50.0 h |
| ncx | — | — | — | — | common_voice_25 | 9.9 | 0.4 | 0.4 | 0.4 | 10.7 | no | trainable 9.9 h < 50.0 h |
| haz | — | — | — | — | common_voice_25 | 9.8 | 1.0 | 0.1 | 0.6 | 10.5 | no | trainable 9.8 h < 50.0 h |
| qxt | — | — | — | — | common_voice_25 | 9.8 | 0.7 | 0.1 | 0.4 | 10.3 | no | trainable 9.8 h < 50.0 h |
| ia | — | — | — | — | common_voice_25 | 9.8 | 5.7 | 2.2 | 2.2 | 14.3 | no | trainable 9.8 h < 50.0 h |
| bri | — | — | — | — | common_voice_25 | 9.8 | 0.8 | 0.2 | 0.4 | 10.4 | no | trainable 9.8 h < 50.0 h |
| bba | — | — | — | — | common_voice_25 | 9.7 | 0.5 | 0.4 | 0.4 | 10.6 | no | trainable 9.7 h < 50.0 h |
| qxu | — | — | — | — | common_voice_25 | 9.7 | 0.8 | 0.0 | 0.4 | 10.1 | no | trainable 9.7 h < 50.0 h |
| bax | — | — | — | — | common_voice_25 | 9.7 | 0.5 | 0.4 | 0.5 | 10.6 | no | trainable 9.7 h < 50.0 h |
| qvj | — | — | — | — | common_voice_25 | 9.7 | 0.6 | 0.6 | 0.6 | 10.8 | no | trainable 9.7 h < 50.0 h |
| btv | — | — | — | — | common_voice_25 | 9.6 | 0.5 | 0.3 | 0.4 | 10.3 | no | trainable 9.6 h < 50.0 h |
| kw | — | — | — | — | common_voice_25 | 9.6 | 7.1 | 0.0 | 2.7 | 12.4 | no | trainable 9.6 h < 50.0 h |
| hux | — | — | — | — | common_voice_25 | 9.6 | 0.7 | 0.0 | 0.4 | 10.0 | no | trainable 9.6 h < 50.0 h |
| wes | — | — | — | — | common_voice_25 | 9.6 | 0.4 | 0.4 | 0.4 | 10.3 | no | trainable 9.6 h < 50.0 h |
| gju | — | — | — | — | common_voice_25 | 9.5 | 2.9 | 0.0 | 0.6 | 10.1 | no | trainable 9.5 h < 50.0 h |
| tay | — | — | — | — | common_voice_25 | 9.5 | 2.0 | 0.7 | 1.3 | 11.5 | no | trainable 9.5 h < 50.0 h |
| an | — | — | — | — | common_voice_25 | 9.5 | 5.3 | 3.6 | 3.7 | 16.9 | no | trainable 9.5 h < 50.0 h |
| xka | — | — | — | — | common_voice_25 | 9.5 | 1.4 | 0.0 | 0.4 | 9.8 | no | trainable 9.5 h < 50.0 h |
| nmg | — | — | — | — | common_voice_25 | 9.5 | 0.9 | 0.5 | 0.5 | 10.4 | no | trainable 9.5 h < 50.0 h |
| mau | — | — | — | — | common_voice_25 | 9.4 | 1.0 | 0.4 | 0.5 | 10.4 | no | trainable 9.4 h < 50.0 h |
| cpy | — | — | — | — | common_voice_25 | 9.4 | 0.6 | 0.2 | 0.4 | 10.0 | no | trainable 9.4 h < 50.0 h |
| mcx | — | — | — | — | common_voice_25 | 9.4 | 1.0 | 0.2 | 0.5 | 10.1 | no | trainable 9.4 h < 50.0 h |
| qxa | — | — | — | — | common_voice_25 | 9.4 | 0.5 | 0.3 | 0.4 | 10.1 | no | trainable 9.4 h < 50.0 h |
| kvx | — | — | — | — | common_voice_25 | 9.4 | 1.3 | 1.0 | 0.7 | 11.0 | no | trainable 9.4 h < 50.0 h |
| qvl | — | — | — | — | common_voice_25 | 9.4 | 0.5 | 0.2 | 0.4 | 10.0 | no | trainable 9.4 h < 50.0 h |
| rof | — | — | — | — | common_voice_25 | 9.4 | 0.5 | 0.5 | 0.5 | 10.4 | no | trainable 9.4 h < 50.0 h |
| qwa | — | — | — | — | common_voice_25 | 9.4 | 0.8 | 0.1 | 0.4 | 9.9 | no | trainable 9.4 h < 50.0 h |
| mcn | — | — | — | — | common_voice_25 | 9.4 | 0.4 | 0.4 | 0.4 | 10.1 | no | trainable 9.4 h < 50.0 h |
| bbl | — | — | — | — | common_voice_25 | 9.3 | 1.0 | 0.9 | 0.9 | 11.2 | no | trainable 9.3 h < 50.0 h |
| bum | — | — | — | — | common_voice_25 | 9.3 | 0.4 | 0.3 | 0.4 | 10.0 | no | trainable 9.3 h < 50.0 h |
| dmk | — | — | — | — | common_voice_25 | 9.3 | 3.1 | 0.0 | 0.9 | 10.2 | no | trainable 9.3 h < 50.0 h |
| odk | — | — | — | — | common_voice_25 | 9.3 | 1.7 | 0.8 | 1.1 | 11.2 | no | trainable 9.3 h < 50.0 h |
| bfd | — | — | — | — | common_voice_25 | 9.3 | 0.4 | 0.4 | 0.4 | 10.1 | no | trainable 9.3 h < 50.0 h |
| gjk | — | — | — | — | common_voice_25 | 9.2 | 1.0 | 0.7 | 0.8 | 10.8 | no | trainable 9.2 h < 50.0 h |
| qva | — | — | — | — | common_voice_25 | 9.2 | 0.6 | 0.2 | 0.4 | 9.8 | no | trainable 9.2 h < 50.0 h |
| mgg | — | — | — | — | common_voice_25 | 9.2 | 1.0 | 0.6 | 0.4 | 10.2 | no | trainable 9.2 h < 50.0 h |
| giz | — | — | — | — | common_voice_25 | 9.1 | 0.6 | 0.5 | 0.4 | 10.1 | no | trainable 9.1 h < 50.0 h |
| mve | — | — | — | — | common_voice_25 | 9.1 | 1.7 | 0.7 | 0.2 | 10.1 | no | trainable 9.1 h < 50.0 h |
| beb | — | — | — | — | common_voice_25 | 9.1 | 0.5 | 0.5 | 0.5 | 10.0 | no | trainable 9.1 h < 50.0 h |
| mua | — | — | — | — | common_voice_25 | 9.0 | 0.5 | 0.3 | 0.4 | 9.7 | no | trainable 9.0 h < 50.0 h |
| prq | — | — | — | — | common_voice_25 | 9.0 | 0.5 | 0.4 | 0.4 | 9.7 | no | trainable 9.0 h < 50.0 h |
| zoc | — | — | — | — | common_voice_25 | 9.0 | 0.6 | 0.5 | 0.5 | 10.1 | no | trainable 9.0 h < 50.0 h |
| gwc | — | — | — | — | common_voice_25 | 9.0 | 5.1 | 1.2 | 1.4 | 11.6 | no | trainable 9.0 h < 50.0 h |
| bas | — | — | — | — | common_voice_25 | 9.0 | 2.3 | 1.4 | 1.7 | 12.1 | no | trainable 9.0 h < 50.0 h |
| sei | — | — | — | — | common_voice_25 | 9.0 | 0.9 | 0.5 | 0.6 | 10.1 | no | trainable 9.0 h < 50.0 h |
| cux | — | — | — | — | common_voice_25 | 9.0 | 1.3 | 0.7 | 0.6 | 10.3 | no | trainable 9.0 h < 50.0 h |
| bce | — | — | — | — | common_voice_25 | 9.0 | 0.5 | 0.5 | 0.5 | 10.0 | no | trainable 9.0 h < 50.0 h |
| cut | — | — | — | — | common_voice_25 | 9.0 | 0.7 | 0.5 | 0.6 | 10.1 | no | trainable 9.0 h < 50.0 h |
| tar | — | — | — | — | common_voice_25 | 8.9 | 0.5 | 0.5 | 0.5 | 10.0 | no | trainable 8.9 h < 50.0 h |
| jqr | — | — | — | — | common_voice_25 | 8.9 | 0.7 | 0.5 | 0.4 | 9.9 | no | trainable 8.9 h < 50.0 h |
| bsk | — | — | — | — | common_voice_25 | 8.9 | 1.4 | 0.4 | 0.9 | 10.2 | no | trainable 8.9 h < 50.0 h |
| eto | — | — | — | — | common_voice_25 | 8.9 | 0.3 | 0.3 | 0.3 | 9.4 | no | trainable 8.9 h < 50.0 h |
| bkh | — | — | — | — | common_voice_25 | 8.9 | 0.7 | 0.5 | 0.5 | 10.0 | no | trainable 8.9 h < 50.0 h |
| qux | — | — | — | — | common_voice_25 | 8.9 | 0.6 | 0.4 | 0.5 | 9.8 | no | trainable 8.9 h < 50.0 h |
| lss | — | — | — | — | common_voice_25 | 8.9 | 0.9 | 0.5 | 0.6 | 9.9 | no | trainable 8.9 h < 50.0 h |
| yaq | — | — | — | — | common_voice_25 | 8.8 | 2.8 | 0.2 | 1.2 | 10.2 | no | trainable 8.8 h < 50.0 h |
| hem | — | — | — | — | common_voice_25 | 8.8 | 0.6 | 0.6 | 0.6 | 9.9 | no | trainable 8.8 h < 50.0 h |
| pua | — | — | — | — | common_voice_25 | 8.8 | 1.6 | 0.7 | 0.7 | 10.2 | no | trainable 8.8 h < 50.0 h |
| gid | — | — | — | — | common_voice_25 | 8.8 | 0.6 | 0.6 | 0.6 | 9.9 | no | trainable 8.8 h < 50.0 h |
| gya | — | — | — | — | common_voice_25 | 8.8 | 0.5 | 0.5 | 0.5 | 9.8 | no | trainable 8.8 h < 50.0 h |
| kdh | — | — | — | — | common_voice_25 | 8.7 | 0.3 | 0.2 | 0.2 | 9.2 | no | trainable 8.7 h < 50.0 h |
| fan | — | — | — | — | common_voice_25 | 8.6 | 0.4 | 0.4 | 0.4 | 9.4 | no | trainable 8.6 h < 50.0 h |
| udl | — | — | — | — | common_voice_25 | 8.6 | 0.6 | 0.4 | 0.5 | 9.5 | no | trainable 8.6 h < 50.0 h |
| nyu | — | — | — | — | common_voice_25 | 8.5 | 2.6 | 0.0 | 0.6 | 9.2 | no | trainable 8.5 h < 50.0 h |
| hno | — | — | — | — | common_voice_25 | 8.5 | 1.0 | 0.9 | 0.8 | 10.2 | no | trainable 8.5 h < 50.0 h |
| xhe | — | — | — | — | common_voice_25 | 8.4 | 3.0 | 0.0 | 1.2 | 9.6 | no | trainable 8.4 h < 50.0 h |
| tvu | — | — | — | — | common_voice_25 | 8.4 | 1.4 | 1.0 | 0.9 | 10.2 | no | trainable 8.4 h < 50.0 h |
| var | — | — | — | — | common_voice_25 | 8.4 | 1.1 | 0.8 | 0.9 | 10.1 | no | trainable 8.4 h < 50.0 h |
| trv | — | — | — | — | common_voice_25 | 8.3 | 1.4 | 0.9 | 0.8 | 9.9 | no | trainable 8.3 h < 50.0 h |
| nla | — | — | — | — | common_voice_25 | 8.2 | 1.1 | 0.3 | 0.4 | 8.9 | no | trainable 8.2 h < 50.0 h |
| tui | — | — | — | — | common_voice_25 | 8.1 | 0.9 | 0.8 | 0.8 | 9.7 | no | trainable 8.1 h < 50.0 h |
| bsh | — | — | — | — | common_voice_25 | 8.1 | 2.1 | 0.8 | 1.0 | 9.9 | no | trainable 8.1 h < 50.0 h |
| tli | — | — | — | — | common_voice_25 | 8.1 | 8.1 | 0.0 | 1.8 | 9.9 | no | trainable 8.1 h < 50.0 h |
| scl | — | — | — | — | common_voice_25 | 7.9 | 1.6 | 1.0 | 1.1 | 10.0 | no | trainable 7.9 h < 50.0 h |
| wbl | — | — | — | — | common_voice_25 | 7.9 | 4.8 | 2.1 | 2.1 | 12.1 | no | trainable 7.9 h < 50.0 h |
| lua | — | — | — | — | common_voice_25 | 7.8 | 0.6 | 0.5 | 0.5 | 8.9 | no | trainable 7.8 h < 50.0 h |
| bg | — | — | — | — | common_voice_25 | 7.6 | 7.6 | 4.5 | 5.2 | 17.3 | no | trainable 7.6 h < 50.0 h |
| bnn | — | — | — | — | common_voice_25 | 7.6 | 1.5 | 1.4 | 1.4 | 10.3 | no | trainable 7.6 h < 50.0 h |
| yav | — | — | — | — | common_voice_25 | 7.5 | 0.8 | 0.5 | 0.6 | 8.6 | no | trainable 7.5 h < 50.0 h |
| kls | — | — | — | — | common_voice_25 | 7.5 | 1.5 | 1.3 | 1.3 | 10.0 | no | trainable 7.5 h < 50.0 h |
| gv | — | — | — | — | common_voice_25 | 7.4 | 3.9 | 1.9 | 0.8 | 10.1 | no | trainable 7.4 h < 50.0 h |
| dru | — | — | — | — | common_voice_25 | 7.4 | 1.7 | 1.5 | 1.5 | 10.4 | no | trainable 7.4 h < 50.0 h |
| dml | — | — | — | — | common_voice_25 | 7.3 | 5.3 | 1.8 | 1.0 | 10.2 | no | trainable 7.3 h < 50.0 h |
| brh | — | — | — | — | common_voice_25 | 7.2 | 3.3 | 1.0 | 1.7 | 9.9 | no | trainable 7.2 h < 50.0 h |
| esu | — | — | — | — | common_voice_25 | 7.1 | 6.5 | 0.0 | 0.5 | 7.6 | no | trainable 7.1 h < 50.0 h |
| ggg | — | — | — | — | common_voice_25 | 7.0 | 1.9 | 0.0 | 0.4 | 7.4 | no | trainable 7.0 h < 50.0 h |
| hi | Hindi | Indo-European (Indo-Aryan) | Deva | indic-vistaar | common_voice_25 | 7.0 | 6.9 | 3.9 | 4.7 | 15.6 | yes | below the rule (trainable 7.0 h < 50.0 h); kept because the smaller presets commit to it |
| ipk | — | — | — | — | common_voice_25 | 6.8 | 6.8 | 0.0 | 0.4 | 7.2 | no | trainable 6.8 h < 50.0 h |
| da | — | — | — | — | common_voice_25 | 6.8 | 4.1 | 3.1 | 3.1 | 13.0 | no | trainable 6.8 h < 50.0 h |
| dav | — | — | — | — | common_voice_25 | 6.7 | 2.4 | 1.4 | 1.1 | 9.3 | no | trainable 6.7 h < 50.0 h |
| eko | — | — | — | — | common_voice_25 | 6.7 | 1.1 | 0.7 | 0.9 | 8.3 | no | trainable 6.7 h < 50.0 h |
| mse | — | — | — | — | common_voice_25 | 6.6 | 0.7 | 0.6 | 0.5 | 7.7 | no | trainable 6.6 h < 50.0 h |
| bgp | — | — | — | — | common_voice_25 | 6.6 | 5.3 | 2.1 | 2.8 | 11.5 | no | trainable 6.6 h < 50.0 h |
| ibb | — | — | — | — | common_voice_25 | 6.2 | 0.9 | 0.8 | 0.8 | 7.8 | no | trainable 6.2 h < 50.0 h |
| ush | — | — | — | — | common_voice_25 | 5.6 | 1.0 | 0.3 | 0.6 | 6.6 | no | trainable 5.6 h < 50.0 h |
| tig | — | — | — | — | common_voice_25 | 5.3 | 3.3 | 2.7 | 2.7 | 10.7 | no | trainable 5.3 h < 50.0 h |
| or | — | — | — | — | common_voice_25 | 4.6 | 3.3 | 1.0 | 0.7 | 6.3 | no | trainable 4.6 h < 50.0 h |
| mt | — | — | — | — | common_voice_25 | 4.4 | 2.5 | 2.1 | 2.2 | 8.7 | no | trainable 4.4 h < 50.0 h |
| mug | — | — | — | — | common_voice_25 | 4.2 | 0.8 | 0.6 | 0.6 | 5.4 | no | trainable 4.2 h < 50.0 h |
| sr | — | — | — | — | common_voice_25 | 4.2 | 2.3 | 1.7 | 1.8 | 7.6 | no | trainable 4.2 h < 50.0 h |
| vi | — | — | — | — | common_voice_25 | 4.2 | 2.1 | 1.5 | 1.6 | 7.3 | no | trainable 4.2 h < 50.0 h |
| sq | — | — | — | — | common_voice_25 | 3.8 | 3.8 | 2.5 | 2.7 | 9.0 | no | trainable 3.8 h < 50.0 h |
| he | — | — | — | — | common_voice_25 | 3.6 | 2.4 | 0.5 | 1.2 | 5.3 | no | trainable 3.6 h < 50.0 h |
| tn | — | — | — | — | common_voice_25 | 3.4 | 1.3 | 0.4 | 0.4 | 4.2 | no | trainable 3.4 h < 50.0 h |
| rm-sursilv | — | — | — | — | common_voice_25 | 3.1 | 2.9 | 2.4 | 2.5 | 7.9 | no | trainable 3.1 h < 50.0 h |
| gn | — | — | — | — | common_voice_25 | 2.9 | 2.2 | 0.8 | 1.4 | 5.2 | no | trainable 2.9 h < 50.0 h |
| lij | — | — | — | — | common_voice_25 | 2.6 | 2.5 | 1.0 | 1.5 | 5.0 | no | trainable 2.6 h < 50.0 h |
| ha | — | — | — | — | common_voice_25 | 2.5 | 2.3 | 0.8 | 0.9 | 4.2 | no | trainable 2.5 h < 50.0 h |
| yo | — | — | — | — | common_voice_25 | 2.4 | 2.4 | 1.6 | 1.8 | 5.8 | no | trainable 2.4 h < 50.0 h |
| myv | — | — | — | — | common_voice_25 | 2.0 | 2.0 | 0.4 | 0.8 | 3.2 | no | trainable 2.0 h < 50.0 h |
| ml | — | — | — | — | common_voice_25 | 1.9 | 1.5 | 1.1 | 1.0 | 4.1 | no | held-out transfer language (malayalam); never trained on |
| oc | — | — | — | — | common_voice_25 | 1.9 | 0.4 | 0.4 | 0.4 | 2.7 | no | trainable 1.9 h < 50.0 h |
| skr | — | — | — | — | common_voice_25 | 1.8 | 1.8 | 1.3 | 1.2 | 4.3 | no | trainable 1.8 h < 50.0 h |
| hsb | — | — | — | — | common_voice_25 | 1.7 | 1.7 | 0.7 | 1.0 | 3.4 | no | trainable 1.7 h < 50.0 h |
| as | — | — | — | — | common_voice_25 | 1.5 | 1.6 | 0.8 | 0.7 | 3.0 | no | trainable 1.5 h < 50.0 h |
| ebr | — | — | — | — | common_voice_25 | 1.5 | 0.7 | 0.0 | 0.3 | 1.8 | no | trainable 1.5 h < 50.0 h |
| nb-NO | — | — | — | — | common_voice_25 | 1.5 | 1.5 | 0.5 | 0.4 | 2.3 | no | trainable 1.5 h < 50.0 h |
| tk | — | — | — | — | common_voice_25 | 1.5 | 1.1 | 0.8 | 0.8 | 3.1 | no | trainable 1.5 h < 50.0 h |
| sc | — | — | — | — | common_voice_25 | 1.4 | 1.2 | 0.7 | 0.9 | 3.0 | no | trainable 1.4 h < 50.0 h |
| pa-IN | — | — | — | — | common_voice_25 | 1.1 | 1.1 | 0.7 | 0.7 | 2.4 | no | trainable 1.1 h < 50.0 h |
| yi | — | — | — | — | common_voice_25 | 1.0 | 0.5 | 0.5 | 0.5 | 2.0 | no | trainable 1.0 h < 50.0 h |
| ko | — | — | — | — | common_voice_25 | 1.0 | 1.0 | 0.7 | 0.8 | 2.5 | no | read from slr40_zeroth_korean instead, which has more trainable audio |
| ig | — | — | — | — | common_voice_25 | 1.0 | 1.0 | 0.9 | 0.9 | 2.7 | no | trainable 1.0 h < 50.0 h |
| am | — | — | — | — | common_voice_25 | 1.0 | 1.0 | 0.4 | 0.5 | 1.9 | no | trainable 1.0 h < 50.0 h |
| kk | — | — | — | — | common_voice_25 | 0.9 | 0.9 | 0.8 | 0.8 | 2.5 | no | read from slr102_ksc_kazakh instead, which has more trainable audio |
| rm-vallader | — | — | — | — | common_voice_25 | 0.9 | 0.9 | 0.8 | 0.8 | 2.5 | no | trainable 0.9 h < 50.0 h |
| cnh | — | — | — | — | common_voice_25 | 0.9 | 0.8 | 0.7 | 0.7 | 2.4 | no | trainable 0.9 h < 50.0 h |
| zza | — | — | — | — | common_voice_25 | 0.9 | 0.9 | 0.5 | 0.5 | 1.9 | no | trainable 0.9 h < 50.0 h |
| zgh | — | — | — | — | common_voice_25 | 0.9 | 0.9 | 0.3 | 0.2 | 1.4 | no | trainable 0.9 h < 50.0 h |
| nn-NO | — | — | — | — | common_voice_25 | 0.7 | 0.7 | 0.4 | 0.5 | 1.6 | no | trainable 0.7 h < 50.0 h |
| os | — | — | — | — | common_voice_25 | 0.7 | 0.6 | 0.4 | 0.3 | 1.4 | no | trainable 0.7 h < 50.0 h |
| ne-NP | — | — | — | — | common_voice_25 | 0.7 | 0.4 | 0.4 | 0.3 | 1.3 | no | trainable 0.7 h < 50.0 h |
| te | — | — | — | — | common_voice_25 | 0.6 | 0.1 | 0.1 | 0.1 | 0.8 | no | held-out transfer language (telugu); never trained on |
| ast | — | — | — | — | common_voice_25 | 0.6 | 0.5 | 0.1 | 0.3 | 1.0 | no | trainable 0.6 h < 50.0 h |
| gsw | — | — | — | — | common_voice_25 | 0.5 | 0.0 | 0.0 | 0.0 | 0.6 | no | trainable 0.5 h < 50.0 h |
| tg | — | — | — | — | common_voice_25 | 0.4 | 0.5 | 0.2 | 0.2 | 0.8 | no | trainable 0.4 h < 50.0 h |
| sat | — | — | — | — | common_voice_25 | 0.4 | 0.4 | 0.1 | 0.2 | 0.7 | no | trainable 0.4 h < 50.0 h |
| az | — | — | — | — | common_voice_25 | 0.3 | 0.3 | 0.1 | 0.2 | 0.7 | no | trainable 0.3 h < 50.0 h |
| af | — | — | — | — | common_voice_25 | 0.3 | 0.3 | 0.2 | 0.2 | 0.8 | no | trainable 0.3 h < 50.0 h |
| sd | — | — | — | — | common_voice_25 | 0.3 | 0.3 | 0.0 | 0.0 | 0.4 | no | trainable 0.3 h < 50.0 h |
| mdf | — | — | — | — | common_voice_25 | 0.3 | 0.3 | 0.1 | 0.2 | 0.5 | no | trainable 0.3 h < 50.0 h |
| tw | — | — | — | — | common_voice_25 | 0.3 | 0.3 | 0.0 | 0.0 | 0.3 | no | trainable 0.3 h < 50.0 h |
| lo | — | — | — | — | common_voice_25 | 0.2 | 0.2 | 0.1 | 0.1 | 0.3 | no | trainable 0.2 h < 50.0 h |
| dyu | — | — | — | — | common_voice_25 | 0.2 | 0.2 | 0.1 | 0.1 | 0.4 | no | trainable 0.2 h < 50.0 h |
| is | — | — | — | — | common_voice_25 | 0.1 | 0.1 | 0.0 | 0.1 | 0.2 | no | read from slr112_samromur instead, which has more trainable audio |
| vot | — | — | — | — | common_voice_25 | 0.1 | 0.1 | 0.0 | 0.0 | 0.1 | no | trainable 0.1 h < 50.0 h |
| ti | — | — | — | — | common_voice_25 | 0.0 | 0.1 | 0.0 | 0.0 | 0.1 | no | trainable 0.0 h < 50.0 h |
| quy | — | — | — | — | common_voice_25 | 0.0 | 0.0 | 0.0 | 0.0 | 0.1 | no | trainable 0.0 h < 50.0 h |
| ms | — | — | — | — | common_voice_25 | 0.0 | 0.0 | 0.0 | 0.0 | 0.1 | no | trainable 0.0 h < 50.0 h |
| nhi | — | — | — | — | common_voice_25 | 0.0 | 0.0 | 0.0 | 0.0 | 0.1 | no | trainable 0.0 h < 50.0 h |
| rup | — | — | — | — | common_voice_25 | 0.0 | 0.0 | 0.0 | 0.0 | 0.1 | no | trainable 0.0 h < 50.0 h |
| ht | — | — | — | — | common_voice_25 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | no | trainable 0.0 h < 50.0 h |
| zu | — | — | — | — | common_voice_25 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | no | read from nchlt_zulu instead, which has more trainable audio |
| nso | — | — | — | — | common_voice_25 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | no | read from nchlt_sepedi instead, which has more trainable audio |
| xh | — | — | — | — | common_voice_25 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | no | read from nchlt_xhosa instead, which has more trainable audio |
| dsb | — | — | — | — | common_voice_25 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | no | trainable 0.0 h < 50.0 h |
| hr | — | — | — | — | common_voice_25 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | no | read from parlaspeech_hr instead, which has more trainable audio |
| nr | — | — | — | — | common_voice_25 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | no | trainable 0.0 h < 50.0 h |
| ss | — | — | — | — | common_voice_25 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | no | trainable 0.0 h < 50.0 h |
| st | — | — | — | — | common_voice_25 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | no | trainable 0.0 h < 50.0 h |
| ts | — | — | — | — | common_voice_25 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | no | read from nchlt_xitsonga instead, which has more trainable audio |
| ve | — | — | — | — | common_voice_25 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | no | read from nchlt_tshivenda instead, which has more trainable audio |
