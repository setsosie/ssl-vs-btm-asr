# Text normalization

Every transcript in this repository passes through a normalizer chosen for the
script it is written in, defined in `src/svb/text/normalize.py` and assigned in
`src/svb/text/registry.py`. Whichever policy a language gets is applied at
exactly three places —
when the character vocabulary is built, when training targets are encoded, and
when references and hypotheses are scored — so training and evaluation cannot
drift apart on text policy. The policy each language ran under is recorded in
every run's `resolved_config.yaml` and `env.json`, and its effect is measured
per language in `text_stats.json`.

The rules below are those of `svb-norm-1`, the single policy this repository
used before the families existed. It is still available by name and is what the
comparability note at the end describes; which policy each language actually
uses now is the table further down.

## What it does, in order

1. **NFKC.** Unifies fullwidth and halfwidth Japanese, Arabic presentation
   forms, and Latin ligatures, which are acoustically identical but encode
   differently. Note that NFC and NFKC both *decompose* the precomposed Indic
   nukta letters, because they are Unicode composition exclusions: Odia `ଡ଼`
   written as U+0B5C is stored and counted as U+0B21 U+0B3C. Character counts,
   and so character error rates, are over the decomposed form.
2. **Malayalam chillu letters** written as consonant + virama + zero-width
   joiner are mapped to their atomic code points, U+0D7A–U+0D7F. No Unicode
   normalization form does this, and without it one letter has two encodings in
   the vocabulary.
3. **Invisible characters removed** — zero-width joiner and non-joiner, soft
   hyphen, byte-order mark, bidi controls. They carry no acoustic content and a
   CTC model cannot align them to audio, so it would learn to emit them from
   context alone. They are deleted rather than replaced with a space, because
   they sit inside words. NFKC does not remove the soft hyphen.
4. **Arabic vocalization removed** — tashkeel and harakat, plus the tatweel
   elongation character U+0640. Arabic vocalization is optional in writing, so
   it appears in some sentences and not others, and an inconsistently written
   target is unlearnable. The rule is scoped by code range rather than by
   language, because it is a property of the script. Tatweel is Unicode category
   `Lm`, so removing it takes an explicit rule.
5. **Case folded** with `str.casefold()`. German `ß` folds to `ss` and Georgian
   Mtavruli capitals fold to Mkhedruli. For the scripts used here this choice
   decides nothing else: `str.lower()` and `str.casefold()` differ in twelve
   code points, all Latin, and in none at all across Cyrillic, Georgian, Arabic,
   Indic or Japanese.
6. **Turkish dotted capital I repaired.** `"İ".casefold()` produces `i` followed
   by a combining dot above, so without this step `İstanbul` and `istanbul` are
   different words and U+0307 enters the vocabulary. A capital dotless `I` still
   folds to a dotted `i`, which is wrong for Turkish; fixing that needs a
   locale-aware mapping, and Common Voice's collection rules reject the all-caps
   sentences where it would mostly appear, so the residual is accepted and
   measured rather than fixed.
7. **Apostrophes unified** to U+0027, then punctuation (Unicode category `P*`)
   and symbols (`S*`) are replaced with a space. Unification comes first because
   Common Voice's French and Italian text uses U+2019 far more than the ASCII
   form, and because U+02BC is category `Lm` and would otherwise survive the
   punctuation pass as a stray letter. Apostrophes *between* word characters
   survive, so French `l'été`, Ukrainian `п'ять` and Swahili `ng'ombe` stay one
   word each. Hyphens and dashes are category `Pd`, so hyphenated words become
   two words and no separate dash rule is needed. Punctuation becomes a space
   rather than vanishing, so a source that omits the space after a comma does
   not end up with two words fused into one.

   A word-initial or word-final apostrophe is removed. For the languages here
   that is silent orthography — an English possessive, an Italian truncation —
   but if you add a language that writes a glottal stop as `'`, set
   `apostrophe_is_letter: true` in the policy and remove U+02BC and U+02BB from
   the apostrophe map, where they are letters in earnest.
8. **Digits kept as characters.** They are not spelled out and utterances
   containing them are not dropped. Common Voice's sentence validator rejects
   sentences containing digits, so the rate is low; its default rule matches
   ASCII only, so Devanagari and Arabic-Indic digits do pass, and OpenSLR
   applies no such rule at all. `text_stats.json` reports the rate per language
   so the assumption is checkable.
9. **Whitespace collapsed** to single spaces and stripped.

Combining marks are otherwise **kept**. This is a deliberate departure from
Whisper's `BasicTextNormalizer`, which replaces every character in Unicode
categories `M`, `S` and `P` with a space. That destroys every Indic script:
`हिंदी में` becomes `ह  द  म  `. We also protect the intra-word apostrophe,
where that normalizer turns `don't` into `don t`, and we do not delete
bracketed spans, which would silently remove real words from corpora that use
no such convention.

Category rules alone are not sufficient in either direction. Three characters
this policy must remove are `Lm` rather than punctuation — tatweel and the two
modifier apostrophes — and one it must keep is `Lm` too: U+30FC, the Japanese
prolonged sound mark, which is a vowel-length letter, so `コーヒー` survives.

## Vocabulary

Character-level. Id 0 is the CTC blank, id 1 is `<unk>`, then one id per
normalized character in sorted order. The space is an ordinary entry, not a
special delimiter token: for CTC there is no functional difference, and a
literal space means a decode is directly scoreable with no post-processing step
to get wrong.

Characters occurring fewer than `text.min_char_count` times across the training
corpus are evicted and listed with their counts in `text_stats.json`. The
default is 1, which keeps everything and is right for a smoke run where a real
character may also be rare; `configs/base.yaml` raises it for a full run, where
the tail is stray characters from corpora that had no collection-time
validation. The floor treats the space like any other character, so a floor
tuned for a full run can evict it from a small corpus; the vocabulary build
refuses that outcome for any preset containing a space-separated language.

Adapting to a held-out language appends its new characters to the end, so every
trained head row keeps its meaning and only the new rows are randomly
initialized. The expansion reuses the vocabulary's stored policy and refuses to
run under a different one, because the new language's characters would
otherwise be stored in one form while scoring produced another.

`vocab.json` stores the policy alongside the characters. Files written before
this policy existed were a bare JSON list built with NFC alone; they still load,
and report themselves as `svb-norm-0` rather than claiming the current policy.

## Scoring

References are the normalized transcripts the collate produced, never a
reconstruction from label ids — a reconstruction silently drops characters that
mapped to `<unk>` and makes the reported error rate optimistic. Hypotheses pass
through the same normalizer, which is close to a no-op except that it collapses
the leading, trailing and doubled spaces greedy CTC does produce.

Utterances whose reference normalizes to nothing are dropped from training and
excluded from scoring, and both counts are reported: an empty reference adds
nothing to the error rate's denominator and its whole hypothesis to the
numerator, so keeping one inflates the corpus number with no weight behind it. A
split where nothing at all is scoreable raises rather than reporting a confident
zero.

Both word and character error rates are reported for every language. Which is
primary follows the `word_boundary` field in `configs/scales/*.yaml`; Japanese
is the only preset language written without spaces. Each run checks that
declaration against its own transcripts and warns when they disagree, since
whitespace tokenization of an unspaced script yields one token per sentence and
a word error rate over it can only be 0 or 100 per utterance. Bootstrap
confidence intervals and paired permutation tests take the same tokenization, so
their point estimates equal the reported metric.

## What a run records

`resolved_config.yaml` carries the whole policy plus two derived digests: the
policy hash, which identifies the settings, and a hash of the normalizer module,
which catches the module being edited without any setting changing. `env.json`
adds the Unicode version, because `unicodedata` decides both character
categories and case folding, so two Python builds can normalize one transcript
differently.

`text_stats.json` holds, per language and per split: utterance and character
counts before and after normalization, how many utterances normalized to
nothing, how many carry digits, what each rule removed by category, the median
whitespace tokens per utterance alongside the declared `word_boundary`, and the
unknown-character rate against the vocabulary that language is actually scored
with. That last number is an irreducible floor under the language's error rate,
because a character the vocabulary lacks cannot be produced however good the
model is, and a run warns when it rises above a tenth of a percent rather than
leaving it in a file nobody opens.

The removal counts are tallied by the normalizer as it works, so they describe
the transformation that happened rather than what a scan of the input would
guess: an intra-word apostrophe that survives is not counted as removed, and a
rule the policy switched off removes nothing.

## Which policy each language uses

Normalization is per script family, not per run. A policy exists because of a
writing system: the reason Devanagari needs its combining marks kept is the
reason Telugu does, and it has nothing to do with Hindi or Telugu in particular.
Each family's policy follows the convention of a reference system for that
family, rather than one global guess applied to every script alike.

| Policy | Languages here | Follows | Source |
|---|---|---|---|
| `whisper-basic` | ab ady ba be ca cs cy de en eo es et eu fi fr fy-NL gl hr hu is it ka kbd kk lv mhr nl pl pt ru uk | OpenAI Whisper, exactly | Radford et al. 2023, appendix C |
| `turkic-tr` | tr | Whisper plus a Turkish locale case pre-map | our fix; see below |
| `latin-marks` | jv kab kmr lg nso rw su sw ts uz ve xh zu | HuggingFace Open ASR Leaderboard `remove_symbols_keep_marks` | `huggingface/open_asr_leaderboard` |
| `indic-vistaar` | bn hi kn ne si ta, and held-out gu ml mr te | AI4Bharat Vistaar / IndicWhisper | `AI4Bharat/vistaar` `evaluation.py` |
| `arabic-ouaal` | ar | Open Universal Arabic ASR Leaderboard | arXiv:2412.13788 |
| `perso-arabic` | ckb fa ps ur | `hazm` and `urduhack` Perso-Arabic conventions | see the caveats |
| `uyghur-ug` | ug | Perso-Arabic without the alef-maksura fold | see the caveats |
| `armenian-hy` | hy | Whisper plus deletion of the in-word marks | our fix; see below |
| `ja-cer` | ja | ReazonSpeech evaluation, character error rate | `reazon-research/ReazonSpeech` |
| `thai-cer` | th | PyThaiNLP and Thonburian Whisper, character error rate | `biodatlab/thonburian-whisper` |
| `han-mer` | yue zh-CN | WeNet `compute-wer.py`, character-level scoring | `wenet-e2e/wenet`; WenetSpeech, arXiv:2110.03370 |
| `ko-kspon` | ko | KsponSpeech, word and character error rate | Bang et al. 2020 |
| `tibetan-syllable` | bo | nothing — reasoned from the orthography | see the caveats |

Kazakh takes Whisper's normalizer rather than the ё-fold. That fold is a
Russian convention drawn from Russian corpora, and importing it into Kazakh
would be applying another language's rule. No Kazakh normalization convention
could be sourced — the corpus paper reports both error rates and describes no
text preprocessing — but the script is verified safe: Kazakh Cyrillic carries no
combining marks, so Whisper's mark rule finds nothing to remove.

Sinhala is assigned by family rather than by its reference system: the Indic
normalizer Vistaar calls has no Sinhala class and falls back to a
whitespace-only base. Bengali, Kannada and Nepali are within its language set.

Also shipped and not assigned to anything: `whisper-marks`, which is Whisper's
own pipeline with the three fixes the community has already made to it and the
base the family policies extend; `cyrillic-yo`, which folds ё to е as the Golos
and Russian Open STT conventions do, available for a run that wants it;
`whisper-basic-nodiacritics`, Whisper's `remove_diacritics` variant; and
`svb-norm-1`, the single policy this repository used before the families
existed, kept because the comparability note below describes it.

### What the mark rule costs, and the one European exception

Whisper replaces every character in Unicode category M with a space. For Latin
and Cyrillic that is nearly inert, because those scripts compose their accents
into single code points. For an abugida it deletes the vowels, and for Turkish
it splits words:

| Input | `whisper-basic` | the assigned policy |
|---|---|---|
| `यह हिंदी है।` (hi) | `यह ह द ह ` — 4 tokens | `यह हिंदी है` — 3 tokens |
| `मलयाळം ഭाഷ` (ml) | consonant skeleton only | marks intact |
| `مَرْحَبًا بِٱلْعَالَم` (ar) | 10 single-letter tokens | `مرحبا بالعالم` — 2 tokens |
| `İstanbul'da yaşıyorum.` (tr) | `i stanbul da yaşıyorum ` — 4 tokens | `istanbul'da yaşıyorum` — 2 tokens |
| `İyi günler` (tr) | `i yi günler` — 3 tokens | `iyi günler` — 2 tokens |

Turkish is the one European-script language that does not use `whisper-basic`.
Lowercasing `İ` yields `i` followed by a combining dot above, which the mark
rule then turns into a space, so every sentence-initial `İ`-word is split in
two. `turkic-tr` maps `İ` to `i` and `I` to `ı` before folding, which is what a
Turkish locale does. No Turkish ASR benchmark was found that specifies this, so
it is our fix rather than an inherited convention.

### Thai, Chinese and Cantonese

**Thai uses NFC, never NFKC.** Thai SARA AM `ำ` U+0E33 is a *letter*, category
`Lo`, and NFKC decomposes it into a combining mark plus a vowel. `ทำงาน` is five
code points under NFC and six under NFKC, so the compatibility form lengthens
every word containing it — moving the character error rate denominator — and
manufactures a mark for a mark rule to find. Its zero-width rule maps U+200B to
a space rather than deleting it, because that is where Thai puts its word
boundary.

Thai has two live conventions and they disagree. Whisper's appendix C measures
character error rate. Thonburian Whisper, the strongest published Thai system,
reports word error rate after `deepcut` segmentation. Character error rate is
primary here, because putting a segmenter inside the metric makes the number
depend on a third-party model version; the policy preserves the tone marks a
`deepcut` word error rate could later be computed from. PyThaiNLP's tone-mark
reordering and SARA AM composition are **not** adopted: they are corpus-cleaning
rules, and applying them to a hypothesis would silently repair model errors.

**Chinese and Cantonese are scored on characters, and that is not what WeNet
reports.** The de-facto script, WeNet's `compute-wer.py`, splits Han characters
individually but keeps Latin runs whole, and WenetSpeech names the result
Mixture Error Rate: *"which considers Mandarin characters and English words as
the tokens in the edit distance calculation"*. This repository has no such
metric — its tokenizers are words or characters — so Chinese and Cantonese are
reported on characters, which is the closer of the two.

The two agree on pure Han text and diverge on code-mixed utterances, because
character scoring charges an English word once per letter. On one example:

| | reference `我用 Python 写代码。` vs hypothesis `我用 Java 写代码。` |
|---|---|
| Mixture Error Rate (WeNet) | 1 substitution over 6 tokens — **16.7%** |
| character error rate (here) | 6 edits over 13 characters — **46.2%** |

So a Chinese number from this repository is not comparable with a published
Mixture Error Rate on material that mixes scripts, and is roughly comparable on
material that does not. Adding the third tokenization is the fix; until then the
tables say `CER` and mean it.

**Traditional is preserved, not folded to Simplified.** Unicode normalization
does not touch it — U+9AD4 `體` is stable under NFKC — no Chinese benchmark
converts, and MDCC keeps Traditional for Cantonese. WenetSpeech-Yue does fold
with OpenCC, so Cantonese is genuinely contested; if `yue` and `zh-CN` ever
share a vocabulary, not folding means two character sets for one spoken family.

### Armenian, Korean and Tibetan

**Armenian writes punctuation inside the word.** Its question and emphasis
marks sit on the stressed syllable rather than at the end of the clause, so the
rule that turns punctuation into a space splits every question in two:
`Ի՞նչ կա։` becomes `ի նչ կա`. `armenian-hy` deletes U+055B–U+055F before that
rule runs, giving `ինչ կա`. This is our fix; no Armenian ASR evaluation
convention was found. NFKC's rewrite of the ligature `և` U+0587 into two letters
is left in place, applied identically to both sides of every score, which is why
`Բարև ձեզ` is expected to read `բարեւ ձեզ`.

**Korean uses NFC and is scored on words.** Two compatibility jamo compose into
one syllable under the compatibility form — `ㄱㅡ 그` is four code points under
NFC and three under NFKC — which would move the character count for a language
whose corpus reports character error rate. Word error rate is primary here,
unlike the other character-scored languages, because Korean is written with
spaces. Its spacing is flexible enough that the word error rate is inflated by
choices that are not recognition errors; KsponSpeech answers that with a
space-normalized variant that rewrites the hypothesis toward the reference,
which is a scoring convention this repository would have to defend separately
and does not adopt. Read Korean word error rate as an upper bound.

**Tibetan is an abugida written without spaces.** Its vowel signs and subjoined
consonants are combining marks, so Whisper's rule reduces a word to its root
letters: `བོད་སྐད་ཡིན།` becomes `བ ད ས ད ཡ ན`. The tsheg U+0F0B, which separates
syllables, is punctuation and so becomes a space — that costs no characters,
since it is one code point either way, and it leaves the syllable boundary
visible to anything that later wants to count syllables rather than characters.

### Three policies whose base could not be verified

**Uyghur.** No Uyghur ASR evaluation convention was found at all. `uyghur-ug` is
constructed by reasoning from the orthography — Uyghur writes /i/ with alef
maksura and /j/ with yeh, so the Perso-Arabic fold of the first onto the second
would merge two distinct letters — and not adopted from anything published. It
is the weakest policy in the set. Treat any Uyghur number as provisional until
a native speaker or a published evaluation confirms the rules.

**Tibetan.** The corpus this policy exists for states no evaluation metric and
no transcription convention, and no Tibetan ASR normalizer was found at all.
`tibetan-syllable` is reasoned from the orthography, exactly as `uyghur-ug` is.
Treat any Tibetan number as provisional.

**Central Kurdish.** `ckb` is assigned by script family, exactly as Pashto is:
it is written in the Arabic script and the Perso-Arabic conventions are the
closer of the two available. Sorani has letters and a vowel system neither
Persian nor Urdu uses, and no Sorani ASR normalizer was consulted. Unverified
for this language specifically.

**Pashto.** `ps` is assigned by script family rather than from a Pashto-specific
source: it is written in the Arabic script and the Perso-Arabic conventions are
the closer of the two available. Pashto has letters neither Persian nor Urdu
uses, and no Pashto ASR normalizer was consulted. Unverified for this language
specifically.

### Overriding a language, or a whole run

Each preset states its policy on the language's own line, so the choice is
visible where a preset author would look:

```yaml
- { code: tr, source: commonvoice, hf_config: tr, normalizer: turkic-tr }
```

Leave it out and the language falls back to a table in
`src/svb/text/registry.py`: first the per-language assignments, then the script
default. The language level exists because the Latin script cannot tell European
from non-European on its own — its European languages use Whisper's normalizer
and the rest need the mark-preserving one. Latin's script default is
`latin-marks` rather than `whisper-basic`, which fails safe in both directions: a European language added
without a line differs from Whisper only in ways Latin text barely notices,
while a Yoruba or Vietnamese one keeps its tone marks. The Arabic script has no
default at all, because its two conventions fold letters in opposite directions
and every guess is wrong for half the family; a language written in it must name
its policy or the run refuses to start.

To score an entire run one way — for comparing against a published system's
numbers, at the cost of what that system's rules do to the scripts it was not
designed for:

```yaml
text:
  override: whisper-basic
```

### What a run records, and what may be pooled

`resolved_config.yaml` lists the policy each language ran under by name and by
hash. `env.json` adds a digest over the whole policy set, so two runs can be
checked for text equivalence without walking the table, alongside the Unicode
version, which decides both character categories and case folding.
`text_stats.json` records the policy per language, because its removal tallies
are not comparable across languages that ran under different rules.

`svb analyze` refuses to compare two runs that normalized a language
differently, and refuses a run that records no policies at all — silence is not
agreement. A macro-average whose languages used different policies says so:
averaging across different text rules is how a multilingual system gets
summarised at all, but the result is not one quantity.

## Whisper's normalizer, and the switch

Most published multilingual ASR numbers are scored with OpenAI Whisper's
`BasicTextNormalizer` (Radford et al. 2023, appendix C). Comparing against them
means scoring the way they were scored, so this repository ships that
normalizer as the `whisper-basic` policy, reproduced exactly rather than
approximated. Every expected string in its test table was produced by running
the upstream code, not by reading it.

Upstream lowercases, deletes `<...>`, `[...]` and `(...)` spans, applies NFKC,
replaces every character in Unicode categories M, S and P with a space,
lowercases again, and collapses runs of whitespace. `whisper-basic-nodiacritics`
is the `remove_diacritics=True` variant: NFKD, delete category Mn, and map the
sixteen letters NFKD does not separate — `œ ø æ ß đ ð þ ł` and their capitals —
through the upstream table.

### The three deviations

| | `svb-norm-1` (default) | `whisper-basic` |
|---|---|---|
| Unicode category M | kept | replaced with a space |
| Intra-word apostrophe | kept, so `don't` is one word | replaced, so `don't` becomes `don t` |
| Bracketed and parenthesized spans | kept | deleted, so `[noise] hello (laughs)` becomes ` hello ` |

Two smaller differences follow from reproducing upstream faithfully. Whisper
never strips, so its output routinely carries a leading or trailing space, which
is one character of character error rate. And it uses `str.lower()` rather than
`str.casefold()`, so German `ß` survives under `whisper-basic` and becomes `ss`
under `whisper-basic-nodiacritics`.

### What the first deviation costs

Replacing every mark with a space is not a cosmetic difference for an abugida.
An Indic vowel sign is category Mc or Mn, so the rule deletes the vowels and
leaves the consonant skeleton, splitting one word into several:

| | Hindi `यह हिंदी है।` | Malayalam `ഇത് മലയാളം ആണ്.` | vocalized Arabic `مَرْحَبًا بِٱلْعَالَم` |
|---|---|---|---|
| source | 12 chars, 3 words | 15 chars, 3 words | 21 chars, 2 words |
| `svb-norm-1` | 11 chars, 3 words | 14 chars, 3 words | 13 chars, 2 words |
| `whisper-basic` | 9 chars, 4 words | 12 chars, 4 words | 21 chars, 10 words |

The Arabic row is the starkest: harakat are category Mn, so a fully vocalized
sentence is shattered into ten single-letter tokens. This project's own rule
deletes those marks instead of spacing them, which is why its output there is
two words.

The direction of the resulting bias is not intuitive. Because the vowel
information is removed from the reference *and* the hypothesis, a model that
gets every vowel sign wrong can no longer be marked wrong for it, so measured
accuracy improves for reasons that have nothing to do with the model. Manohar,
Pillai and Sherly ("What is lost in Normalization? Exploring Pitfalls in
Multilingual ASR Model Evaluations", EMNLP 2024, arXiv:2409.02449) survey the
normalizers used by Whisper, MMS, Seamless and Conformer and find this class of
routine "fundamentally flawed when applied to Indic scripts", producing
"artificially improved performance metrics"; they recommend normalization
grounded in native linguistic expertise.

### How to choose, and how to switch

Use `whisper-basic` when the comparison is against published Whisper numbers
**and** the languages being scored are Latin or Cyrillic, where the three
deviations touch little: those scripts carry few combining marks once NFKC has
composed them, so the mark rule is close to inert. Do not use it for any result
that includes Indic or Arabic scoring — the numbers it produces there are not
measuring what they appear to measure, and are not comparable to this project's
own.

The default stays `svb-norm-1`. Switching is one line of config:

```yaml
text:
  policy: whisper-basic
```

The name is recorded in `resolved_config.yaml` as `policy_name`, beside the
resolved rule set and the policy hash. The hash is what makes the choice
visible after the fact: results produced under two different policies carry two
different hashes and must not be pooled.

## Changing the policy

Bump `NORMALIZER_VERSION` for any change in behaviour. Results produced under
different policy hashes are not comparable and must not be pooled.
