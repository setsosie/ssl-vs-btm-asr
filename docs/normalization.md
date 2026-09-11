# Text normalization

Every transcript in this repository is normalized with OpenAI Whisper's
`BasicTextNormalizer`, reproduced exactly, unless that normalizer is
demonstrably destructive or word-breaking for the language's orthography.
Twelve script families and four Latin languages are such cases and each is
listed below with the failure that puts it there. Policies are defined in
`src/svb/text/normalize.py` and assigned in `src/svb/text/registry.py`.

Whichever policy a language gets is applied at exactly three places — when the
character vocabulary is built, when training targets are encoded, and when
references and hypotheses are scored — so training and evaluation cannot drift
apart on text policy. The policy each language ran under is recorded in every
run's `resolved_config.yaml` and `env.json`, and its effect is measured per
language in `text_stats.json`.

The rule list immediately below is `svb-norm-1`, the single policy this
repository used before any of this existed. It is still reachable by name, is
what the comparability note at the end describes, and is no language's policy
now. What the default actually does is in "What the default is, exactly".

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

## Whisper-basic by default; exceptions and why

Every language is normalized with OpenAI Whisper's `BasicTextNormalizer`,
reproduced exactly (Radford et al. 2023, appendix C). It is the convention most
published multilingual numbers are scored under, so it is what a reader will
compare against.

A language moves off it only where that normalizer is **demonstrably**
destructive or word-breaking for its orthography. Every such case is below with
the concrete failure and the convention adopted instead. Nothing is an exception
on grounds of taste.

The test that decides this is per language and lives in
`tests/test_whisper_default.py`: enumerate the orthography's non-ASCII letters,
normalize them, and require that none leaves a combining mark, that none is
punctuation or a symbol the mark rule would eat, and that a real phrase in the
language keeps its word count. A language cannot be put on the default without
passing it.

### The exceptions

| Language or script | Policy | What the default does to it | Convention adopted |
|---|---|---|---|
| Devanagari, Bengali, Gurmukhi, Gujarati, Odia, Tamil, Telugu, Kannada, Malayalam, Sinhala | `indic-vistaar` | Vowel signs are combining marks, so it deletes the vowels: `यह हिंदी है।` becomes `यह ह द ह ` | AI4Bharat Vistaar / IndicWhisper |
| Thai | `thai-cer` | Same mark defect; also deletes the zero-width word separator, and the compatibility form splits SARA AM into a mark plus a vowel | PyThaiNLP, Thonburian Whisper |
| Tibetan | `tibetan-syllable` | Same mark defect: stacked vowel signs and subjoined consonants vanish, `བོད་སྐད་ཡིན།` becomes `བ ད ས ད ཡ ན ` | none — reasoned from the orthography |
| Korean | `ko-kspon` | The compatibility form composes two jamo into one syllable, moving the character count | KsponSpeech |
| Armenian | `armenian-hy` | The question mark is written inside the word, so `Ի՞նչ կա։` becomes two words | none — our fix |
| Turkish | `turkic-tr` | Lowercasing `İ` yields a combining dot the mark rule spaces, so `İstanbul'da yaşıyorum.` becomes four words instead of two | none — our fix |
| Arabic (`ar`) | `arabic-ouaal` | Vocalization becomes spaces, shattering `مَرْحَبًا بِٱلْعَالَم` into ten single-letter tokens | Open Universal Arabic ASR Leaderboard |
| Persian, Urdu, Pashto, Central Kurdish | `perso-arabic` | Same, plus it deletes the non-joiner that marks a Persian morpheme boundary | `hazm`, `urduhack` |
| Uyghur | `uyghur-ug` | Same as Perso-Arabic, which would additionally fold away the letter Uyghur writes /i/ with | none — reasoned from the orthography |
| Swahili, Kinyarwanda | `latin-marks` | The apostrophe inside `ng'ombe` and `y'u` is punctuation, so one word becomes two | HuggingFace Open ASR Leaderboard |
| Yoruba, Igbo | `latin-marks` | A tone accent on a dot-below vowel has no single code point, so the tone is deleted | HuggingFace Open ASR Leaderboard |
| Japanese, Chinese, Cantonese | `ja-cer`, `han-mer` | Nothing — the text is unchanged. These exist for the metric, which is characters rather than words | ReazonSpeech; WeNet |

Everything else takes the default, including every European Latin, Cyrillic and
Georgian language, and the eleven Latin-script languages that were on
`latin-marks` until the check above cleared them: Luganda, Kabyle, Uzbek,
Northern Kurdish, Javanese, Sundanese, Zulu, Xhosa, Northern Sotho, Tsonga and
Venda. Two results from that check are worth stating, because they look like
they should fail and do not: Uzbek's `ʻ` U+02BB is a modifier letter, category
`Lm`, not punctuation, so it survives; and Kabyle's `ɛ ɣ ḥ ṭ ẓ č` are all single
`Ll` code points after normalization, so no mark is left behind.

A language named in neither table takes the default and **warns**, naming
itself. A default nobody sees is how a script ends up scored under rules written
for another one.

### Policies kept but unassigned

`latin-marks` (used only by the four Latin exceptions), `cyrillic-yo` (the
Russian ё-fold, which no language here adopts), `whisper-marks` (the base the
family policies extend) and `svb-norm-1` (what this repository used before the
families existed, and what the comparability note below describes) all remain
reachable by name.

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

## What the default is, exactly

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

### Overriding it

`whisper-basic` is already the default, so there is nothing to switch on to. The
override exists for the opposite move — forcing every language onto one policy,
exceptions included:

```yaml
text:
  override: whisper-basic
```

That is worth doing once, as a measurement: it says how much the exceptions are
worth by scoring the same predictions without them. It is not worth publishing.
The numbers it produces for Indic, Arabic, Thai and Tibetan are not measuring
what they appear to measure, for the reasons in the exception table.

A single language moves with the `normalizer:` line on its own row in
`configs/scales/*.yaml`. Every shipped preset states one, including where it
equals the default, so the assignment is auditable from the preset alone without
resolving anything.

Each run records the policy every language used, by name and by hash. The hash
is what makes the choice visible afterwards: results produced under different
hashes are not comparable and must not be pooled.

## Changing the policy

Bump `NORMALIZER_VERSION` for any change in behaviour. Results produced under
different policy hashes are not comparable and must not be pooled.
