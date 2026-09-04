# Text normalization

Every transcript in this repository passes through one normalizer, `svb-norm-1`,
defined in `src/svb/text/normalize.py`. It is applied at exactly three places —
when the character vocabulary is built, when training targets are encoded, and
when references and hypotheses are scored — so training and evaluation cannot
drift apart on text policy. The active policy is recorded in every run's
`resolved_config.yaml` and `env.json`, and its per-language effect is measured
in `text_stats.json`.

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
