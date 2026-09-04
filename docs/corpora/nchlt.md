# NCHLT (South African languages)

Ten languages in one format, fetched from SADiLaR's DSpace 7 repository by
[`scripts/prepare_nchlt.py`](../../scripts/prepare_nchlt.py). Five of them are in
the large preset: isiZulu (`zu`), isiXhosa (`xh`), Sepedi (`nso`), Xitsonga
(`ts`) and Tshivenda (`ve`). Afrikaans (`af`), isiNdebele (`nr`), Siswati (`ss`),
Sesotho (`st`) and Setswana (`tn`) ship identically and the preparer can fetch
them, but they are in no preset and are not recorded in `configs/corpora.yaml`,
so `--langs` has to name them.

```bash
export CORPORA_ROOT=/path/to/corpora
python scripts/prepare_nchlt.py --root $CORPORA_ROOT              # the preset five
python scripts/prepare_nchlt.py --root $CORPORA_ROOT --langs zu
```

Unlike the OpenSLR preparers this one imports `svb.data.splits`, so it needs the
project installed rather than only the standard library.

## What is downloaded

One zip per language, between 4.6 GB and 5.1 GB. There is no static URL: a
bitstream is resolved from the item handle at fetch time.

| Language | Code | ISO 639-3 | Handle | Archive | Bytes |
|---|---|---|---|---|---|
| isiZulu | zu | zul | `20.500.12185/275` | `nchlt.speech.corpus.zul.zip` | 4,683,189,802 |
| isiXhosa | xh | xho | `20.500.12185/279` | `nchlt.speech.corpus.xho.zip` | 5,013,413,061 |
| Sepedi | nso | nso | `20.500.12185/270` | `nchlt.speech.corpus.nso.zip` | 4,606,487,018 |
| Xitsonga | ts | tso | `20.500.12185/277` | `nchlt.speech.corpus.tso.zip` | 4,748,770,206 |
| Tshivenda | ve | ven | `20.500.12185/276` | `nchlt.speech.corpus.ven.zip` | 4,724,571,913 |
| Afrikaans | af | afr | `20.500.12185/280` | `nchlt.speech.corpus.afr.zip` | 4,841,810,323 |
| isiNdebele | nr | nbl | `20.500.12185/272` | `nchlt.speech.corpus.nbl.zip` | 4,669,151,540 |
| Siswati | ss | ssw | `20.500.12185/271` | `nchlt.speech.corpus.ssw.zip` | 4,675,934,111 |
| Sesotho | st | sot | `20.500.12185/278` | `nchlt.speech.corpus.sot.zip` | 5,015,044,724 |
| Setswana | tn | tsn | `20.500.12185/281` | `nchlt.speech.corpus.tsn.zip` | 5,147,538,851 |

The archive names use ISO 639-3 and the presets use their own language codes.
isiZulu is `zul` inside the archive and `zu` in the manifest tree; conflating the
two is the mistake this table exists to prevent.

## Access, which was the open question

SADiLaR's resources page mentions accepting terms of use, which would be a
click-through gate and would disqualify the corpus under this project's rule that
nothing behind a gate, a form or an email request is used. It is not one. The
DSpace REST API serves the bitstream to an unauthenticated client — no cookie, no
token, no Shibboleth — and every step below was run anonymously against the live
repository while writing the preparer:

```
GET /server/api/pid/find?id=hdl:20.500.12185/275   -> 302, then the item JSON
    _links.bundles                                  -> the ORIGINAL bundle
    _links.bitstreams                               -> one zip
    _links.content                                  -> 206, application/zip, PK magic
```

All ten handles resolve, and all ten answer with the same licence string and the
same description.

## Licence

Verbatim, from `dc.rights.license` on every one of the ten items:

> Creative Commons Attribution 3.0 Unported License (CC BY 3.0):
> http://creativecommons.org/licenses/by/3.0/legalcode

The preparer reads that field at fetch time and refuses to ingest a corpus whose
licence no longer contains "Creative Commons Attribution 3.0". A licence copied
into a config file is a claim about the past; this is the one moment the
repository can be asked directly.

`dc.description`, also read at fetch time and recorded beside the data:

> Orthographically transcribed broadband speech corpus of approximately 56 hours,
> including a test suite of 8 speakers.

## Integrity

NCHLT is the only corpus in this repository that publishes a checksum. The
bitstream JSON carries one:

```json
"checkSum": { "checkSumAlgorithm": "MD5", "value": "aba238a62d72932edb86bd1e5295ea60" }
```

so the archive is compared against a published value rather than merely digested
for later comparison, and a mismatch deletes it. The SHA-256 goes into
`fetch_manifest.json` as well, for the same provenance reason as everywhere else.

## Layout

Read out of one archive's zip central directory over HTTP Range — 44,891
members, no download — and confirmed by the `README.txt` inside it, which states
the structure as:

```
nchlt_<ISO 639-3>
├── audio
│   ├── <spk_id>
│   │   ├── nchlt_<ISO 639-3>_<spk_id><gender>_<file_number>.wav
│   ...
└── transcriptions
   ├── nchlt_<ISO 639-3>.trn.xml
   └── nchlt_<ISO 639-3>.tst.xml
```

The archive itself has no `nchlt_zul/` wrapper: its members are `LICENSE.txt`,
`README.txt`, `audio/<spk_id>/…` and `transcriptions/…` at the root. The
transcripts, however, name audio *with* that component, so the preparer strips it
rather than copying the attribute through — a manifest built the naive way would
point every row at a path that does not exist.

The transcript format, from the `.dtd` the archive ships beside each `.xml`:

```
<!ELEMENT corpus  ( speaker+ )>          name
<!ELEMENT speaker  ( recording+ )>       id age gender location
<!ELEMENT recording  ( orth )>           audio md5sum duration pdp_score
<!ELEMENT orth  ( #PCDATA )>
```

and one real entry, from `transcriptions/nchlt_zul.trn.xml`:

```xml
<corpus name="nchlt zul">
  <speaker id="001" age="22" gender="male">
    <recording audio="nchlt_zul/audio/001/nchlt_zul_001m_0001.wav"
               md5sum="2711c6cb4d4c910694b6c72407666007"
               duration="3.12" pdp_score="-0.6875">
      <orth>ulwazi oluthile mayelana</orth>
    </recording>
```

`location` is declared `#REQUIRED` by the DTD and is absent from the real
elements, so the preparer does not require it. The speaker id is the directory
number the audio sits in, which is also embedded in the file name together with a
gender letter (`001m`). `utt_id` is the wav basename, unique within a language.

**Audio.** 16-bit mono PCM WAV at 16 kHz, stated in the archive README and
consistent with every file size in the central directory. Nothing is resampled
here; the loader resamples.

## Split policy

Two sides ship, and only two: `.trn` and `.tst`. There is no dev file.

- **test** is the published test suite, used exactly as shipped. The README says
  its speaker ids are the integers 500–507, eight speakers. The preparer asserts
  that no speaker appears in both sides rather than assuming it.
- **validation** is *derived* from the training side with `svb.data.splits`, the
  same speaker-disjoint derivation the corpora that ship no split at all get. The
  three-way derivation returns train/validation/test; its validation slice
  becomes dev and the other two are folded back into train, which puts dev near a
  tenth of the training side.
- **train** is everything else.

Deriving in the preparer rather than in the loader is forced. The manifest layer
refuses a partition that is part shipped and part derived, and clearing the
column so the loader could derive one would throw away the published test suite —
the only partition the release actually publishes.

**This is not the dev set the literature uses.** The per-language train/dev/test
hours recorded in `configs/corpora.yaml` come from van der Westhuizen & Niesler
(Stellenbosch SU-EE-1501), which divides the training side three ways. That
division is not in the release. A write-up comparing against those numbers has to
say that the dev set differs; the test set does not.

## Caveats

- **Scripted prompts, narrow domain.** Every utterance is a read prompt. A result
  that improves here should be reported as improving on read speech, not on
  speech. The whole non-Common-Voice addition skews this way and NCHLT is the
  largest block of it.
- **The test side is eight speakers.** For every language. A per-language test
  number rests on eight voices, and the variance that implies belongs next to it.
- **Train alone is under 50 hours for several languages.** isiZulu 48.5 h,
  isiXhosa 49.4 h, Xitsonga 49.8 h, Tshivenda 49.6 h against a ~56 h corpus
  total. The 50-hour rule is met on the corpus, not on the published train side.
- **Speaker overlap between languages is not checked.** The NCHLT collection ran
  one protocol across eleven languages and the README's "nchlt-baseline (no
  duplicate speakers)" variant exists precisely because duplicates were a known
  issue. Nothing here establishes that a speaker in isiZulu is not also in
  Sepedi, and a multilingual mix makes that a live question rather than a
  theoretical one.
- **This is the "nchlt-clean" release.** The README says so: problematic
  utterances are already removed, and two other cuts of the same collection exist
  (`nchlt-raw`, `nchlt-baseline`).

## Citation

> Etienne Barnard, Marelie H. Davel, Charl van Heerden, Febe de Wet and Jaco
> Badenhorst, "The NCHLT Speech Corpus of the South African languages," in Proc.
> 4th International Workshop on Spoken Language Technologies for Under-resourced
> Languages (SLTU), St Petersburg, Russia, May 2014.

Distributed by SADiLaR; collected for the South African Department of Arts and
Culture by CSIR Meraka and North-West University.
