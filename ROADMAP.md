# Roadmap

What is planned here, in the order it is planned, and why each item comes before
the next. Ordering is by dependency, not by date.

## Where this stands

**No GPU run has been executed against this code.** The CPU test suite passes and not
one number has come from real audio. `README.md` § Status records the four gaps that
follow from that, and this file says what is being done about them.

Everything before "The first run" below exists to make that run *trustworthy*, not
merely possible. The repository can already produce a number; the work is making it
a number worth reporting.

## Before the first run

### 1. Make the encoder the reference implementation

`src/svb/model/xeus_standalone.py` re-expresses the XEUS encoder in PyTorch so the
published checkpoint loads without ESPnet, and every number this repository would
produce currently comes out of it. `docs/espnet-crosscheck.md` states why that is a
risk in the repository's own words: an earlier version of that file had E-Branchformer
forward deviations from the reference that invalidated training results before anyone
noticed — the encoder ran, loss went down, and the features were wrong.

Spending GPU hours against an unverified reimplementation is how that happens twice.
The hybrid backend (reference ESPnet when the fork is installed, standalone otherwise)
retires the question before any hours are spent.

Tracked in [#1](https://github.com/setsosie/ssl-vs-btm-asr/issues/1). A partial
implementation exists on a branch and needs rebasing onto current `main`.

### 2. Answer the two staging questions that metadata can answer

Ten corpus preparers are implemented and **none has been run against a real archive**.
Their format claims come from reading zip central directories and tar header chains
over HTTP Range — real evidence, but not an ingest.

Two are known to be unresolved, and both are answerable without staging anything:

- **Tibetan's transcript convention is unverified.** `prepare_tibmd.py` discovers it and
  refuses rather than guessing, which is correct behaviour and stays. This supplies the
  answer it is refusing to guess.
- **Samrómur's `info.txt` describes only train and dev**, so its test split is unconfirmed
  until ingest.

Answering these from a laptop is cheaper than discovering them mid-ingest on a server.

### 3. Publish a staging order and a peak-disk figure

TalTech Estonian needs roughly 320 GB of peak disk — its 159 GB tar and its extracted
tree exist at once, and its half-hour recordings are then cut at transcript bounds into
roughly 600,000 small WAVs. ParlaSpeech needs about 116 GB twice over. The small-file
count matters independently of the byte count: inode pressure and small-file write
throughput are what make that stage slow, and neither shows up in a gigabyte figure.

`README.md` currently says "budget before starting either", which is an instruction to
a reader rather than a plan. `configs/corpora.yaml` already records what a real plan
would be computed from.

### 4. Say four held-out languages, everywhere

Odia is not in the held-out set; the set is four languages. `README.md` says so and says
any write-up should say four. A note in one file that every other surface must be
reconciled against by hand is the same failure mode as the staging budget — correct, and
not enforced. The count should have one definition, and drift should fail a test rather
than survive into a paper.

## The first run

Arms **A** (SSL-only fine-tune), **B** (BTM-on-SSL) and **C** (BTM-from-scratch), across
scales, multi-seed, with mean±std, bootstrap intervals and paired permutation tests —
the design `README.md` § Evaluation protocol describes. Smallest scale and cheapest arm
first, so that partial results degrade gracefully rather than yielding nothing.

## After the first run

- **Cross-check the encoder against the reference.** The check already exists
  (`docs/espnet-crosscheck.md`, `scripts/crosscheck_espnet.py`); it needs the checkpoint
  and the reference backend from item 1.
- **Stage the 64-language tier end to end**, in the order item 3 produces.

## Not planned

- Re-running or re-scoring results from other repositories. Numbers produced here are
  produced here, under this repository's normalization policy, and are not comparable
  to numbers produced elsewhere.
- Corpora behind gated access or licences that forbid redistribution. The tier is
  public-data-only by design.
