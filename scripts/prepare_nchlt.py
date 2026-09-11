"""Prepare the NCHLT South African corpora into the manifest layer.

Ten languages, one archive layout, one preparer. The five the large preset draws
on are isiZulu, isiXhosa, Sepedi, Xitsonga and Tshivenda; Afrikaans, isiNdebele,
Siswati, Sesotho and Setswana ship in exactly the same shape and are reachable
here by name, though they are not in ``configs/corpora.yaml`` and not in the
preset, so ``--langs`` defaults to the five that are.

    python scripts/prepare_nchlt.py --root $CORPORA_ROOT              # the preset five
    python scripts/prepare_nchlt.py --root $CORPORA_ROOT --langs zu
    python scripts/prepare_nchlt.py --root $CORPORA_ROOT --langs af st tn

**Access.** SADiLaR's resources page mentions accepting terms of use, but the
DSpace 7 bitstream endpoint answers an unauthenticated client. A handle resolves
through ``/server/api/pid/find?id=hdl:<handle>`` to the item, then
``_links.bundles`` -> the ``ORIGINAL`` bundle -> ``_links.bitstreams`` -> the one
zip's ``_links.content.href``. The same JSON carries ``dc.rights.license``,
``dc.description`` and an MD5 for the archive, so all three are asserted at fetch
time rather than trusted from ``configs/corpora.yaml``. NCHLT is the only corpus
in this repository that publishes a checksum; the archive is checked against it.

**Layout**, verified by reading archives' central directories over HTTP Range
rather than downloading them, and confirmed by the ``README.txt`` inside::

    [nchlt_<iso>/]LICENSE.txt
    [nchlt_<iso>/]README.txt
    [nchlt_<iso>/]audio/<spk_id>/nchlt_<iso>_<spk_id><gender>_<file_number>.wav
    [nchlt_<iso>/]transcriptions/nchlt_<iso>.trn.xml   (+ .trn.dtd)
    [nchlt_<iso>/]transcriptions/nchlt_<iso>.tst.xml   (+ .tst.dtd)

That leading component is optional because **the ten archives are not packed the
same way**. Nine wrap everything in ``nchlt_<iso>/``, which is what the README
documents; isiZulu's puts ``audio/`` and ``transcriptions/`` at the root. Checked
on all ten by reading the first local file header of each. So the root is
detected after extraction rather than assumed either way — assuming would leave
nine languages, or one, with a manifest whose every path is wrong.

``<iso>`` is ISO 639-3 and is *not* the language code the preset uses: isiZulu is
``zul`` in the archive and ``zu`` in the manifest tree. The transcripts are one
XML file per side, shaped by the shipped DTD::

    <corpus name="nchlt zul">
      <speaker id="001" age="22" gender="male">
        <recording audio="nchlt_zul/audio/001/nchlt_zul_001m_0001.wav"
                   md5sum="..." duration="3.12" pdp_score="-0.6875">
          <orth>ulwazi oluthile mayelana</orth>

The ``audio`` attribute always carries the ``nchlt_<iso>/`` component whether or
not the archive does, so it is stripped from the attribute and the prefix the
archive really used is put back.

**Split.** Only two sides ship: ``.trn`` and ``.tst``. The README states the test
suite is speaker ids 500-507, eight speakers, and this preparer asserts the two
sides share no speaker rather than assuming it. So the published test suite
becomes ``test`` untouched, and a dev split is *derived* from the training side
with ``svb.data.splits`` — the same speaker-disjoint derivation the corpora that
ship nothing get. The three-way per-language hours quoted in the literature come
from a further division of the training side that the release does not ship;
this is a different dev set from theirs, and a write-up should say so.

Deriving in the preparer rather than in the loader is forced: the manifest layer
refuses a partition that is part shipped and part derived, and leaving the whole
column empty would throw away the published test suite.

Audio is 16 kHz 16-bit mono PCM WAV, stated in the archive README and consistent
with every file size in the central directory. No resampling happens here.

Unlike ``prepare_google_crowdsourced.py`` this preparer imports ``svb``, so it
needs the project installed rather than only the standard library.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.request import Request, urlopen

from corpus_fetch import (
    CHUNK,
    UA,
    archive_record,
    download,
    extract_archive,
    verify_archive,
    write_fetch_manifest,
)

from svb.data.splits import SPLITS, derive_splits

API_ROOT = "https://repo.sadilar.org/server/api"
MANIFEST_COLUMNS = ("utt_id", "path", "text", "speaker", "split")

#: The licence every NCHLT item states today. A change here is a decision for a
#: person, not something to ingest through, so it stops the run.
EXPECTED_LICENCE = "Creative Commons Attribution 3.0"


@dataclass(frozen=True)
class Corpus:
    corpus_id: str
    lang: str  # the preset's language code, and the manifest directory
    iso: str  # ISO 639-3, which is what the archive's own names use
    handle: str


#: All ten. `handle` was resolved against the DSpace API for each one; the item
#: names and ISO codes below are the API's own.
CORPORA = (
    Corpus("nchlt_zulu", "zu", "zul", "20.500.12185/275"),
    Corpus("nchlt_xhosa", "xh", "xho", "20.500.12185/279"),
    Corpus("nchlt_sepedi", "nso", "nso", "20.500.12185/270"),
    Corpus("nchlt_xitsonga", "ts", "tso", "20.500.12185/277"),
    Corpus("nchlt_tshivenda", "ve", "ven", "20.500.12185/276"),
    Corpus("nchlt_afrikaans", "af", "afr", "20.500.12185/280"),
    Corpus("nchlt_ndebele", "nr", "nbl", "20.500.12185/272"),
    Corpus("nchlt_siswati", "ss", "ssw", "20.500.12185/271"),
    Corpus("nchlt_sesotho", "st", "sot", "20.500.12185/278"),
    Corpus("nchlt_setswana", "tn", "tsn", "20.500.12185/281"),
)
BY_LANG = {c.lang: c for c in CORPORA}
LANGUAGES = tuple(c.lang for c in CORPORA)
CORPUS_IDS = tuple(c.corpus_id for c in CORPORA)

#: The five that `configs/corpora.yaml` records and the large preset reads. The
#: other five work, but nothing asks for them, and each is another 4.7 GB.
PRESET_LANGUAGES = ("zu", "xh", "nso", "ts", "ve")


@dataclass(frozen=True)
class Utterance:
    utt_id: str
    path: str  # relative to the language directory
    text: str
    speaker: str


# --- the DSpace 7 API ---------------------------------------------------------


def api_get(url: str) -> dict[str, Any]:
    """One JSON document from the repository API.

    ``pid/find`` answers 302 to the item it resolves; urllib follows that, which
    is the whole reason a handle can be the thing recorded in the config.
    """
    request = Request(url, headers={**UA, "Accept": "application/json"})
    with urlopen(request, timeout=60) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise ValueError(f"{url}: expected a JSON object, got {type(payload).__name__}")
    return payload


def resolve_item(handle: str) -> dict[str, Any]:
    return api_get(f"{API_ROOT}/pid/find?id=hdl:{handle}")


def metadata_value(item: dict[str, Any], key: str) -> str:
    values = item.get("metadata", {}).get(key) or []
    return str(values[0].get("value", "")) if values else ""


def check_licence(corpus: Corpus, item: dict[str, Any]) -> tuple[str, str]:
    """The item's licence and description, refusing a licence that has changed.

    Both are read here rather than taken from ``configs/corpora.yaml`` because a
    licence recorded in a file is a claim about the past. This is the one moment
    the repository can be asked directly.
    """
    licence = metadata_value(item, "dc.rights.license")
    description = metadata_value(item, "dc.description")
    if EXPECTED_LICENCE not in licence:
        raise ValueError(
            f"{corpus.lang}: SADiLaR now states the licence as {licence!r}, which does not "
            f"contain {EXPECTED_LICENCE!r}. configs/corpora.yaml records CC BY 3.0 for this "
            "corpus; settle the difference before ingesting it."
        )
    return licence, description


def original_zip(corpus: Corpus, item: dict[str, Any]) -> dict[str, Any]:
    """The single zip bitstream of the item's ORIGINAL bundle."""
    bundles = api_get(item["_links"]["bundles"]["href"])["_embedded"]["bundles"]
    original = next((b for b in bundles if b.get("name") == "ORIGINAL"), None)
    if original is None:
        raise LookupError(
            f"{corpus.lang}: item {corpus.handle} has no ORIGINAL bundle; "
            f"bundles present: {[b.get('name') for b in bundles]}"
        )
    bitstreams = api_get(original["_links"]["bitstreams"]["href"])["_embedded"]["bitstreams"]
    zips = [b for b in bitstreams if str(b.get("name", "")).lower().endswith(".zip")]
    if len(zips) != 1:
        raise LookupError(
            f"{corpus.lang}: expected exactly one zip in the ORIGINAL bundle of "
            f"{corpus.handle}, found {[b.get('name') for b in bitstreams]}"
        )
    return zips[0]


def md5_file(path: Path) -> str:
    # MD5 because that is what SADiLaR publishes. It is being compared against a
    # value from the same server over the same connection, so it detects a
    # truncated or corrupted transfer and claims nothing more than that.
    digest = hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def check_published_md5(archive: Path, bitstream: dict[str, Any]) -> str | None:
    """Compare against the repository's own MD5, or say there was none.

    Every other corpus here publishes no checksum, so the digest recorded beside
    the data is only good for comparing two fetches to each other. NCHLT is the
    exception and is worth actually checking.
    """
    checksum = bitstream.get("checkSum") or {}
    if str(checksum.get("checkSumAlgorithm", "")).upper() != "MD5":
        return None
    published = str(checksum.get("value", "")).lower()
    if not published:
        return None
    got = md5_file(archive)
    if got != published:
        archive.unlink(missing_ok=True)
        raise OSError(
            f"{archive.name}: MD5 {got} does not match the {published} SADiLaR publishes; "
            "deleted, re-run to fetch it again"
        )
    return published


# --- the transcripts ----------------------------------------------------------


def corpus_root(dest: Path, iso: str) -> Path:
    """Where the archive actually unpacked to.

    **The ten archives are not packed the same way.** Nine wrap everything in an
    ``nchlt_<iso>/`` directory, which is what the shared README documents.
    isiZulu's does not: its members are ``audio/``, ``transcriptions/``,
    ``LICENSE.txt`` and ``README.txt`` at the root. Verified by reading the first
    local file header of all ten over HTTP Range. Assuming either shape would
    leave nine languages, or one, with a manifest whose every path is wrong.
    """
    for candidate in (dest / f"nchlt_{iso}", dest):
        if (candidate / "audio").is_dir() and (candidate / "transcriptions").is_dir():
            return candidate
    raise FileNotFoundError(
        f"neither {dest / f'nchlt_{iso}'} nor {dest} holds both audio/ and transcriptions/ "
        "after extraction; the archive layout has changed"
    )


def strip_corpus_prefix(audio: str, iso: str) -> str:
    """``nchlt_zul/audio/001/x.wav`` -> ``audio/001/x.wav``.

    The transcripts name every file under an ``nchlt_<iso>/`` component whether or
    not the archive has one, so the attribute is normalised here and the prefix
    the archive really used is put back by the caller.
    """
    parts = PurePosixPath(audio).parts
    if parts and parts[0] == f"nchlt_{iso}":
        parts = parts[1:]
    if not parts or parts[0] != "audio":
        raise ValueError(
            f"transcript names audio at {audio!r}, which is not under 'audio/' once the "
            f"'nchlt_{iso}/' component is removed; the archive layout has changed"
        )
    return str(PurePosixPath(*parts))


def read_corpus_xml(path: Path, iso: str, prefix: str = "") -> list[Utterance]:
    """Utterances of one ``nchlt_<iso>.{trn,tst}.xml``.

    Speaker and transcript both come from the file: ``<speaker id>`` is the
    directory number the audio sits in, and ``<orth>`` is the orthographic
    prompt. Whitespace in the prompt is collapsed because the manifest is TSV and
    a stray tab there would shift every column after it.

    Args:
        prefix: What the archive unpacked under, ``""`` or ``"nchlt_<iso>/"``, so
            the manifest's paths are relative to the language directory.
    """
    root = ElementTree.parse(path).getroot()
    utterances: list[Utterance] = []
    for speaker in root.iter("speaker"):
        speaker_id = (speaker.get("id") or "").strip()
        if not speaker_id:
            raise ValueError(f"{path}: a <speaker> element carries no id")
        for recording in speaker.iter("recording"):
            audio = (recording.get("audio") or "").strip()
            orth = recording.find("orth")
            text = " ".join((orth.text or "").split()) if orth is not None else ""
            if not audio or not text:
                continue
            relative = strip_corpus_prefix(audio, iso)
            utterances.append(
                Utterance(
                    utt_id=PurePosixPath(relative).stem,
                    path=f"{prefix}{relative}",
                    text=text,
                    speaker=speaker_id,
                )
            )
    return utterances


def partition(
    train_side: list[Utterance], test_side: list[Utterance]
) -> list[tuple[Utterance, str]]:
    """Published test suite as ``test``, a derived speaker-disjoint dev, rest train.

    ``derive_splits`` returns a three-way partition; the published test suite
    already fills one of them, so its ``validation`` slice becomes dev and its
    other two slices are folded back into train. That lands dev near a tenth of
    the training side, which is the shape the corpus's own documentation reports
    even though the release does not ship the division.
    """
    if not train_side:
        raise ValueError("the .trn transcript yielded no utterances, so there is nothing to split")
    shared = {u.speaker for u in train_side} & {u.speaker for u in test_side}
    if shared:
        raise ValueError(
            f"speakers {sorted(shared)} appear in both the .trn and .tst transcripts. "
            "The published suites are supposed to be speaker-disjoint, and a dev split "
            "derived from a training side that overlaps test would leak into the score."
        )

    parts, policy, _ = derive_splits(train_side, lambda u: u.speaker or None, lambda u: u.utt_id)
    if policy != "speaker":
        raise ValueError(
            f"the dev split fell back to a {policy}-level derivation, so it would share "
            "speakers with train. NCHLT ships a speaker id on every utterance; a corpus "
            "that no longer does needs a decision rather than a silent leak."
        )
    dev = {u.utt_id for u in parts["validation"]}
    rows = [(u, "validation" if u.utt_id in dev else "train") for u in train_side]
    rows += [(u, "test") for u in test_side]
    return sorted(rows, key=lambda row: row[0].utt_id)


def write_manifest(dest: Path, rows: list[tuple[Utterance, str]]) -> Path:
    path = dest / "manifest.tsv"
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(MANIFEST_COLUMNS)
        for utterance, split in rows:
            writer.writerow(
                [utterance.utt_id, utterance.path, utterance.text, utterance.speaker, split]
            )
    return path


# --- the preparer -------------------------------------------------------------


def prepare(corpus: Corpus, root: Path, *, keep_archives: bool = False) -> Path:
    """Fetch, extract and write the manifest for one language."""
    dest = root / corpus.corpus_id / corpus.lang
    staging = dest / ".archives"
    dest.mkdir(parents=True, exist_ok=True)

    print(f"[svb] {corpus.lang}: {corpus.corpus_id} (SADiLaR {corpus.handle})")
    item = resolve_item(corpus.handle)
    licence, description = check_licence(corpus, item)
    print(f"    licence: {licence}")
    print(f"    the repository says: {description}")

    bitstream = original_zip(corpus, item)
    url = str(bitstream["_links"]["content"]["href"])
    size = bitstream.get("sizeBytes")
    archive = download(
        url, staging / str(bitstream["name"]), expected=size if isinstance(size, int) else None
    )
    problem = verify_archive(archive)
    if problem:
        archive.unlink(missing_ok=True)
        raise OSError(f"{archive.name}: {problem}; deleted, re-run to fetch it again")
    published_md5 = check_published_md5(archive, bitstream)
    print("    md5 matches the published value" if published_md5 else "    no md5 published")

    record = archive_record(archive, url)
    extract_archive(archive, dest)
    if not keep_archives:
        archive.unlink(missing_ok=True)

    root = corpus_root(dest, corpus.iso)
    prefix = "" if root == dest else f"{root.name}/"
    transcriptions = root / "transcriptions"
    train_side = read_corpus_xml(transcriptions / f"nchlt_{corpus.iso}.trn.xml", corpus.iso, prefix)
    test_side = read_corpus_xml(transcriptions / f"nchlt_{corpus.iso}.tst.xml", corpus.iso, prefix)

    # Drop rows whose audio is not on disk before deriving anything, so the
    # partition is over what the manifest will actually hold. A path that does
    # not resolve would otherwise fail the loader mid-epoch instead of here.
    def present(side: list[Utterance]) -> list[Utterance]:
        return [u for u in side if (dest / u.path).exists()]

    kept_train, kept_test = present(train_side), present(test_side)
    missing = (len(train_side) - len(kept_train)) + (len(test_side) - len(kept_test))
    if not kept_test:
        raise ValueError(
            f"{corpus.lang}: none of the published test suite's audio is on disk, so there "
            "would be no test split at all"
        )

    rows = partition(kept_train, kept_test)
    manifest = write_manifest(dest, rows)
    counts = {name: sum(1 for _, split in rows if split == name) for name in SPLITS}
    print(f"    {len(rows)} utterances written {counts}, {missing} rows without audio")

    write_fetch_manifest(
        dest,
        {
            "corpus": corpus.corpus_id,
            "language": corpus.lang,
            "iso_639_3": corpus.iso,
            "handle": corpus.handle,
            "item_uuid": item.get("uuid"),
            "source_page": f"https://repo.sadilar.org/handle/{corpus.handle}",
            "licence": licence,
            "description": description,
            "published_md5": published_md5,
            "archive_root": prefix or ".",
            "split_policy": "published test suite, derived speaker-disjoint dev",
            "counts": counts,
            "speakers": {
                name: len({u.speaker for u, split in rows if split == name}) for name in SPLITS
            },
            "rows_without_audio": missing,
            "archives": [record],
        },
    )
    if not keep_archives and staging.exists() and not any(staging.iterdir()):
        staging.rmdir()
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="$CORPORA_ROOT")
    parser.add_argument(
        "--langs",
        nargs="*",
        choices=LANGUAGES,
        help=f"default: the preset five {' '.join(PRESET_LANGUAGES)}",
    )
    parser.add_argument(
        "--keep-archives", action="store_true", help="keep the zips so a re-extract is free"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for lang in args.langs or PRESET_LANGUAGES:
        prepare(BY_LANG[lang], args.root, keep_archives=args.keep_archives)
    return 0


if __name__ == "__main__":
    sys.exit(main())
