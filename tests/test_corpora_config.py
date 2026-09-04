"""`configs/corpora.yaml` is the licence and provenance record for every corpus
that is not Common Voice. It is read by the preparers and by the language
selection, so its shape is checked rather than trusted.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from svb.data.corpora import (
    HELD_OUT_LANGUAGES,
    SPEAKER_ID_STATES,
    SPLIT_STATES,
    CorpusSpec,
    load_corpora,
)


@pytest.fixture
def corpora(pytestconfig: pytest.Config) -> list[CorpusSpec]:
    return load_corpora(Path(pytestconfig.rootpath) / "configs")


def test_the_file_lists_corpora(corpora: list[CorpusSpec]) -> None:
    assert corpora
    assert all(isinstance(c, CorpusSpec) for c in corpora)


def test_no_corpus_id_is_used_twice(corpora: list[CorpusSpec]) -> None:
    ids = [c.id for c in corpora]

    assert len(ids) == len(set(ids)), sorted({i for i in ids if ids.count(i) > 1})


def test_no_language_is_served_by_two_corpora(corpora: list[CorpusSpec]) -> None:
    """One language, one corpus. Two would mean the preset has to say which, and
    it says only the language."""
    codes = [code for c in corpora for code in c.languages]

    assert len(codes) == len(set(codes)), sorted({c for c in codes if codes.count(c) > 1})


def test_no_held_out_transfer_language_is_offered_as_training_data(
    corpora: list[CorpusSpec],
) -> None:
    """The one rule in this file that is not about bookkeeping.

    Malayalam, Marathi, Telugu and Gujarati are the transfer set, and Odia is
    the pending fifth. A corpus that served any of them as training data would
    make the held-out number meaningless, and several of the corpora here come
    from the same crowdsourced family as the held-out four, so the guard is not
    hypothetical.
    """
    offered = {code for c in corpora for code in c.languages}

    assert not offered & HELD_OUT_LANGUAGES, sorted(offered & HELD_OUT_LANGUAGES)


def test_every_corpus_names_its_licence_verbatim_and_where_it_says_so(
    corpora: list[CorpusSpec],
) -> None:
    """A licence name with no URL is a claim; with one it is a citation."""
    for corpus in corpora:
        assert corpus.licence_name, corpus.id
        assert corpus.licence_url.startswith("https://"), corpus.id


def test_every_corpus_cites_a_source_page_and_something_to_download(
    corpora: list[CorpusSpec],
) -> None:
    for corpus in corpora:
        assert corpus.source_page.startswith("https://"), corpus.id
        assert corpus.downloads, corpus.id
        for item in corpus.downloads:
            assert item.get("name"), corpus.id
            assert item.get("url") or item.get("url_pattern") or item.get("api"), corpus.id


def test_split_and_speaker_states_come_from_a_fixed_vocabulary(
    corpora: list[CorpusSpec],
) -> None:
    """Free text here would be unreadable by the preparer that has to act on it."""
    for corpus in corpora:
        assert corpus.ships_split in SPLIT_STATES, (corpus.id, corpus.ships_split)
        assert corpus.speaker_ids in SPEAKER_ID_STATES, (corpus.id, corpus.speaker_ids)


def test_a_corpus_that_ships_no_split_can_have_one_derived(corpora: list[CorpusSpec]) -> None:
    """Deriving a speaker-disjoint split needs speaker ids. A corpus with
    neither a shipped split nor recoverable speakers cannot be trained on
    honestly, and this is where that shows up rather than mid-run."""
    for corpus in corpora:
        if corpus.ships_split == "none":
            assert corpus.speaker_ids != "absent", corpus.id


def test_every_corpus_points_at_a_preparer_that_exists(
    corpora: list[CorpusSpec], pytestconfig: pytest.Config
) -> None:
    root = Path(pytestconfig.rootpath)

    for corpus in corpora:
        assert (root / corpus.preparer).exists(), f"{corpus.id} -> {corpus.preparer}"


def test_hours_are_recorded_even_when_only_a_total_is_published(
    corpora: list[CorpusSpec],
) -> None:
    """Several source pages publish one total and no per-split figure. That is
    worth recording as a total rather than left blank and re-derived by hand."""
    for corpus in corpora:
        assert corpus.total_hours is not None or corpus.train_hours is not None, corpus.id


def test_an_unknown_key_is_rejected_rather_than_ignored(tmp_path: Path) -> None:
    """A typo in a key silently drops the value it was meant to set, and the
    values here are licences."""
    (tmp_path / "corpora.yaml").write_text(
        "corpora:\n  - id: x\n    langauges: [xx]\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="langauges"):
        load_corpora(tmp_path)


def test_a_missing_required_key_names_the_corpus(tmp_path: Path) -> None:
    (tmp_path / "corpora.yaml").write_text("corpora:\n  - id: x\n", encoding="utf-8")

    with pytest.raises(ValueError, match="x"):
        load_corpora(tmp_path)
