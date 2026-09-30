import pytest

from crag.logic import Verdict, classify_retrieval, decompose, select_knowledge, split_sentences

UP, LO = 0.7, 0.3


class TestClassifyRetrieval:
    def test_correct_when_any_score_above_upper(self):
        assert classify_retrieval([0.1, 0.8, 0.2], UP, LO) is Verdict.CORRECT

    def test_incorrect_when_all_scores_below_lower(self):
        assert classify_retrieval([0.0, 0.1, 0.29], UP, LO) is Verdict.INCORRECT

    def test_ambiguous_in_between(self):
        assert classify_retrieval([0.1, 0.5], UP, LO) is Verdict.AMBIGUOUS

    def test_boundaries_are_strict(self):
        # Exactly on a threshold is neither "above upper" nor "below lower".
        assert classify_retrieval([0.7], UP, LO) is Verdict.AMBIGUOUS
        assert classify_retrieval([0.3], UP, LO) is Verdict.AMBIGUOUS

    def test_empty_retrieval_is_incorrect(self):
        assert classify_retrieval([], UP, LO) is Verdict.INCORRECT

    @pytest.mark.parametrize("upper,lower", [(0.3, 0.7), (0.5, 0.5), (1.5, 0.2), (0.7, -0.1)])
    def test_invalid_thresholds_raise(self, upper, lower):
        with pytest.raises(ValueError):
            classify_retrieval([0.5], upper, lower)


class TestSelectKnowledge:
    def test_correct_uses_internal_only(self):
        assert select_knowledge(Verdict.CORRECT, ["i"], ["w"]) == ["i"]

    def test_incorrect_uses_web_only(self):
        assert select_knowledge(Verdict.INCORRECT, ["i"], ["w"]) == ["w"]

    def test_ambiguous_uses_both_internal_first(self):
        assert select_knowledge(Verdict.AMBIGUOUS, ["i"], ["w"]) == ["i", "w"]


class TestDecompose:
    TEXT = (
        "Batch normalization normalizes across the batch dimension. "
        "Layer normalization normalizes across the feature dimension! "
        "Short one. "
        "Is layer norm independent of batch size?"
    )

    def test_split_drops_short_fragments(self):
        sents = split_sentences(self.TEXT)
        assert len(sents) == 3
        assert "Short one." not in sents

    def test_whitespace_is_normalised(self):
        assert split_sentences("A sentence   that\n\nis long enough.") == [
            "A sentence that is long enough."
        ]

    def test_empty_text(self):
        assert split_sentences("   ") == []
        assert decompose("") == []

    def test_one_sentence_per_strip_matches_split(self):
        assert decompose(self.TEXT, 1) == split_sentences(self.TEXT)

    def test_groups_sentences_into_strips(self):
        strips = decompose(self.TEXT, 2)
        assert len(strips) == 2
        assert strips[0].startswith("Batch normalization") and "Layer normalization" in strips[0]

    def test_invalid_strip_size(self):
        with pytest.raises(ValueError):
            decompose(self.TEXT, 0)
