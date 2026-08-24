from cmc_bbdm.mvd.observability_dataset import StudentInputs


def test_mvd_student_never_reads_candidate_embedding() -> None:
    assert "candidate_embeddings" not in StudentInputs.__dataclass_fields__
    assert "candidate_embedding" not in StudentInputs.__dataclass_fields__
