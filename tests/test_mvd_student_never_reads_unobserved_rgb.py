from cmc_bbdm.mvd.observability_dataset import StudentInputs


def test_mvd_student_never_reads_unobserved_rgb() -> None:
    assert "rgb" not in StudentInputs.__dataclass_fields__
    assert set(StudentInputs.__dataclass_fields__) == {
        "initial_embedding",
        "current_prediction",
        "candidate_features",
    }
