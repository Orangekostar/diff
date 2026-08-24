from cmc_bbdm.mvd.observability_dataset import StudentInputs


def test_mvd_student_never_reads_true_cai() -> None:
    assert "target" not in StudentInputs.__dataclass_fields__
    assert "true_cai" not in StudentInputs.__dataclass_fields__
