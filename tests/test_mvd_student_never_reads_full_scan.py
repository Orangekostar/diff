from cmc_bbdm.mvd.observability_dataset import StudentInputs


def test_mvd_student_never_reads_full_scan() -> None:
    assert "image" not in StudentInputs.__dataclass_fields__
    assert "full_scan" not in StudentInputs.__dataclass_fields__
