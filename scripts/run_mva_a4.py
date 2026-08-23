import os


def _main() -> int:
    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = "1"

    from cmc_bbdm.mva.a4_cli import main

    return main()

if __name__ == "__main__":
    raise SystemExit(_main())
