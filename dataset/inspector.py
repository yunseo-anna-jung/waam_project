"""
dataset/inspector.py

WAAM Dataset Inspector

Recorder가 저장한 Dataset을 검사하는 모듈

Author : Yunseo Anna Jung
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np


class DatasetInspector:

    def __init__(

        self,

        dataset_root: str | Path,

        metadata_path: str | Path,

    ):

        self.dataset_root = Path(dataset_root)

        self.metadata_path = Path(metadata_path)

        self.metadata = pd.read_csv(

            self.metadata_path

        )

    # ======================================================

    def count_labels(self):

        counts = (

            self.metadata["label"]

            .value_counts()

            .to_dict()

        )

        return {

            "normal": counts.get("normal", 0),

            "warning": counts.get("warning", 0),

            "danger": counts.get("danger", 0),

        }

    # ======================================================

    def check_missing_files(self):

        missing = []

        for _, row in self.metadata.iterrows():

            file = (

                self.dataset_root

                / row["label"]

                / row["filename"]

            )

            if not file.exists():

                missing.append(file)

        return missing

    # ======================================================

    def check_corrupted_files(self):

        corrupted = []

        for _, row in self.metadata.iterrows():

            file = (

                self.dataset_root

                / row["label"]

                / row["filename"]

            )

            try:

                np.load(file)

            except Exception:

                corrupted.append(file)

        return corrupted

    # ======================================================

    def summary(self):

        counts = self.count_labels()

        missing = self.check_missing_files()

        corrupted = self.check_corrupted_files()

        total = sum(counts.values())

        print()

        print("=" * 50)

        print(" WAAM Dataset Inspector ")

        print("=" * 50)

        print(f"Normal   : {counts['normal']}")

        print(f"Warning  : {counts['warning']}")

        print(f"Danger   : {counts['danger']}")

        print("-" * 50)

        print(f"Total Samples    : {total}")

        print(f"Missing Files    : {len(missing)}")

        print(f"Corrupted Files  : {len(corrupted)}")

        print("-" * 50)

        if len(missing) == 0 and len(corrupted) == 0:

            print("Dataset Status : PASS")

        else:

            print("Dataset Status : FAIL")

        print("=" * 50)

        return {

            "counts": counts,

            "missing": missing,

            "corrupted": corrupted,

        }