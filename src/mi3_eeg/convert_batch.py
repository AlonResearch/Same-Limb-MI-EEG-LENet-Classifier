"""Batch-convert raw MI3 .mat files in a folder.

Behavior:
    - Scans a folder for .mat files (non-recursive)
    - Converts raw format (task_data/task_label/rest_data)
    - Saves standardized output to Datasets/MI3/derivatives

Terminal usage example:
    python -m mi3_eeg.convert_batch "G:/My Drive/ML/dataset"
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Iterable

import scipy.io as scio

from mi3_eeg.config import Paths
from mi3_eeg.dataformatter import detect_format, format_and_save
from mi3_eeg.logger import setup_logger, logger


def _extract_subject_id(file_path: Path) -> str:
    """Extract subject_id from filename like sub-017_task-motorimagery_eeg.mat.

    Args:
        file_path: Path to a .mat file.

    Returns:
        Subject ID string (e.g., "sub-017") or "sub-unknown".
    """
    filename = file_path.stem
    if "sub-" in filename:
        parts = filename.split("sub-")
        if len(parts) > 1:
            subject_num = parts[1].split("_")[0].split("-")[0]
            return f"sub-{subject_num}"
    return "sub-unknown"


def _iter_mat_files(folder: Path) -> Iterable[Path]:
    """Yield .mat files from a folder (non-recursive)."""
    return sorted(folder.glob("*.mat"))


def _is_raw_mat_file(file_path: Path) -> bool:
    """Return True if a .mat file has raw MI3 keys.

    Args:
        file_path: Path to a .mat file.

    Returns:
        True if keys include task_data/task_label/rest_data; else False.
    """
    try:
        mat_data = scio.loadmat(str(file_path))
        return detect_format(mat_data) == "raw"
    except Exception as exc:
        logger.warning(f"Could not inspect {file_path.name}: {exc}")
        return False


def main() -> None:
    """CLI entrypoint for batch conversion.

    Args (CLI):
        argv[1]: input folder path containing .mat files
    """
    setup_logger()

    if len(sys.argv) < 2:
        logger.error("Missing folder argument.")
        logger.info("Usage: python -m mi3_eeg.convert_batch <folder_path>")
        sys.exit(1)

    input_folder = Path(sys.argv[1])
    if not input_folder.exists() or not input_folder.is_dir():
        logger.error(f"Folder not found: {input_folder}")
        sys.exit(1)

    paths = Paths.from_here()
    output_dir = paths.dataset_derivatives
    output_dir.mkdir(parents=True, exist_ok=True)

    mat_files = list(_iter_mat_files(input_folder))
    if not mat_files:
        logger.warning(f"No .mat files found in: {input_folder}")
        return

    logger.info("=" * 80)
    logger.info("MI3 Batch Conversion Script")
    logger.info("=" * 80)
    logger.info(f"Input folder: {input_folder}")
    logger.info(f"Output folder: {output_dir}")
    logger.info(f"Found {len(mat_files)} .mat files")

    converted = 0
    skipped = 0
    failed = 0

    for mat_file in mat_files:
        subject_id = _extract_subject_id(mat_file)
        mat_filename = mat_file.name

        logger.info("-" * 80)
        logger.info(f"Processing file: {mat_filename}")
        logger.info(f"Subject ID: {subject_id}")

        if not _is_raw_mat_file(mat_file):
            logger.info("Skipping (not raw format).")
            skipped += 1
            continue

        try:
            output_path = format_and_save(
                input_path=mat_file,
                output_dir=output_dir,
                subject_id=subject_id,
            )
            logger.info(f"Formatted file saved: {output_path}")
            converted += 1
        except Exception as exc:
            logger.error(f"Failed to convert {mat_filename}: {exc}")
            failed += 1

    logger.info("=" * 80)
    logger.info("Batch conversion summary")
    logger.info(f"Converted: {converted}")
    logger.info(f"Skipped:   {skipped}")
    logger.info(f"Failed:    {failed}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
