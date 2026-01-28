"""Convert a single raw MI3 .mat file into standardized format.

Input (raw .mat keys): task_data, task_label, rest_data
Output (standardized keys): all_data, all_label, sampling_rate, source_file

Terminal usage example:
    python -m mi3_eeg.data_formatting.convert_subject "G:/My Drive/ML/dataset/sub-017_task-motorimagery_eeg.mat" sub-017
"""

from pathlib import Path
import sys

from mi3_eeg.data_formatting.dataformatter import format_and_save
from mi3_eeg.logger import setup_logger, logger


def main():
    """CLI entrypoint.

    Args (CLI):
        argv[1]: input .mat path (raw format)
        argv[2]: optional subject_id (e.g., sub-017)
    """
    # Setup logging
    setup_logger()
    
    # Get input file path from command line or use default
    if len(sys.argv) > 1:
        input_file = Path(sys.argv[1])
    else:
        # Default path for testing
        input_file = Path("G:/My Drive/ML/dataset/sub-017_task-motorimagery_eeg.mat")
    
    # Get subject ID if provided
    subject_id = sys.argv[2] if len(sys.argv) > 2 else None
    
    logger.info("=" * 80)
    logger.info("MI3 Raw Data Formatting Script")
    logger.info("=" * 80)
    logger.info(f"Input file: {input_file}")
    
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        logger.info("Usage: python -m mi3_eeg.data_formatting.convert_subject <input_file> [subject_id]")
        logger.info("Example: python -m mi3_eeg.data_formatting.convert_subject sub-017_task-motorimagery_eeg.mat sub-017")
        sys.exit(1)
    
    try:
        # Format the file
        output_file = format_and_save(
            input_path=input_file,
            subject_id=subject_id
        )
        
        logger.info("=" * 80)
        logger.info("✓ Formatting completed successfully!")
        logger.info(f"  Input:  {input_file}")
        logger.info(f"  Output: {output_file}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Formatting failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
