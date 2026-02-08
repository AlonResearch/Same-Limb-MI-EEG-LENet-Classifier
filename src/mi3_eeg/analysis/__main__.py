"""Entry point for the analysis package.

This module allows the analysis package to be executed as:
    python -m mi3_eeg.analysis <arguments>

without causing module duplication warnings in sys.modules.
"""

from mi3_eeg.analysis.group_analysis import main

if __name__ == "__main__":
    main()
