"""Run training on all subjects in the derivatives folder."""

from pathlib import Path
import subprocess
import sys

def main():
    """Run training on all .mat files in derivatives folder."""
    derivatives_path = Path(__file__).parent / "Datasets" / "MI3" / "derivatives"
    metrics_path = Path(__file__).parent / "reports" / "metrics"
    
    # Find all *_eeg200hz.mat files (skip the original sub-011_eeg.mat for now)
    mat_files = sorted(derivatives_path.glob("*_eeg200hz.mat"))
    
    if not mat_files:
        print("No *_eeg200hz.mat files found in derivatives folder!")
        return
    
    print(f"Found {len(mat_files)} subject files:")
    for f in mat_files:
        print(f"  - {f.name}")
    
    # Filter out subjects that already have results
    completed = []
    pending = []
    for mat_file in mat_files:
        subject_id = mat_file.name.split('_')[0]
        results_file = metrics_path / f"{subject_id}_lenet_results.json"
        if results_file.exists():
            completed.append(mat_file.name)
        else:
            pending.append(mat_file)
    
    if completed:
        print(f"\n✓ Already completed ({len(completed)}):")
        for f in completed[:5]:
            print(f"  - {f}")
        if len(completed) > 5:
            print(f"  ... and {len(completed) - 5} more")
    
    if not pending:
        print("\n✓ All subjects already processed!")
        return
    
    print(f"\n{'='*80}")
    print(f"Starting training runs with 50 epochs each ({len(pending)} remaining)...")
    print(f"{'='*80}\n")
    
    # Run training on each pending file
    for i, mat_file in enumerate(pending, 1):
        print(f"\n{'='*80}")
        print(f"[{i}/{len(pending)}] Processing: {mat_file.name}")
        print(f"{'='*80}\n")
        
        # Run the training
        cmd = [
            sys.executable,
            "-m",
            "mi3_eeg.main",
            "--subject-file",
            mat_file.name,
            "--epochs",
            "50"
        ]
        
        try:
            result = subprocess.run(cmd, check=True)
            print(f"\n✓ Successfully completed: {mat_file.name}")
        except subprocess.CalledProcessError as e:
            print(f"\n✗ Failed on: {mat_file.name}")
            print(f"Error: {e}")
            # Continue with next file
            continue
    
    print(f"\n{'='*80}")
    print(f"All training runs completed!")
    print(f"{'='*80}")
    print(f"Results saved in:")
    print(f"  - models/")
    print(f"  - reports/metrics/")
    print(f"  - reports/figures/")

if __name__ == "__main__":
    main()
