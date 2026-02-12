import subprocess
import time
import sys

# Configuration
TRAINING_SCRIPT = "qadv_trades_untarget.py"
EVAL_SCRIPT = "eval_AA.py"

# Get run name from command line
if len(sys.argv) > 1:
    run_name = sys.argv[1]
else:
    run_name = "default"
    print(f"No run name provided, using '{run_name}'")

# Extract script name without extension
script_name = TRAINING_SCRIPT.replace('.py', '')

print("="*60)
print("TRAIN AND EVALUATE PIPELINE")
print("="*60)
print(f"Run name: {run_name}")
print(f"Training script: {TRAINING_SCRIPT}")
print(f"Evaluation script: {EVAL_SCRIPT}")
print("="*60)

# STEP 1: Run Training
print(f"\n{'#'*60}")
print("STEP 1: TRAINING")
print(f"{'#'*60}\n")
print(f"Running: python {TRAINING_SCRIPT} {run_name}")
print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

train_start = time.time()

try:
    subprocess.run(["python", TRAINING_SCRIPT, run_name], check=True)
    train_duration = (time.time() - train_start) / 60
    print(f"\n{'='*60}")
    print(f"✓ Training completed successfully in {train_duration:.2f} minutes")
    print(f"{'='*60}\n")
except subprocess.CalledProcessError as e:
    print(f"\n{'='*60}")
    print(f"✗ Training failed with exit code {e.returncode}")
    print(f"{'='*60}\n")
    sys.exit(1)

# STEP 2: Run AutoAttack Evaluation
print(f"\n{'#'*60}")
print("STEP 2: AUTOATTACK EVALUATION")
print(f"{'#'*60}\n")
print(f"Running: python {EVAL_SCRIPT} {run_name} {script_name}")
print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

eval_start = time.time()

try:
    subprocess.run(["python", EVAL_SCRIPT, run_name, script_name], check=True)
    eval_duration = (time.time() - eval_start) / 60
    print(f"\n{'='*60}")
    print(f"✓ AutoAttack evaluation completed in {eval_duration:.2f} minutes")
    print(f"{'='*60}\n")
except subprocess.CalledProcessError as e:
    print(f"\n{'='*60}")
    print(f"✗ AutoAttack evaluation failed with exit code {e.returncode}")
    print(f"{'='*60}\n")
    sys.exit(1)

# Final Summary
total_duration = (time.time() - train_start) / 60
print("\n" + "="*60)
print("PIPELINE COMPLETED SUCCESSFULLY")
print("="*60)
print(f"Training time: {train_duration:.2f} minutes")
print(f"Evaluation time: {eval_duration:.2f} minutes")
print(f"Total time: {total_duration:.2f} minutes")
print("="*60)
print(f"\nCheckpoints saved in: ./Checkpoints/{run_name}/{script_name}/")
print(f"Training results in: ./CSV/{run_name}/{script_name}/")
print(f"AutoAttack results in: ./AutoAttack_Results/{run_name}/{script_name}/")
print("="*60)