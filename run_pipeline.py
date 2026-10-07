import subprocess
import re
import sys
import matplotlib.pyplot as plt

def run_pipeline():
    cmd = [
        sys.executable, "train_v2.py",
        "--data-dir", "data",
        "--out-dir", "checkpoints_v3",
        "--batch-size", "64",
        "--grad-accum", "2",
        "--max-steps", "2000",
        "--eval-every", "100",
        "--log-every", "20",
        "--gen-prompt", "Once upon a time",
        "--gen-tokens", "50"
    ]

    print("🚀 Launching DomLM Training Pipeline...\n")
    
    train_steps, train_losses = [], []
    val_steps, val_losses = [], []

    with open("train.log", "w") as log_file:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_file.write(line)

            t_match = re.search(r"step (\d+)/\d+ loss ([\d.]+)", line)
            if t_match:
                train_steps.append(int(t_match.group(1)))
                train_losses.append(float(t_match.group(2)))

            v_match = re.search(r"\[eval\] step (\d+) val_loss ([\d.]+)", line)
            if v_match:
                val_steps.append(int(v_match.group(1)))
                val_losses.append(float(v_match.group(2)))

        process.wait()

    if process.returncode != 0:
        print(f"\n❌ Training crashed with exit code {process.returncode}")
        return

    print("\n✅ Training complete! Building loss curve graph...")

    fig, ax = plt.subplots(figsize=(11, 5), dpi=120)

    ax.plot(train_steps, train_losses, color="#3498db", alpha=0.4, label="Train Loss (Batches)", linewidth=1.5)
    ax.plot(val_steps, val_losses, color="#e74c3c", marker="o", linewidth=2.5, label="Validation Loss")

    if val_losses:
        min_val = min(val_losses)
        best_step = val_steps[val_losses.index(min_val)]
        ax.scatter([best_step], [min_val], color="#2ecc71", s=140, zorder=5, 
                   label=f"Best Checkpoint (Step {best_step}: {min_val:.4f})")

    ax.set_xlabel("Training Steps (X-Axis)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Loss Value (Y-Axis)", fontsize=11, fontweight="bold")
    ax.set_title("DomLM v3.0 Training & Validation Loss Dynamics", fontsize=13, pad=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9)

    plt.tight_layout()
    plt.savefig("domlm_learning_curve.png")
    plt.show()
    print("📊 Graph saved as 'domlm_learning_curve.png'")

if __name__ == "__main__":
    run_pipeline()
