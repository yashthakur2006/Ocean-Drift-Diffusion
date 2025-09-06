import subprocess, sys, os, json

def test_smoke_train_eval():
    # Run a very short training and evaluation on synthetic data
    subprocess.check_call([sys.executable, "-m", "odd.train", "--data", "synthetic", "--epochs", "1", "--batch-size", "4", "--device", "cpu"])
    subprocess.check_call([sys.executable, "-m", "odd.eval", "--data", "synthetic", "--num-samples", "8"])
    assert os.path.exists("outputs/metrics.json")
    with open("outputs/metrics.json") as f:
        m = json.load(f)
    assert "ADE" in m
