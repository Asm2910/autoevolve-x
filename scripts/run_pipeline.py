"""Run the whole pipeline end to end: generate -> validate -> features -> split -> train.

Run with:  python -m scripts.run_pipeline
Skip steps already done:  python -m scripts.run_pipeline --from features
"""

import argparse
import subprocess
import sys
import time

STEPS = [
    ("generate", "scripts.generate_data"),
    ("validate", "scripts.validate_data"),
    ("features", "scripts.build_features"),
    ("split", "scripts.split_data"),
    ("train", "scripts.train_models"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="start", choices=[s for s, _ in STEPS], default="generate")
    args = parser.parse_args()

    names = [name for name, _ in STEPS]
    started = time.time()
    for name, module in STEPS[names.index(args.start):]:
        print(f"\n=== [{name}] python -m {module} ===", flush=True)
        step_start = time.time()
        result = subprocess.run([sys.executable, "-m", module], check=False)
        if result.returncode != 0:
            print(f"\nPipeline stopped: step '{name}' failed (exit {result.returncode}).")
            sys.exit(result.returncode)
        print(f"=== [{name}] done in {time.time() - step_start:.0f}s ===", flush=True)

    print(f"\nPipeline finished in {time.time() - started:.0f}s. Models are in models/artifacts/.")


if __name__ == "__main__":
    main()
