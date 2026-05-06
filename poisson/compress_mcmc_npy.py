#!/home/verano13/miniconda3/envs/stochastic-interpolants/bin/python3
"""
Compress all .npy files inside mcmc_* subdirectories of poisson/ to .npz format.
Verifies each file before deleting the original.
Safe to re-run: skips files where .npz already exists.
"""
import numpy as np
import os
import glob
from tqdm import tqdm

POISSON_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(POISSON_DIR, "compress_mcmc_npy.log")


def log(msg):
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")


def compress_file(npy_path):
    npz_path = npy_path.replace(".npy", ".npz")

    if os.path.exists(npz_path):
        return "skipped"

    try:
        arr = np.load(npy_path)
        np.savez_compressed(npz_path, arr=arr)

        # verify
        loaded = np.load(npz_path)["arr"]
        if loaded.shape != arr.shape or not np.array_equal(loaded, arr):
            os.remove(npz_path)
            return "verify_failed"

        original_size = os.path.getsize(npy_path)
        compressed_size = os.path.getsize(npz_path)
        os.remove(npy_path)
        return "ok", original_size, compressed_size

    except Exception as e:
        if os.path.exists(npz_path):
            os.remove(npz_path)
        return "error", str(e)


def main():
    npy_files = sorted(glob.glob(
        os.path.join(POISSON_DIR, "mcmc_*", "**", "*.npy"), recursive=True
    ))

    total = len(npy_files)
    log(f"Found {total} .npy files to process under {POISSON_DIR}/mcmc_*/\n")

    done = 0
    skipped = 0
    errors = 0
    total_original = 0
    total_compressed = 0

    with tqdm(npy_files, unit="file") as pbar:
        for i, path in enumerate(npy_files, 1):
            rel = os.path.relpath(path, POISSON_DIR)
            pbar.set_description(rel[-60:])
            result = compress_file(path)

            if result == "skipped":
                skipped += 1
                log(f"[{i}/{total}] SKIP  {rel}")
            elif result[0] == "ok":
                _, orig, comp = result
                total_original += orig
                total_compressed += comp
                ratio = comp / orig if orig > 0 else 0
                log(f"[{i}/{total}] OK    {rel}  {orig/1e9:.2f}GB -> {comp/1e9:.2f}GB ({ratio:.2f}x)")
                done += 1
            elif result == "verify_failed":
                errors += 1
                log(f"[{i}/{total}] FAIL  {rel}  (verification mismatch, .npz removed)")
            else:
                errors += 1
                log(f"[{i}/{total}] ERROR {rel}  {result[1]}")

            pbar.set_postfix(done=done, skipped=skipped, errors=errors,
                             saved_GB=f"{(total_original - total_compressed)/1e9:.1f}")
            pbar.update(1)

    savings = total_original - total_compressed
    log(f"\nDone. {done} compressed, {skipped} skipped, {errors} errors.")
    log(f"Space freed: {savings/1e9:.1f} GB  ({total_original/1e9:.1f} GB -> {total_compressed/1e9:.1f} GB)")


if __name__ == "__main__":
    main()
