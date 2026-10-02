# Running on CPU, Apple MPS and NVIDIA CUDA

The experiment can run on three PyTorch backends. Model, optimizer, LR schedule, sequence
lengths, batch size, step counts, seeds and dtypes (including the complex64 S4D kernel) are
identical on all of them. Only the device and the output location change. Mixed precision is not
used on any backend, because the S4D layer computes with complex tensors.

## Device selection

`src/train.py` and `experiments/run_context_length_sweep.py` accept
`--device {auto,cpu,mps,cuda}`.

- `auto` (the default) picks CUDA if available, then MPS, then CPU.
- An explicit device that is not available stops the run with an error instead of falling back.
  On a Mac, `--device cuda` exits with status 1 and reports that `torch.cuda.is_available()`
  is False.

The requested and resolved device, PyTorch, CUDA and cuDNN versions, GPU name and platform are
written to `device_info` in every result JSON.

## Commands

Mac (MPS):

```bash
python experiments/run_context_length_sweep.py --device mps           # -> results/mps/
```

NVIDIA GPU (CUDA):

```bash
nvidia-smi
python experiments/run_context_length_sweep.py --device cuda          # -> results/cuda/
```

CPU:

```bash
python experiments/run_context_length_sweep.py --device cpu           # -> results/cpu/
```

Useful options:

- `--out_dir DIR` writes somewhere other than `results/<device>/`.
- `--dry_run` prints the ten `train.py` commands without running them.
- `--overwrite` replaces files from an earlier run in the same output directory.

To plot a new dataset without touching the tracked figures:

```bash
python experiments/make_plots.py --results_dir results/cuda   # figures + summary_table.md in results/cuda/
```

A single run:

```bash
python src/train.py --mixer s4d --device cuda --block_size 256 --max_iters 60 \
    --eval_interval 30 --eval_iters 5 --out results/cuda/smoke/s4d_bs256.json
```

## Colab

Open `colab/run_cuda.ipynb` with a GPU runtime and set `REF` to the commit SHA to measure. The
notebook then:

1. runs `nvidia-smi`;
2. clones the repository and checks out `REF`;
3. records the PyTorch build before and after installing the requirements, and confirms that CUDA
   is still available;
4. runs the test suite;
5. runs a 60-step smoke test of each mixer at block_size 256 into `results/cuda/smoke/`, then
   checks finite losses, the resolved CUDA device, the timing fields and the CUDA memory fields.

The notebook then stops at a gate (`RUN_FULL_SWEEP = False`). The gate cell and the sweep cell
re-read both smoke results and re-run every check, so the sweep cannot start unless both smoke
runs passed. Once the gate is opened, the notebook:

- runs the full sweep into `results/cuda/`;
- confirms that the historical MPS files are unchanged;
- prints a CUDA-only table;
- generates plots and a summary table in `results/cuda/`;
- zips `results/cuda/` with an `environment.json` for download.

## Kaggle

The same notebook works on Kaggle with a GPU accelerator and internet access enabled. Set
`WORKDIR = "/kaggle/working/s4d-transformer-comparison"`. The archive is written next to
`WORKDIR`, so download it from `/kaggle/working`.

## Output locations

| path | contents |
|---|---|
| `results/*.json`, `results/summary_table.md`, `figures/` | historical Apple MPS dataset quoted in the README |
| `results/mps/`, `results/cuda/`, `results/cpu/` | new sweeps, one directory per device |
| `results/cuda/smoke/` | short Colab smoke runs (pipeline checks, not results; ignored by the sweep and plots) |

Three guards protect the historical files:

- the sweep refuses `--out_dir results`;
- `train.py` refuses to replace any existing file directly in `results/`, even with
  `--overwrite`;
- both scripts refuse to replace any other existing result unless `--overwrite` is given.

`make_plots.py` without arguments reads only the top-level `results/*_bs*.json`, so per-device
subdirectories never mix into the historical figures.

## Timing methodology

Each result JSON contains these timing fields:

| field | definition |
|---|---|
| `tokens_per_second` | training tokens / `total_seconds`. The interval spans the whole loop, so it **includes** periodic evaluation, CPU-side batch assembly and host-to-device copies. This is the metric in the historical files and the README. |
| `mean_step_seconds` | mean time per optimizer step. Each step is timed from after batch assembly until after `optimizer.step()`, followed by `torch.cuda.synchronize()` or `torch.mps.synchronize()`. Includes the host-to-device copy of the batch. |
| `median_step_seconds` | median of the same per-step times, less sensitive to warm-up steps. |
| `train_step_tokens_per_second` | training tokens / sum of per-step times (step-only throughput). |
| `eval_seconds` | time spent in evaluation. Each evaluation ends with `loss.item()`, which synchronizes the device. |

`tokens_per_second` and `train_step_tokens_per_second` measure different things and should not
be compared with each other. The last three fields are new. The historical MPS files contain only
the first two.

Single runs vary noticeably on the same machine. Re-running the block_size 64 configurations on
the original Mac reproduced the historical final validation loss to about 4e-7. The measured
`tokens_per_second` was about 25% lower than the recorded value, however, so a throughput
difference between devices of that size is within run-to-run variation.

## Memory methodology

| field | backend | meaning |
|---|---|---|
| `mps_current_allocated_bytes_sampled_max` | MPS | `torch.mps.current_allocated_memory()` sampled only at evaluation checkpoints, so a lower bound on the true peak |
| `cuda_max_memory_allocated_bytes` | CUDA | `torch.cuda.max_memory_allocated()`: exact peak of live tensor memory since `torch.cuda.reset_peak_memory_stats()`, called immediately before step 0 (model parameters count; optimizer state is created inside the region) |
| `cuda_max_memory_reserved_bytes` | CUDA | `torch.cuda.max_memory_reserved()`: peak memory held by the CUDA caching allocator |
| `peak_process_rss_bytes` | all | whole-process peak resident memory from `getrusage` |
| `device_memory_api` | all | which API produced the device-memory figure (`null` on CPU) |

No device-memory figure is reported for CPU runs. The MPS and CUDA numbers come from different
APIs with different semantics (sampled lower bound vs. exact allocator peak), so they are not
directly comparable. Process RSS is not comparable across backends either, because GPU buffers
can count towards RSS on Apple unified memory but not on CUDA.

## Why the existing headline stays MPS-specific

The README's throughput figures (for example ~172.9k vs. ~38.2k tokens/s at block_size 1024)
were measured on one Apple M-series machine with the MPS backend. They come from
`results/s4d_bs1024.json`, `results/attn_bs1024.json` and the other top-level result files.
They describe that implementation on that device and remain labelled as such.

## Why CUDA runs are a new dataset

A CUDA sweep repeats the same experiment on different hardware with a different kernel stack
(cuBLAS, cuFFT, CUDA caching allocator vs. Metal). Model initialisation and the sequence of
training batches are identical between MPS and CUDA runs, because both come from the CPU
generator seeded with 42 and dropout draws from the GPU's own generator. (On CPU, dropout also
draws from the CPU generator, so the batch sequence differs.) Dropout masks and floating-point
reduction order differ between devices, so losses agree only approximately. CUDA results are stored separately under `results/cuda/` and should be reported
as their own measurement, with the GPU name and software versions from `device_info`. Any
comparison with the MPS numbers needs repeated runs per device and the methodological
differences above stated alongside it.
