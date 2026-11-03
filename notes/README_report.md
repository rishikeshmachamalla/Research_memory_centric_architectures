# Report package (IEEE conference format, LaTeX)

`report/report.tex` is a **draft** built from your own run logs (`results/final_scalar`, `results/final_avx2`).
`report/report.pdf` is how it looks right now (figure boxes and bold `[...]` placeholders are intentional).

## 1. Fill the numbers that must come from YOUR runs
Two experiments were first tried on a different (cloud Xeon) machine while developing. Their numbers are **not** in the report;
run them on your laptop and enter your own values in the block at the top of `report.tex` (`\mbRotScalar` ... `\mutTotal`):

```bash
cd ~/research/pidcomm-sim && source .venv/bin/activate
CPU=2 ./cpp_backend/microbench/run_microbench.sh      # 1) kernel alone  2) real data path   (3 repeats each)
./cpp_backend/microbench/mutation_test.sh             # N failed / M passed with a deliberately broken rotation
```
* `\mbRotScalar \mbRotPtr \mbRotInl` = the "scalar via pointer", "AVX2 via pointer", "AVX2 direct (inlined)" lines of experiment 1.
* `\mbPathScalar \mbPathPtr \mbPathWide` = "scalar kernel, staged", "AVX2 kernel, staged [current]", "AVX2, staged but whole loop compiled for AVX2" of experiment 2.
* `\mutFail`, `\mutTotal` = failed count, and failed+passed, from the first line of the mutation test.
* Use a typical (median) of the 3 repeats. **Check the sentence** in Sec. VI-C: keep "refuted" only if your pointer and inlined rotate times are close;
  if your numbers look different, rewrite that paragraph to say what you measured.

## 2. Copy the figures into `report/figures/` (exact file names)
| file name in report/figures | copy from |
|---|---|
| `fig_ablation.png` | `results/final_scalar/cpp_results/fig_ablation.png` |
| `fig_model_pred_vs_measured.png` | `results/run_scalar/model/fig_pred_vs_measured.png` |
You can add more (data size, PE sweep, BFS: same folders) with the `\fig{file}{width}{label}` macro. Look at each figure before submitting.

## 3. Write the contribution statement
The last section is a placeholder on purpose. Each member should say honestly what they did.

## 4. Compile
Easiest: upload `report.tex` + the `figures/` folder to Overleaf (class `IEEEtran` is built in).
Locally: `latexmk -pdf report.tex` (Fedora: `sudo dnf install texlive-scheme-medium latexmk`).

## 5. Before you submit (checklist)
* Every number in the report must be traceable to a file in `results/final_*` - open them and compare.
* Everyone in the group must be able to explain every claim, especially the **negative results** (model does not beat "always L3";
  AVX2 gain is unreliable; PR alone slower than baseline; authors' code crashed and was not used).
* Check your course's rules on AI assistance and disclose use if required. Rewrite sections in your own words where you can.
* Tell the instructor the group has three members (one dropped the course) and that the authors' code could not be run
  (segfault in `dpu_rank_handler_get_rank`, cause not established) so a software model was used.
* The presentation is still to do: problem, paper idea, setup, reproduction vs paper (Table I), BFS, cost model (incl. negative results), limitations.
