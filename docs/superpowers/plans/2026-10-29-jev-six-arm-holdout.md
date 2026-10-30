# Jev Six-Arm Holdout Comparison Implementation Plan

**Goal:** Prepare the existing comparison runner for a six-arm evaluation on the new 20-task held-out manifest, with a five-hour Slurm allocation.

**Architecture:** Add a `holdout` profile to `experiments.compare_routers` for native LLM, legacy router, and Jev thresholds 0.50/0.65/0.70/0.75. Make execution counts and group rotation derive from the selected profile, then point the existing Slurm runner at the held-out task list and extend its time limit to five hours.

**Tech Stack:** Python, pytest, YAML, Bash, Slurm, Singularity.

**Run size:** 6 smoke cells plus 20 tasks × 6 arms = 126 total Agent executions. The five-hour limit is tight against the prior run rate and should be treated as a maximum; the report preserves partial progress if Slurm reaches the limit.

## Files

- Modify `experiments/compare_routers.py`: profile groups, threshold mapping, dynamic group-count accounting, and evaluation metadata.
- Modify `tests/test_compare_routers.py`: expected six-arm configurations and complete dynamic-run accounting.
- Modify `deploy/jev/router_comparison.sbatch`: holdout profile default, isolated run directory, five-hour limit, and source/input fingerprint coverage.
- Use `experiments/instances-jev-holdout-20.txt` as the fixed held-out input manifest; do not modify prior task lists or run evidence.

## Tasks

1. Add tests for the six-arm profile, thresholds and route isolation; add a fake-executor run asserting six smoke cells, six groups per task, correct progress totals, and complete report metadata.
2. Run the focused tests to confirm the new expectations fail against the prior runner.
3. Implement the profile and derive all cell-count/rotation logic from the selected group list; preserve baseline and threshold profiles.
4. Update the Slurm script to default to the holdout profile, use the new manifest path supplied at submission, record the current runner and manifest hashes, and request five hours.
5. Re-run focused tests and static checks; verify all 20 task IDs still belong to the unused `test_normal` families and are present locally.

The experiment itself is not submitted as part of this code-preparation plan; submission follows after the server receives and accepts these exact files.
