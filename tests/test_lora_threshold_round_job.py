from pathlib import Path
import unittest


SCRIPT = Path(__file__).parents[1] / "deploy" / "jev" / "lora_v1_dev_threshold_round.sbatch"


class LoraThresholdRoundJobTests(unittest.TestCase):
    def test_round_job_runs_all_thresholds_serially_in_one_24_hour_allocation(self):
        script = SCRIPT.read_text(encoding="utf-8")

        self.assertIn("#SBATCH --time=24:00:00", script)
        self.assertIn("#SBATCH --gres=gpu:A800:1", script)
        self.assertIn("THRESHOLDS=(0.5 0.55 0.6 0.65 0.7 0.75 0.8)", script)
        self.assertIn('source "$THRESHOLD_SCRIPT"', script)
        self.assertIn("cleanup\n  jev_pid=\n  trap - EXIT TERM INT", script)
        self.assertNotIn("sbatch ", script)
        self.assertIn('JEV_COMPARE_BASE="${JEV_COMPARE_BASE:?', script)
        self.assertIn("export JEV_COMPARE_BASE JEV_LARGE_API_BASE JEV_SMALL_API_BASE", script)


if __name__ == "__main__":
    unittest.main()
