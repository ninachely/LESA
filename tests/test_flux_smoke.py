"""Small CPU regression checks, without model downloads or CUDA."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/mlspace"))
sys.path.insert(0, str(REPO / "LESA_FLUX.1-dev_FLUX.1-schnell"))

from smoke_flux import select_prompts


class SmokeTests(unittest.TestCase):
    def test_train_test_do_not_overlap(self):
        train, test = select_prompts("train\nshared\n", "shared\ntest1\ntest2\n", 2)
        self.assertEqual(train, ["train", "shared"])
        self.assertEqual(test, ["test1", "test2"])

    def test_prompts_from_repository(self):
        root = REPO / "LESA_FLUX.1-dev_FLUX.1-schnell/prompts"
        train, test = select_prompts((root / "train_prompts_image.txt").read_text(),
                                    (root / "DrawBench200.txt").read_text())
        self.assertEqual(len(train), 8)
        self.assertEqual(len(test), 2)
        self.assertFalse(set(train) & set(test))

    def test_original_predictor_trains_and_reloads(self):
        import torch
        from predictor import Config, Predictor
        config = Config()
        config.model.device = "cpu"
        config.model.kan.hidden_dim = 8
        torch.manual_seed(0)
        predictor = Predictor(config, num_steps=50, enable_training=True)
        inputs = [torch.randn(1, 4, 8, dtype=torch.bfloat16) for _ in range(8)]
        times = [1 - i / 50 for i in range(51)]
        with torch.no_grad(), torch.set_grad_enabled(True):
            pred = predictor.model(inputs, times, 0.95)
            loss = predictor.train(pred, torch.zeros_like(pred))
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertEqual(predictor.scheduler.last_epoch, 1)
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "predictor.pt")
            predictor.save_model(path)
            restored = Predictor(config, num_steps=50, enable_training=False)
            restored.load_model(path)
            for a, b in zip(predictor.model.parameters(), restored.model.parameters()):
                torch.testing.assert_close(a, b)

    def test_smoke_schedule_has_sufficient_history(self):
        import math
        from predictor import cache_init, cal_type
        # Formula used by get_schedule(50, image_seq_len=4096, shift=True).
        mu = 1.15
        schedule = [math.exp(mu) / (math.exp(mu) + (1 / (1 - i / 50) - 1)) for i in range(50)]
        cache, current = cache_init(num_steps=50, height=1024, width=1024,
            test_FLOPs=False, monitor_gpu_usage=False, data_prepare=False,
            data_dir=None, phase=None, idx=0, interval=7, first_enhance=3)
        full_count = 0
        for step, t in enumerate(schedule):
            current["step"] = step
            cal_type(cache, current)
            if current["type"] == "cache":
                self.assertGreaterEqual(step, 4 if t > 0.9 else 8)
            else:
                full_count += 1
        self.assertGreater(full_count, 0)
        self.assertLess(full_count, 50)

    @unittest.skipUnless(importlib.util.find_spec("skimage"), "scikit-image is not installed")
    def test_psnr_avoids_uint8_overflow(self):
        import numpy as np
        from compare_flux import image_metrics
        black = np.zeros((16, 16, 3), dtype=np.uint8)
        white = np.full_like(black, 255)
        self.assertAlmostEqual(image_metrics(black, white)["psnr_db"], 0.0)
        self.assertEqual(image_metrics(black, black)["psnr_db"], "inf")
        self.assertAlmostEqual(image_metrics(black, black)["ssim_rgb"], 1.0)


if __name__ == "__main__":
    unittest.main()
