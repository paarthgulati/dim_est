import unittest
import os
import shutil
import torch
import numpy as np
from dim_est.dim_est.run.run_single_experiment import run_dsib_infinite, run_dsib_finite
from dim_est.run.parallel_sweeps import run_sweep_parallel
from dim_est.utils.h5_result_store import H5ResultStore

class TestCoreIntegration(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Create a temp directory for all test outputs."""
        cls.test_dir = "test_suite_results"
        os.makedirs(cls.test_dir, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        """Cleanup after all tests are done."""
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)

    def test_infinite_run(self):
        outfile = os.path.join(self.test_dir, "test_inf.h5")
        # Run extremely short loop
        mi_est, cfg = run_dsib_infinite(
            dataset_type="joint_gaussian",
            setup="infinite_data_iter",
            outfile=outfile,
            training_overrides={"n_iter": 10, "show_progress": False},
            dataset_overrides={"latent": {"latent_dim": 2, "mi_bits": 1.0}},
            critic_overrides={"embed_dim": 2}
        )
        self.assertTrue(os.path.exists(outfile))
        self.assertIsInstance(mi_est, np.ndarray)

    def test_finite_synthetic_run(self):
        outfile = os.path.join(self.test_dir, "test_finite_syn.h5")
        mi_est, cfg = run_dsib_finite(
            dataset_type="joint_gaussian",
            setup="finite_data_epoch",
            outfile=outfile,
            training_overrides={"n_epoch": 1, "n_samples": 50, "batch_size": 10, "show_progress": False},
            dataset_overrides={"latent": {"latent_dim": 2, "mi_bits": 1.0}},
            critic_overrides={"embed_dim": 2}
        )
        self.assertTrue(os.path.exists(outfile))
        # Returns [train_trace, test_trace]
        self.assertEqual(len(mi_est), 2)

    def test_external_data_and_splitting(self):
        """Test loading external .pt file + 'temporal' split strategy."""
        data_path = os.path.join(self.test_dir, "temp_single_stream.pt")
        outfile = os.path.join(self.test_dir, "test_split.h5")
        
        # Create dummy time series (100, 4)
        single_stream = torch.randn(100, 4)
        torch.save(single_stream, data_path)
        
        try:
            run_dsib_finite(
                dataset_type="joint_gaussian", 
                setup="finite_data_epoch",
                outfile=outfile,
                training_overrides={"n_epoch": 1, "batch_size": 20, "show_progress": False},
                dataset_overrides={
                    "source": "external",
                    "data_path": data_path,
                    "split_strategy": "temporal",
                    "split_params": {"lag": 1}
                },
                critic_overrides={"embed_dim": 2, "Nx": 4, "Ny": 4}
            )
            self.assertTrue(os.path.exists(outfile))
        finally:
            if os.path.exists(data_path):
                os.remove(data_path)

    def test_parallel_sweep(self):
        """Test parallel execution helper."""
        outfile = os.path.join(self.test_dir, "test_parallel.h5")
        if os.path.exists(outfile):
            os.remove(outfile)
            
        # 2 simple jobs
        configs = []
        for i in range(2):
            cfg = {
                "dataset_type": "joint_gaussian",
                "setup": "finite_data_epoch",
                "training_overrides": {"n_epoch": 1, "n_samples": 20, "batch_size": 10, "show_progress": False},
                "dataset_overrides": {"latent": {"latent_dim": 2, "mi_bits": 1.0}},
                "critic_overrides": {"embed_dim": 2}
            }
            configs.append(cfg)
            
        run_sweep_parallel(
            func=run_dsib_finite,
            sweep_configs=configs,
            final_outfile=outfile,
            n_jobs=2,
            temp_dir=os.path.join(self.test_dir, "temp_sweeps")
        )
        
        self.assertTrue(os.path.exists(outfile))
        with H5ResultStore(outfile, "r") as rs:
            self.assertEqual(len(rs.list_runs()), 2)

if __name__ == '__main__':
    unittest.main()