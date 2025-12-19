import unittest
import torch
import numpy as np
from dim_est.models.critic_builders import make_critic
from dim_est.models.encoders import make_encoder
from dim_est.models.estimators import infonce_lower_bound, clip_lower_bound, smile_lower_bound

class TestComponents(unittest.TestCase):
    
    def test_encoder_factory(self):
        """Verify make_encoder produces valid modules with correct output shapes."""
        batch_size = 4
        
        # 1. MLP
        input_dim = 10
        embed_dim = 5
        enc = make_encoder("mlp", input_dim, embed_dim, hidden_dim=16)
        x = torch.randn(batch_size, input_dim)
        out = enc(x)
        self.assertEqual(out.shape, (batch_size, embed_dim))
        
        # 2. CNN (Input: B, C, H, W)
        enc = make_encoder("cnn", input_dim=0, embed_dim=8, input_channels=3)
        img = torch.randn(batch_size, 3, 32, 32)
        out = enc(img)
        self.assertEqual(out.shape, (batch_size, 8))
        
        # 3. RNN (GRU) (Input: B, Seq, Feat)
        enc = make_encoder("gru", input_dim=16, embed_dim=6)
        seq = torch.randn(batch_size, 10, 16)
        out = enc(seq)
        self.assertEqual(out.shape, (batch_size, 6))

    def test_critic_builders(self):
        """Verify critic builders return working modules."""
        # Separable
        critic, _, _ = make_critic("separable", {
            "Nx": 10, "Ny": 10, "embed_dim": 5, "encoder_type": "mlp"
        })
        x = torch.randn(4, 10)
        y = torch.randn(4, 10)
        scores = critic(x, y)
        self.assertEqual(scores.shape, (4, 4))
        
        # Concat
        critic, _, _ = make_critic("concat", {
            "Nx": 10, "Ny": 10, "pair_hidden_dim": 32
        })
        scores = critic(x, y)
        self.assertEqual(scores.shape, (4, 4))

    def test_siamese_check(self):
        """Verify share_encoder=True uses the same object."""
        critic, _, _ = make_critic("separable", {
            "Nx": 10, "Ny": 10, "embed_dim": 5, 
            "encoder_type": "mlp", 
            "share_encoder": True
        })
        self.assertIs(critic.encoder_x, critic.encoder_y)

    def test_estimators(self):
        """Verify estimators return scalar loss and valid extras."""
        scores = torch.randn(8, 8) # (Batch, Batch)
        
        # InfoNCE
        mi, extras = infonce_lower_bound(scores)
        self.assertTrue(torch.isfinite(mi))
        
        # L-Clip
        mi, extras = clip_lower_bound(scores)
        self.assertTrue(torch.isfinite(mi))
        self.assertIn("mi_i1", extras)
        
        # SMILE
        mi, extras = smile_lower_bound(scores, clip=5.0)
        self.assertTrue(torch.isfinite(mi))

if __name__ == '__main__':
    unittest.main()