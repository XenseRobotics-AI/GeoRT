import unittest
import numpy as np
import torch
from geort.kinematics import FingerKinematics, TorchFingerKinematics
from geort.model import build_ik_model, ik_input
from geort.train_posture import initialize_model, posture_loss
from geort.utils.config_utils import get_config


class TrainingPostureTests(unittest.TestCase):
    def test_exact_fk_and_gradients(self):
        chain = FingerKinematics(get_config('wuji_hand2_beta1_right'))
        q = torch.tensor(np.random.default_rng(8).uniform(chain.lower, chain.upper, (8, 4)), requires_grad=True)
        tip, direction = TorchFingerKinematics(chain, dtype=torch.float64)(q)
        for i in range(len(q)):
            ref = chain.forward(q[i].detach().numpy())
            np.testing.assert_allclose(tip[i].detach(), ref[0], atol=1e-10)
            np.testing.assert_allclose(direction[i].detach(), ref[1], atol=1e-10)
            for k in range(3):
                grad = torch.autograd.grad(tip[i, k], q, retain_graph=True)[0][i]
                np.testing.assert_allclose(grad, ref[2][k], atol=1e-10)
        self.assertTrue(torch.autograd.gradcheck(TorchFingerKinematics(chain, dtype=torch.float64), (q,)))

    def test_legacy_initialization_and_frozen_fingers(self):
        cfg = get_config('wuji_hand2_beta1_right')
        parent = build_ik_model(cfg).eval()
        cfg2, model = initialize_model(cfg, parent)
        human = torch.randn(12, 21, 3) * .05
        before = parent(ik_input(cfg, human)).detach()
        torch.testing.assert_close(model(human), before)
        opt = torch.optim.Adam(model.nets[4].parameters(), lr=.01)
        opt.zero_grad()
        model(human)[:, -4:].square().mean().backward()
        opt.step()
        torch.testing.assert_close(model(human)[:, :16], before[:, :16], rtol=0, atol=0)
        restored = build_ik_model(cfg2).eval()
        restored.load_state_dict(model.state_dict())
        torch.testing.assert_close(restored(ik_input(cfg2, human)), model(human))
        with self.assertRaises(ValueError):
            restored(human[:, [4, 8, 12, 16, 20]])

    def test_degenerate_bone_and_single_frame_loss(self):
        q = torch.zeros(1, 4, requires_grad=True)
        tip = torch.zeros(1, 3)
        loss, terms = posture_loss(q, tip, tip, tip, torch.zeros(1, 21, 3))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertEqual(float(terms['temporal']), 0.)
        bent = q.detach().clone(); bent[:, 2:] = -1
        self.assertGreater(float(posture_loss(bent, tip, tip, tip, torch.zeros(1,21,3))[0]), float(loss))


if __name__ == '__main__':
    unittest.main()
