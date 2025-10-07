import os
import tempfile
import unittest
import torch
import torch.nn as nn

from many import _standard_paths
from trading_bot.model_builder import create_hybrid_model, create_ltc_model, create_memory_focused_model, ModelBuilder

# A simple toy model used for tests
class ToyModelA(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 16)
        self.fc2 = nn.Linear(16, 1)
    def forward(self, x):
        x = self.fc1(x).relu()
        return self.fc2(x)

class ToyModelB(nn.Module):
    def __init__(self):
        super().__init__()
        # keep same fc1/fc2 shapes as ToyModelA but add an extra layer so keys differ
        self.fc1 = nn.Linear(10, 16)
        self.fc_extra = nn.Linear(1, 1)
        self.fc2 = nn.Linear(16, 1)
    def forward(self, x):
        x = self.fc1(x).relu()
        # extra layer not used in forward (just to create additional params)
        return self.fc2(x)


class ToyModelC(nn.Module):
    # model with size mismatches to force RuntimeError on load
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 8)
        self.fc2 = nn.Linear(8, 1)
    def forward(self, x):
        x = self.fc1(x).relu()
        return self.fc2(x)

class TestStateDictLoad(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='test_models_')
        self.model_path = os.path.join(self.tmpdir, 'TOY_model.pth')

    def tearDown(self):
        try:
            os.remove(self.model_path)
        except Exception:
            pass

    def test_strict_load_fails_on_mismatch(self):
        # Save ToyModelA state_dict
        a = ToyModelA()
        payload = {'model_state_dict': a.state_dict(), 'saved_at': 0.0}
        torch.save(payload, self.model_path)

        # Attempt to load into ToyModelC strictly should raise due to size mismatch
        c = ToyModelC()
        with self.assertRaises(Exception):
            c.load_state_dict(torch.load(self.model_path)['model_state_dict'])

    def test_non_strict_load_reports_missing_unexpected(self):
        # Save ToyModelA state_dict
        a = ToyModelA()
        payload = {'model_state_dict': a.state_dict(), 'saved_at': 0.0}
        torch.save(payload, self.model_path)

        # Non-strict load into ToyModelB should not raise but return keys info
        b = ToyModelB()
        res = b.load_state_dict(torch.load(self.model_path)['model_state_dict'], strict=False)
        # The result may be a NamedTuple or dict depending on torch version; check attributes
        missing = getattr(res, 'missing_keys', None)
        unexpected = getattr(res, 'unexpected_keys', None)
        # At least one of missing/unexpected should be non-empty lists
        self.assertTrue((isinstance(missing, list) and len(missing) > 0) or (isinstance(unexpected, list) and len(unexpected) > 0))

    def _roundtrip_builder_payload(self, builder: ModelBuilder, tmp_model_path: str):
        # ensure input size set
        if builder.input_size is None:
            builder.set_input_size(10)
        model = builder.build()
        payload = {'model_state_dict': model.state_dict(), 'model_config': builder.get_config(), 'saved_at': 0.0}
        torch.save(payload, tmp_model_path)
        # reload payload and reconstruct builder
        loaded = torch.load(tmp_model_path)
        self.assertIn('model_config', loaded)
        cfg = loaded['model_config']
        rb = ModelBuilder.load_config(cfg)
        self.assertIsInstance(rb, ModelBuilder)
        rb.set_input_size(cfg.get('input_size', 10))
        rebuilt = rb.build()
        # strict load should succeed because config describes same architecture
        rebuilt.load_state_dict(loaded['model_state_dict'])

    def test_hybrid_model_roundtrip(self):
        b = create_hybrid_model(input_size=10, hidden_size=128)
        self._roundtrip_builder_payload(b, self.model_path)

    def test_ltc_model_roundtrip(self):
        b = create_ltc_model(input_size=10, hidden_size=64)
        self._roundtrip_builder_payload(b, self.model_path)

    def test_memory_focused_model_roundtrip(self):
        b = create_memory_focused_model(input_size=10, hidden_size=64)
        self._roundtrip_builder_payload(b, self.model_path)

if __name__ == '__main__':
    unittest.main()
