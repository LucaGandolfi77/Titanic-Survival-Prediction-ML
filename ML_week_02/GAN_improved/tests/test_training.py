"""
Unit tests for training utilities.
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import tempfile

from src.utils.checkpointing import save_checkpoint, load_checkpoint, find_latest_checkpoint
from src.utils.config_loader import load_config, get_device, merge_configs


# ── Config Tests ─────────────────────────────────────────────────────────

class TestConfigLoader:
    def test_load_vanilla_config(self, tmp_path):
        """Test loading a YAML config file."""
        config_content = """
experiment:
  name: "test"
  stage: "vanilla_gan"
  seed: 42
  device: "cpu"
model:
  latent_dim: 64
training:
  n_epochs: 5
paths:
  data_dir: "data"
"""
        config_file = tmp_path / "config" / "test.yaml"
        config_file.parent.mkdir(parents=True)
        config_file.write_text(config_content)

        config = load_config(config_file)
        assert config["experiment"]["name"] == "test"
        assert config["model"]["latent_dim"] == 64

    def test_merge_configs(self):
        base = {"a": 1, "b": {"c": 2, "d": 3}}
        override = {"b": {"c": 99}, "e": 5}
        merged = merge_configs(base, override)
        assert merged["a"] == 1
        assert merged["b"]["c"] == 99
        assert merged["b"]["d"] == 3
        assert merged["e"] == 5

    def test_get_device_cpu(self):
        config = {"experiment": {"device": "cpu"}}
        device = get_device(config)
        assert device == torch.device("cpu")


# ── Checkpoint Tests ─────────────────────────────────────────────────────

class TestCheckpointing:
    @pytest.fixture
    def simple_models(self):
        gen = nn.Linear(10, 10)
        disc = nn.Linear(10, 1)
        opt_g = torch.optim.Adam(gen.parameters(), lr=0.001)
        opt_d = torch.optim.Adam(disc.parameters(), lr=0.001)
        return gen, disc, opt_g, opt_d

    def test_save_and_load(self, simple_models, tmp_path):
        gen, disc, opt_g, opt_d = simple_models

        # Save
        path = save_checkpoint(
            gen, disc, opt_g, opt_d,
            epoch=10, global_step=500,
            save_dir=tmp_path,
            metrics={"fid": 42.0},
        )
        assert path.exists()

        # Load into fresh models
        gen2 = nn.Linear(10, 10)
        disc2 = nn.Linear(10, 1)
        info = load_checkpoint(path, gen2, disc2, device="cpu")

        assert info["epoch"] == 10
        assert info["global_step"] == 500
        assert info["metrics"]["fid"] == 42.0

        # Verify weights match
        for p1, p2 in zip(gen.parameters(), gen2.parameters()):
            assert torch.allclose(p1, p2)

    def test_find_latest(self, simple_models, tmp_path):
        gen, disc, opt_g, opt_d = simple_models

        # Save multiple
        for epoch in [5, 10, 15]:
            save_checkpoint(gen, disc, opt_g, opt_d, epoch=epoch,
                          global_step=epoch * 100, save_dir=tmp_path)

        latest = find_latest_checkpoint(tmp_path)
        assert latest is not None
        assert "latest" in latest.name

    def test_cleanup_old_checkpoints(self, simple_models, tmp_path):
        gen, disc, opt_g, opt_d = simple_models

        for epoch in range(1, 11):
            save_checkpoint(gen, disc, opt_g, opt_d, epoch=epoch,
                          global_step=epoch * 100, save_dir=tmp_path,
                          keep_last_n=3)

        epoch_files = list(tmp_path.glob("checkpoint_epoch_*.pt"))
        assert len(epoch_files) == 3  # only last 3 kept
