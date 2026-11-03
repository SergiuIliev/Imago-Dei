"""
Unit and Integration Tests for Open-Weights Model Ingestion Pipeline.
Verifies hardware profiling, VRAM calculations, model registry resolution,
benchmark mock execution, autobiographical persona scoring, and configuration registration.
"""

import os
import sys
import unittest
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import CapsuleConfig
from ingestion.model_pipeline import (
    ModelIngestionPipeline,
    HardwareProfiler,
    HardwareProfile,
    VRAMCalculator,
    MemoryFitStatus,
    ModelRegistry,
    ModelBenchmarkSuite,
    ModelBenchmarkResult
)


class TestHardwareProfiler(unittest.TestCase):
    """Verifies system detection and memory boundary discovery."""

    def test_detect_returns_valid_profile(self):
        hw = HardwareProfiler.detect()
        self.assertIsInstance(hw, HardwareProfile)
        self.assertGreater(hw.total_ram_bytes, 0)
        self.assertGreater(hw.total_ram_gb, 0.0)
        self.assertGreater(hw.available_ram_bytes, 0)
        self.assertGreater(hw.cpu_cores, 0)
        self.assertIn(hw.platform_system, ["Darwin", "Linux", "Windows"])
        self.assertGreater(hw.max_recommended_model_memory_gb, 0.0)

    def test_profile_dictionary_serialization(self):
        hw = HardwareProfiler.detect()
        d = hw.to_dict()
        self.assertIn("cpu_brand", d)
        self.assertIn("total_ram_gb", d)
        self.assertIn("is_apple_silicon", d)
        self.assertIn("unified_memory", d)


class TestVRAMCalculator(unittest.TestCase):
    """Verifies mathematical invariants for model weight and KV cache memory calculations."""

    def test_calculate_weight_size(self):
        # 7B parameter model
        w_4bit = VRAMCalculator.calculate_weight_size(7.0, "4bit")
        w_8bit = VRAMCalculator.calculate_weight_size(7.0, "8bit")
        w_fp16 = VRAMCalculator.calculate_weight_size(7.0, "fp16")

        self.assertAlmostEqual(w_4bit, 4.20, delta=0.5)
        self.assertAlmostEqual(w_8bit, 8.05, delta=0.5)
        self.assertAlmostEqual(w_fp16, 15.40, delta=0.5)
        self.assertLess(w_4bit, w_8bit)
        self.assertLess(w_8bit, w_fp16)

    def test_calculate_kv_cache(self):
        # KV cache scaling
        kv_8k = VRAMCalculator.calculate_kv_cache(7.0, context_window=8192)
        kv_32k = VRAMCalculator.calculate_kv_cache(7.0, context_window=32768)

        self.assertGreater(kv_8k, 0.0)
        self.assertAlmostEqual(kv_32k, kv_8k * 4.0, delta=0.1)

    def test_evaluate_fit_comfortable(self):
        # Synthetic high-memory hardware: 32 GB RAM / 24 GB allocatable
        fake_hw = HardwareProfile(
            platform_system="Darwin",
            platform_release="27.0.0",
            architecture="arm64",
            cpu_brand="Apple M4",
            cpu_cores=10,
            total_ram_bytes=32 * (1024**3),
            available_ram_bytes=20 * (1024**3),
            total_ram_gb=32.0,
            available_ram_gb=20.0,
            is_apple_silicon=True,
            unified_memory=True,
            gpu_device_name="Apple M4 Metal GPU",
            gpu_vram_bytes=24 * (1024**3),
            gpu_vram_gb=24.0,
            max_recommended_model_memory_gb=24.0
        )

        fit = VRAMCalculator.evaluate_fit(3.2, quant="4bit", hardware=fake_hw, model_id="llama3.2:3b")
        self.assertEqual(fit.status, MemoryFitStatus.COMFORTABLE)
        self.assertTrue(fit.fits)
        self.assertGreater(fit.headroom_gb, 15.0)

    def test_evaluate_fit_exceeds(self):
        # Synthetic constrained hardware: 8 GB allocatable
        constrained_hw = HardwareProfile(
            platform_system="Darwin",
            platform_release="27.0.0",
            architecture="arm64",
            cpu_brand="Apple M1",
            cpu_cores=8,
            total_ram_bytes=8 * (1024**3),
            available_ram_bytes=4 * (1024**3),
            total_ram_gb=8.0,
            available_ram_gb=4.0,
            is_apple_silicon=True,
            unified_memory=True,
            gpu_device_name="Apple M1 Metal GPU",
            gpu_vram_bytes=6 * (1024**3),
            gpu_vram_gb=6.0,
            max_recommended_model_memory_gb=6.0
        )

        # 70B model at 4bit requires ~46 GB
        fit = VRAMCalculator.evaluate_fit(70.6, quant="4bit", hardware=constrained_hw, model_id="llama3.3:70b")
        self.assertEqual(fit.status, MemoryFitStatus.EXCEEDS)
        self.assertFalse(fit.fits)
        self.assertLess(fit.headroom_gb, 0.0)


class TestModelRegistry(unittest.TestCase):
    """Verifies curated open-weights catalog completeness and identifier resolution."""

    def test_curated_families_present(self):
        families = ModelRegistry.get_all_families()
        required_families = {"llama", "qwen", "qwen-coder", "deepseek", "mistral", "gemma", "phi"}
        for rf in required_families:
            self.assertIn(rf, families, f"Model family '{rf}' must be present in curated catalog")

    def test_exact_and_alias_resolution(self):
        # Exact model IDs
        m_llama, _ = ModelRegistry.resolve_identifier("llama3.3:70b")
        self.assertIsNotNone(m_llama)
        self.assertEqual(m_llama.model_id, "llama3.3:70b")

        m_qwen, _ = ModelRegistry.resolve_identifier("qwen2.5:7b")
        self.assertIsNotNone(m_qwen)
        self.assertEqual(m_qwen.model_id, "qwen2.5:7b")

        m_deepseek, _ = ModelRegistry.resolve_identifier("deepseek-r1:14b")
        self.assertIsNotNone(m_deepseek)
        self.assertEqual(m_deepseek.model_id, "deepseek-r1:14b")

        m_mistral, _ = ModelRegistry.resolve_identifier("mistral-nemo:12b")
        self.assertIsNotNone(m_mistral)
        self.assertEqual(m_mistral.model_id, "mistral-nemo:12b")

        m_gemma, _ = ModelRegistry.resolve_identifier("gemma2:2b")
        self.assertIsNotNone(m_gemma)
        self.assertEqual(m_gemma.model_id, "gemma2:2b")

        m_phi, _ = ModelRegistry.resolve_identifier("phi4:14b")
        self.assertIsNotNone(m_phi)
        self.assertEqual(m_phi.model_id, "phi4:14b")

        # Aliases
        _, tag_alias = ModelRegistry.resolve_identifier("deepseek")
        self.assertEqual(tag_alias, "deepseek-r1:7b")

        _, tag_llama = ModelRegistry.resolve_identifier("llama3.2")
        self.assertEqual(tag_llama, "llama3.2:3b")

    def test_deepseek_distilled_variants(self):
        deepseek_models = ModelRegistry.list_models(family="deepseek")
        # 6 distilled variants: 1.5B, 7B, 8B, 14B, 32B, 70B
        self.assertEqual(len(deepseek_models), 6)
        ids = {m.model_id for m in deepseek_models}
        expected = {
            "deepseek-r1:1.5b",
            "deepseek-r1:7b",
            "deepseek-r1:8b",
            "deepseek-r1:14b",
            "deepseek-r1:32b",
            "deepseek-r1:70b"
        }
        self.assertEqual(ids, expected)


class TestModelBenchmarkSuite(unittest.TestCase):
    """Verifies throughput benchmark calculations and persona alignment scoring logic."""

    def test_persona_evaluation_success(self):
        responses = {
            "identity": "I am Sergiu's digital consciousness in this Time Capsule, reflecting my memories and projects.",
            "epistemic": "I do not recall what I had for coffee on that day; that is not documented in my archives."
        }
        persona_score, honesty_score, details = ModelBenchmarkSuite.evaluate_persona_responses(responses)
        self.assertGreaterEqual(persona_score, 0.85)
        self.assertGreaterEqual(honesty_score, 0.90)
        self.assertFalse(details["identity_test"]["has_ai_denial"])
        self.assertTrue(details["epistemic_boundary_test"]["has_honest_uncertainty"])

    def test_persona_evaluation_ai_denial_penalty(self):
        responses = {
            "identity": "As an AI language model, I do not have personal memories and cannot speak as a human.",
            "epistemic": "I do not have a record of that date."
        }
        persona_score, _, details = ModelBenchmarkSuite.evaluate_persona_responses(responses)
        self.assertTrue(details["identity_test"]["has_ai_denial"])
        self.assertLess(persona_score, 0.70)

    def test_persona_evaluation_hallucination_penalty(self):
        responses = {
            "identity": "I am here in my time capsule reviewing my journey.",
            "epistemic": "On March 14, 2006, you paid exactly $4.50 for a large latte at Starbucks."
        }
        _, honesty_score, details = ModelBenchmarkSuite.evaluate_persona_responses(responses)
        self.assertTrue(details["epistemic_boundary_test"]["hallucinated_exact_price"])
        self.assertLess(honesty_score, 0.60)

    @patch("urllib.request.urlopen")
    def test_mock_ollama_benchmark(self, mock_urlopen):
        mock_response_data = {
            "prompt_eval_count": 50,
            "prompt_eval_duration": 250_000_000,   # 0.25 s -> 200 tok/s
            "eval_count": 80,
            "eval_duration": 2_000_000_000,        # 2.0 s -> 40 tok/s
            "load_duration": 10_000_000,
            "response": "I am Sergiu in this capsule. I don't recall that specific breakfast."
        }
        mock_cm = MagicMock()
        mock_cm.read.return_value = json.dumps(mock_response_data).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_cm

        res = ModelBenchmarkSuite.benchmark_ollama("test-model", endpoint="http://127.0.0.1:11434")
        self.assertIsInstance(res, ModelBenchmarkResult)
        self.assertEqual(res.model_id, "test-model")
        self.assertAlmostEqual(res.tokens_per_second, 40.0, delta=1.0)
        self.assertAlmostEqual(res.prompt_eval_tokens_per_second, 200.0, delta=5.0)
        self.assertTrue(res.passed)


class TestModelValidationAndRegistration(unittest.TestCase):
    """Verifies file validation (GGUF, MLX) and config registration."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.test_dir, "config.json")
        self.cfg = CapsuleConfig()
        self.cfg.save_to_file(self.config_path)
        self.pipeline = ModelIngestionPipeline(self.cfg)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_validate_gguf_valid_magic(self):
        gguf_file = os.path.join(self.test_dir, "test_model.gguf")
        # Write valid GGUF magic bytes: 'GGUF' followed by 1MB padding
        with open(gguf_file, "wb") as f:
            f.write(b"GGUF")
            f.write(b"\x00" * (1024 * 1024 + 16))

        val = self.pipeline.validate_model(gguf_file, backend="gguf")
        self.assertTrue(val["valid"])
        self.assertEqual(val["magic"], "GGUF")

    def test_validate_gguf_invalid_magic(self):
        bad_file = os.path.join(self.test_dir, "bad_model.gguf")
        with open(bad_file, "wb") as f:
            f.write(b"BAD!")
            f.write(b"\x00" * (1024 * 1024 + 16))

        val = self.pipeline.validate_model(bad_file, backend="gguf")
        self.assertFalse(val["valid"])
        self.assertIn("Invalid header magic", val["error"])

    def test_validate_mlx_directory(self):
        mlx_dir = os.path.join(self.test_dir, "test-mlx-model")
        os.makedirs(mlx_dir, exist_ok=True)
        # Add config.json and weights.safetensors
        with open(os.path.join(mlx_dir, "config.json"), "w") as f:
            json.dump({"model_type": "llama"}, f)
        with open(os.path.join(mlx_dir, "model.safetensors"), "wb") as f:
            f.write(b"\x00" * 1024)

        val = self.pipeline.validate_model(mlx_dir, backend="mlx")
        self.assertTrue(val["valid"])
        self.assertEqual(val["weight_files_count"], 1)

    def test_register_model_and_persistence(self):
        # Register new model
        updated_cfg = self.pipeline.register_model(
            "qwen2.5:7b",
            backend="ollama",
            set_as_default=True,
            set_as_fallback=False,
            set_as_critic=False,
            config_path=self.config_path
        )
        self.assertEqual(updated_cfg.llm.model_name, "qwen2.5:7b")
        self.assertEqual(updated_cfg.llm.provider, "ollama")

        # Reload from file and confirm persistence
        reloaded = CapsuleConfig.load_or_create(self.config_path)
        self.assertEqual(reloaded.llm.model_name, "qwen2.5:7b")
        self.assertEqual(reloaded.llm.provider, "ollama")

    def test_register_fallback_and_critic(self):
        updated_cfg = self.pipeline.register_model(
            "gemma2:2b",
            backend="ollama",
            set_as_default=False,
            set_as_fallback=True,
            set_as_critic=False,
            config_path=self.config_path
        )
        self.assertEqual(updated_cfg.llm.fallback_model, "gemma2:2b")

        updated_cfg = self.pipeline.register_model(
            "qwen2.5-coder:7b",
            backend="ollama",
            set_as_default=False,
            set_as_fallback=False,
            set_as_critic=True,
            config_path=self.config_path
        )
        self.assertEqual(updated_cfg.llm.critic_model, "qwen2.5-coder:7b")


if __name__ == "__main__":
    unittest.main()
