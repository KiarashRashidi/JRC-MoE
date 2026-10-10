"""Opt-in GPU integration: JRC_MOE_RUN_GPU_INTEGRATION=1 on the model server."""
import os
import unittest


GPU_OPT_IN = os.environ.get("JRC_MOE_RUN_GPU_INTEGRATION") == "1"


@unittest.skipUnless(GPU_OPT_IN, "requires explicit opt-in and cached pinned LLaDA-MoE checkpoint")
class CheckpointFidelityTests(unittest.TestCase):
    def test_native_patch_multiphase_logits_router_and_legal_k(self):
        import torch
        from dataclasses import replace
        from jrc_moe.config import GEN_LENGTH, MASK_ID, MODEL_REVISION
        from jrc_moe.model import input_device, load_model
        from jrc_moe.jrc_v4.fidelity import native_patch_fidelity, verified_contract_from_loaded_model
        from jrc_moe.jrc_v4.controller import LayerwiseKController
        from jrc_moe.jrc_v4.patch import install_layerwise_patch

        model, tokenizer = load_model()
        encoded = tokenizer("Solve 1 + 1.", return_tensors="pt")["input_ids"].to(input_device(model))
        prompt_len = encoded.shape[1]
        total = prompt_len + GEN_LENGTH
        cases = []
        token = int(encoded[0, -1])
        for phase in (0, 1, 2):
            x = torch.full((1, total), MASK_ID, dtype=torch.long, device=encoded.device)
            x[:, :prompt_len] = encoded
            block = phase if phase < 2 else 1
            if phase == 1:
                x[:, prompt_len:prompt_len + 16] = token
                positions = tuple(range(prompt_len + 16, prompt_len + 32))
            elif phase == 2:
                x[:, prompt_len:prompt_len + 32] = token
                x[:, prompt_len + 32:prompt_len + 48] = token
                positions = tuple(range(prompt_len + 48, prompt_len + 64))
            else:
                positions = tuple(range(prompt_len, prompt_len + 32))
            cases.append((x, positions))

        contract, evidence = verified_contract_from_loaded_model(model, MODEL_REVISION)
        self.assertTrue(contract.router_semantics_verified, evidence)
        parity = native_patch_fidelity(model, None, contract, cases)
        self.assertTrue(parity["passed"], parity)
        verified = replace(contract, native_fixed8_equivalence_verified=True)
        names = tuple(name for name, module in model.named_modules()
                      if module.__class__.__name__ == "LLaDAMoESparseMoeBlock")
        x = cases[0][0][:, :prompt_len + 32]
        controller = LayerwiseKController(names, min_k=2, max_k=8)
        install_layerwise_patch(model, controller, verified)
        try:
            for nfe, k in enumerate((2, 4, 6, 8), start=1):
                controller.begin_forward([k] * x.shape[1], global_nfe=nfe)
                model(x)
                info = controller.end_forward()
                self.assertEqual(info["layer_pair_counts"], (k * x.shape[1],) * len(names))
        finally:
            for _, module in model.named_modules():
                if hasattr(module, "_jrc_v4_original_forward"):
                    module.forward = module._jrc_v4_original_forward


if __name__ == "__main__":
    unittest.main()
