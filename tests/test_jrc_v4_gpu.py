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
        from jrc_moe.jrc_v4.controller import BudgetExchange, LayerwiseKController
        from jrc_moe.jrc_v4.patch import install_layerwise_patch
        from jrc_moe.jrc_v4.replay import ReplayState
        import jrc_moe.generation as generation

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

        # Exercise full replay and branch-order invariance with one 32-token
        # block to keep this opt-in integration check bounded.
        old_length, old_blocks = generation.GEN_LENGTH, generation.NUM_BLOCKS
        generation.GEN_LENGTH, generation.NUM_BLOCKS = 32, 1
        total = prompt_len + 32
        replay_controller = LayerwiseKController(names, min_k=2, max_k=8)
        install_layerwise_patch(model, replay_controller, verified)
        base_provider = lambda **kw: [[6] * total for _ in names]
        fast_provider = lambda **kw: [[8] * total for _ in names]
        state_box = {}
        def capture_first(state):
            state_box["state"] = state.clone()
            return True
        try:
            _, _, paused = generation.generate_with_routing_mode(
                model, encoded, "jrc_v4_test", replay_controller,
                allocation_provider=base_provider,
                before_forward=capture_first,
                before_forward_nfe=1, checkpoint_revision=MODEL_REVISION,
                config_digest="gpu-integration-test")
            self.assertTrue(paused.get("paused_before_forward"))
            state = state_box["state"]
            exchange = BudgetExchange(1, 0, prompt_len, prompt_len + 1, 1)
            state.selected_layer_idx = 0
            state.recipient_position = prompt_len
            state.donor_position = prompt_len + 1

            def run_order(order):
                seen = {}
                for branch in order:
                    seen[branch] = generation.generate_with_routing_mode(
                        model, encoded, "jrc_v4_test", replay_controller,
                        allocation_provider=fast_provider, resume_state=state.clone(),
                        branch=branch, exchange=exchange, checkpoint_revision=MODEL_REVISION,
                        config_digest="gpu-integration-test")
                return seen

            ab = run_order(("control", "treatment"))
            ba = run_order(("treatment", "control"))
            for branch in ("control", "treatment"):
                self.assertTrue(torch.equal(ab[branch][0], ba[branch][0]))
                self.assertTrue(ab[branch][1].drop(columns=["forward_seconds"], errors="ignore").equals(
                    ba[branch][1].drop(columns=["forward_seconds"], errors="ignore")))
                self.assertEqual(ab[branch][2]["total_nfe"], ba[branch][2]["total_nfe"])
                self.assertEqual(ab[branch][2]["total_pairs"], ba[branch][2]["total_pairs"])
            def target_layer_pairs(trace):
                row = trace[trace["global_nfe"] == 1].iloc[0]
                return int(row["layer_budgets"][0])
            self.assertEqual(target_layer_pairs(ab["control"][1]), target_layer_pairs(ab["treatment"][1]))
        finally:
            generation.GEN_LENGTH, generation.NUM_BLOCKS = old_length, old_blocks
            for _, module in model.named_modules():
                if hasattr(module, "_jrc_v4_original_forward"):
                    module.forward = module._jrc_v4_original_forward


if __name__ == "__main__":
    unittest.main()
