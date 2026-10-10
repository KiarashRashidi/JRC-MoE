# Replay validation

The v4 path now has a deep-copyable `ReplayState` contract, Python/NumPy/Torch RNG snapshots, a pre-forward callback on the shared REFLEX decoder, and a paired continuation wrapper that reuses the existing threshold/fallback generation loop. Its per-layer controller accepts layer-by-position K matrices and a one-unit donor/recipient exchange at one global NFE.

**Runtime validation status: not yet run.** CPU synthetic tests pass for state isolation, RNG restoration, layer locality, one-shot override accounting, and branch order. The G0 audit calls the multi-phase native/patched Fixed8 parity, real K=2/4/6/8 forwards, and no-op full-continuation replay checks. The pilot runner uses the same replay API for complete treatment/control continuations, but none of these runtime paths has been verified on the pinned checkpoint.

The replay state is captured at a forward boundary. Subsequent transformer pre-intervention layers are recomputed from copied token IDs; no full hidden-state trajectory is stored. The initial current decoder has deterministic argmax/threshold behavior and performs no random sampling, but the RNG snapshot remains part of the state and must be exercised by the GPU replay test before determinism is asserted.
