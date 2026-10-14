"""Layer-local K plans and one-shot, budget-neutral exchange overrides."""
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class BudgetExchange:
    global_nfe: int
    layer_idx: int
    recipient_position: int
    donor_position: int
    amount: int = 1

    @property
    def candidate_positions(self) -> tuple[int, int]:
        return (int(self.recipient_position), int(self.donor_position))

    @property
    def position_deltas(self) -> tuple[tuple[int, int], tuple[int, int]]:
        return ((int(self.recipient_position), int(self.amount)),
                (int(self.donor_position), -int(self.amount)))

    def validate(self, num_layers: int, sequence_length: int, min_k: int, max_k: int) -> None:
        if self.global_nfe < 1 or not 0 <= self.layer_idx < num_layers:
            raise ValueError("exchange NFE/layer is out of range")
        if self.recipient_position == self.donor_position:
            raise ValueError("recipient and donor must differ")
        if not (0 <= self.recipient_position < sequence_length and 0 <= self.donor_position < sequence_length):
            raise ValueError("exchange position is out of range")
        if self.amount < 1 or min_k >= max_k:
            raise ValueError("exchange amount/K bounds are invalid")


class LayerwiseKController:
    """Validate and expose [layer, position] K values for exactly one forward.

    All inputs and accounting use Python values so CPU synthetic tests do not
    require a model download. A Torch dispatcher can convert returned rows to
    device tensors after the checkpoint routing semantics have been audited.
    """

    def __init__(self, expected_module_names: Sequence[str], *, min_k: int = 2, max_k: int = 8,
                 monitored_layers: Sequence[int] = (), monitored_positions: Sequence[int] = ()):
        self.module_names = tuple(expected_module_names)
        if not self.module_names or len(set(self.module_names)) != len(self.module_names):
            raise ValueError("expected module names must be nonempty and unique")
        self.min_k, self.max_k = int(min_k), int(max_k)
        if self.min_k < 1 or self.max_k < self.min_k:
            raise ValueError("invalid K bounds")
        self.monitored_layers = frozenset(int(i) for i in monitored_layers)
        self.monitored_positions = frozenset(int(i) for i in monitored_positions)
        if any(i < 0 or i >= len(self.module_names) for i in self.monitored_layers):
            raise ValueError("monitored layer index out of range")
        self.enabled = False
        self._plan: tuple[tuple[int, ...], ...] | None = None
        self._actual: list[list[int]] = []
        self._global_nfe = 0
        self._branch = "control"
        self._exchange: BudgetExchange | None = None
        self._calls: list[int] = []
        self._observed_layer_pairs: list[int | None] = []
        self._override_fires = 0
        self._route_records: dict[tuple[int, int], dict] = {}
        self.capture_expert_contributions = False
        self.mass_mode = False
        self._mass_summary: dict | None = None

    @property
    def sequence_length(self) -> int | None:
        return len(self._plan[0]) if self._plan else None

    def begin_forward(self, k_by_layer: Sequence, *, global_nfe: int, branch: str = "control",
                      exchange: BudgetExchange | None = None) -> None:
        if self.enabled:
            raise RuntimeError("previous K forward was not closed")
        if branch not in {"control", "treatment"}:
            raise ValueError("branch must be control or treatment")
        rows = list(k_by_layer)
        if rows and isinstance(rows[0], (int, float)):
            rows = [list(k_by_layer) for _ in self.module_names]
        if len(rows) != len(self.module_names) or not rows or not rows[0]:
            raise ValueError("K plan must be [position] or [layer, position]")
        width = len(rows[0])
        matrix = []
        flat_values = [value for row in rows for value in row]
        mass_mode = bool(flat_values) and all(isinstance(value, float) for value in flat_values)
        for row in rows:
            if len(row) != width:
                raise ValueError("all layer K rows must have the same sequence length")
            if mass_mode:
                values = tuple(float(v) for v in row)
                if any(not 0.0 < v <= 1.0 for v in values):
                    raise ValueError("mass plan values must be in (0, 1]")
            else:
                values = tuple(int(v) for v in row)
                if any(v < self.min_k or v > self.max_k for v in values):
                    raise ValueError(f"K plan values must be in [{self.min_k}, {self.max_k}]")
            matrix.append(values)
        if global_nfe < 1:
            raise ValueError("global_nfe is one-based and must be positive")
        if exchange is not None:
            if hasattr(exchange, "candidate_positions") and hasattr(exchange, "k_before"):
                exchange.validate(len(self.module_names), width, self.min_k, self.max_k)
            else:
                exchange.validate(len(self.module_names), width, self.min_k, self.max_k)
            if exchange.global_nfe == global_nfe:
                layer_plan = matrix[exchange.layer_idx]
                if hasattr(exchange, "k_before"):
                    live_before = tuple(layer_plan[position] for position in exchange.candidate_positions)
                    if live_before != tuple(exchange.k_before):
                        raise ValueError("multi-token action K_before differs from the active layer plan")
                if sum(delta for _, delta in exchange.position_deltas) != 0:
                    raise ValueError("exchange adjustments must sum to zero")
                for position, delta in exchange.position_deltas:
                    if not 0 <= position < width or not self.min_k <= layer_plan[position] + delta <= self.max_k:
                        raise ValueError("exchange adjustment is infeasible at the live K plan")
        if branch == "treatment" and exchange is None:
            raise ValueError("treatment branch requires an exchange")
        self._plan = tuple(matrix)
        self.mass_mode = mass_mode
        self._actual = [[0] * len(row) for row in matrix] if mass_mode else [list(row) for row in matrix]
        self._global_nfe = int(global_nfe)
        self._branch = branch
        self._exchange = exchange
        self._calls = [0] * len(self.module_names)
        self._observed_layer_pairs = [None] * len(self.module_names)
        self._override_fires = 0
        self._route_records = {}
        self._mass_summary = ({"expert_count_histogram": {}, "target_mass_histogram": {},
                               "expert_count_by_target": {}, "achieved_mass_sum_by_target": {},
                               "achieved_mass_sum": 0.0, "achieved_mass_min": 1.0,
                               "achieved_mass_max": 0.0, "mass_target_violations": 0,
                               "token_count": 0} if mass_mode else None)
        self.enabled = True

    def for_layer(self, layer_idx: int, module_name: str, sequence_length: int) -> tuple[int, ...]:
        if not self.enabled or self._plan is None:
            raise RuntimeError("begin_forward must be called before requesting a layer plan")
        if not 0 <= layer_idx < len(self.module_names):
            raise IndexError("layer index out of range")
        if self.module_names[layer_idx] != module_name:
            raise RuntimeError(f"module identity mismatch at layer {layer_idx}: {module_name!r}")
        if sequence_length != len(self._plan[layer_idx]):
            raise RuntimeError("allocation length does not match sequence length")
        self._calls[layer_idx] += 1
        if self._calls[layer_idx] != 1:
            raise RuntimeError(f"layer {layer_idx} invoked more than once in one forward")
        ex = self._exchange
        if self._branch == "treatment" and ex and ex.global_nfe == self._global_nfe and ex.layer_idx == layer_idx:
            if self._override_fires:
                raise RuntimeError("exchange override fired more than once")
            deltas = getattr(ex, "position_deltas", None)
            if deltas is None:
                deltas = ((ex.recipient_position, ex.amount), (ex.donor_position, -ex.amount))
            for position, delta in deltas:
                if not 0 <= position < sequence_length:
                    raise ValueError("exchange position is outside the live layer plan")
                adjusted = self._actual[layer_idx][position] + delta
                if not self.min_k <= adjusted <= self.max_k:
                    raise ValueError("exchange adjustment violates K bounds at application time")
                self._actual[layer_idx][position] = adjusted
            self._override_fires += 1
        return tuple(self._plan[layer_idx] if self.mass_mode else self._actual[layer_idx])

    def record_resolved_k(self, layer_idx: int, k_values: Sequence[int]) -> None:
        """Store per-token K selected from a mass target inside the router patch."""
        if not self.enabled or not self.mass_mode or not 0 <= layer_idx < len(self.module_names):
            raise RuntimeError("resolved K counts require an active mass-routed forward")
        values = tuple(int(v) for v in k_values)
        if len(values) != len(self._actual[layer_idx]) or any(v < 1 or v > 64 for v in values):
            raise ValueError("resolved K counts are outside the sequence/expert bounds")
        self._actual[layer_idx] = list(values)

    def record_mass_routing(self, layer_idx: int, target_masses: Sequence[float],
                            achieved_masses: Sequence[float]) -> None:
        """Aggregate realized mass and expert counts without retaining token logits."""
        if not self.enabled or not self.mass_mode or self._mass_summary is None:
            raise RuntimeError("mass statistics require an active mass-routed forward")
        targets = [float(v) for v in target_masses]
        masses = [float(v) for v in achieved_masses]
        counts = self._actual[layer_idx]
        if len(targets) != len(masses) or len(masses) != len(counts):
            raise ValueError("mass audit vectors differ in length")
        summary = self._mass_summary
        for target, mass, count in zip(targets, masses, counts):
            k_key, p_key = str(count), f"{target:.1f}"
            summary["expert_count_histogram"][k_key] = summary["expert_count_histogram"].get(k_key, 0) + 1
            summary["target_mass_histogram"][p_key] = summary["target_mass_histogram"].get(p_key, 0) + 1
            by_target = summary["expert_count_by_target"].setdefault(p_key, {})
            by_target[k_key] = by_target.get(k_key, 0) + 1
            summary["achieved_mass_sum_by_target"][p_key] = (
                summary["achieved_mass_sum_by_target"].get(p_key, 0.0) + mass)
            summary["achieved_mass_sum"] += mass
            summary["achieved_mass_min"] = min(summary["achieved_mass_min"], mass)
            summary["achieved_mass_max"] = max(summary["achieved_mass_max"], mass)
            summary["mass_target_violations"] += int(mass + 5e-4 < target)
            summary["token_count"] += 1

    def record_routing(self, layer_idx: int, position: int, experts: Sequence[int], weights: Sequence[float],
                       *, router_logits: Sequence[float] | None = None, hidden_state_norm: float | None = None) -> None:
        if (not self.enabled or layer_idx not in self.monitored_layers
                or (self.monitored_positions and position not in self.monitored_positions)):
            return
        if len(experts) != len(weights):
            raise ValueError("selected experts and weights differ in length")
        k = self._actual[layer_idx][position]
        if len(experts) != k:
            raise ValueError("recorded selected experts do not match actual K")
        record = {
            "module": self.module_names[layer_idx],
            "k": k,
            "experts": tuple(int(e) for e in experts),
            "weights": tuple(float(w) for w in weights),
            "hidden_state_norm": None if hidden_state_norm is None else float(hidden_state_norm),
            "expert_contributions": [],
        }
        if router_logits is not None:
            from .features import router_statistics
            record["router_statistics"] = router_statistics(router_logits, k, selected_experts=experts)
        self._route_records[(layer_idx, position)] = record

    def record_expert_contribution(self, layer_idx: int, position: int, expert_idx: int,
                                   router_weight: float, output_norm: float,
                                   weighted_output_norm: float) -> None:
        """Record optional token/expert contribution norms for bounded mechanism probes."""
        if not self.capture_expert_contributions:
            return
        record = self._route_records.get((layer_idx, position))
        if record is None:
            return
        record["expert_contributions"].append({
            "expert_id": int(expert_idx),
            "router_weight": float(router_weight),
            "expert_output_l2": float(output_norm),
            "weighted_output_l2": float(weighted_output_norm),
        })

    def record_layer_pairs(self, layer_idx: int, module_name: str, pair_count: int) -> None:
        if not self.enabled or not 0 <= layer_idx < len(self.module_names):
            raise RuntimeError("cannot record pair count outside an active forward")
        if module_name != self.module_names[layer_idx] or self._calls[layer_idx] != 1:
            raise RuntimeError("pair count recorded for the wrong layer/module")
        if self._observed_layer_pairs[layer_idx] is not None:
            raise RuntimeError("layer pair count recorded more than once")
        if pair_count < 0:
            raise ValueError("pair count cannot be negative")
        self._observed_layer_pairs[layer_idx] = int(pair_count)

    def end_forward(self) -> dict:
        if not self.enabled or self._plan is None:
            raise RuntimeError("no open controller forward")
        missing = [i for i, n in enumerate(self._calls) if n != 1 or self._observed_layer_pairs[i] is None]
        if missing:
            self.enabled = False
            raise RuntimeError(f"expected one call and one observed pair count per MoE layer; invalid layers={missing}")
        for i, row in enumerate(self._plan):
            if self._observed_layer_pairs[i] != sum(self._actual[i]):
                self.enabled = False
                raise RuntimeError(f"observed expert-token pairs disagree with actual K plan at layer {i}")
        ex = self._exchange
        should_fire = bool(self._branch == "treatment" and ex and ex.global_nfe == self._global_nfe)
        if self._override_fires != int(should_fire):
            self.enabled = False
            raise RuntimeError(f"override fired {self._override_fires} times; expected {int(should_fire)}")
        if ex and ex.global_nfe == self._global_nfe:
            before = sum(self._plan[ex.layer_idx])
            after = sum(self._actual[ex.layer_idx])
            if before != after:
                self.enabled = False
                raise RuntimeError("budget-neutral exchange changed target-layer budget")
            changed = [i for i, (a, b) in enumerate(zip(self._plan[ex.layer_idx], self._actual[ex.layer_idx])) if a != b]
            expected_deltas = getattr(ex, "position_deltas", None)
            if expected_deltas is None:
                expected_deltas = ((ex.recipient_position, ex.amount), (ex.donor_position, -ex.amount))
            expected_changed = {position for position, delta in expected_deltas if delta != 0}
            if self._branch == "treatment" and set(changed) != expected_changed:
                self.enabled = False
                raise RuntimeError("exchange changed positions outside its sparse delta action")
            if self._branch == "treatment":
                for position, delta in expected_deltas:
                    if self._actual[ex.layer_idx][position] - self._plan[ex.layer_idx][position] != delta:
                        self.enabled = False
                        raise RuntimeError("applied sparse K delta differs from requested action")
        result = {
            "global_nfe": self._global_nfe,
            "branch": self._branch,
            "module_names": self.module_names,
            "k_by_layer": tuple(tuple(row) for row in self._actual),
            "layer_pair_counts": tuple(int(n) for n in self._observed_layer_pairs),
            "direct_pairs": sum(int(n) for n in self._observed_layer_pairs),
            "override_fires": self._override_fires,
            "candidate_positions": tuple(getattr(ex, "candidate_positions", ())) if ex else (),
            "position_deltas": tuple(getattr(ex, "position_deltas", ())) if ex else (),
            "action_digest": getattr(ex, "action_digest", None) if ex else None,
            "routing_records": dict(self._route_records),
            "mass_summary": self._mass_summary,
        }
        self.enabled = False
        self._plan = None
        return result

    def abort_forward(self) -> None:
        """Release per-forward state after OOM, Ctrl+C, or a model exception."""
        self.enabled = False
        self._plan = None
        self._actual = []
        self._calls = []
        self._observed_layer_pairs = []
        self._route_records = {}
        self._mass_summary = None
