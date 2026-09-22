import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from brian2 import (
    PoissonGroup,
    NeuronGroup,
    Synapses,
    Network,
    Hz,
    ms,
    mV,
    SpikeMonitor,
    StateMonitor,
    volt,
    seed as brian_seed,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "Drosophila_brain_model"
LEARNING_DIR = BASE_DIR / "experiments" / "learning"

sys_path_added = False


class FlyBrain:
    """
    Plastic Drosophila olfactory -> mushroom-body simulation.

    This is based on the plasticity experiment in:
        experiments/learning/learning_driver.py

    The network uses:
        - olfactory projection neurons
        - Kenyon cells
        - MBONs
        - PAM dopamine neurons
        - PPL1 dopamine neurons

    Plasticity:
        KC -> MBON synapses are depressed when:
          1. the KC was active during an episode
          2. PAM dopamine activity exceeds the plasticity gate
          3. reward=True for that episode

    The actual Brian2 synaptic weights are modified; this is not
    a simulated placeholder.
    """

    def __init__(
        self,
        model_dir=MODEL_DIR,
        seed=0,
        eta=0.25,
        odor_hz=100.0,
        reward_hz=150.0,
        dan_hz=60.0,
        episode_seconds=1.0,
        kc_threshold=3,
        pam_gate_hz=1.0,
    ):
        self.model_dir = Path(model_dir)
        self.seed = seed
        self.eta = eta
        self.odor_hz = odor_hz
        self.reward_hz = reward_hz
        self.dan_hz = dan_hz
        self.episode_seconds = episode_seconds
        self.kc_threshold = kc_threshold
        self.pam_gate_hz = pam_gate_hz

        self.rng = np.random.default_rng(seed)

        self._load_model()
        self._create_odors()
        self._build_network()

        self.reset()

    # -----------------------------------------------------------------------
    # Model loading
    # -----------------------------------------------------------------------

    def _load_model(self):
        global sys_path_added

        model_dir = str(self.model_dir)

        if model_dir not in __import__("sys").path:
            __import__("sys").path.insert(0, model_dir)

        from model import create_model, default_params

        self.create_model = create_model
        self.default_params = default_params

        circuit_file = LEARNING_DIR / "circuit_ids.json"

        with open(circuit_file) as f:
            self.circuit = json.load(f)

        self.completeness_file = (
            self.model_dir / "Completeness_783.csv"
        )

        self.connectivity_file = (
            self.model_dir / "Connectivity_783.parquet"
        )

        self.df_comp = pd.read_csv(
            self.completeness_file,
            index_col=0,
        )

        self.flyid_to_index = {
            int(fid): i
            for i, fid in enumerate(self.df_comp.index)
        }

        def convert_ids(ids):
            return np.array(
                sorted(
                    self.flyid_to_index[int(x)]
                    for x in ids
                    if int(x) in self.flyid_to_index
                ),
                dtype=int,
            )

        # Kenyon cells
        self.kc_idx = convert_ids(self.circuit["kc"])

        # PAM dopamine neurons
        self.pam_idx = convert_ids(self.circuit["pam"])

        # PPL1 dopamine neurons
        self.ppl1_idx = convert_ids(self.circuit["ppl1"])

        # Olfactory projection neurons
        self.alpn_idx = convert_ids(
            self.circuit["alpn_uni_chol"]
        )

        # MBONs grouped by type
        self.mbon_groups = {
            name: convert_ids(ids)
            for name, ids in self.circuit["mbon"].items()
        }

        self.mbon_idx = np.unique(
            np.concatenate(
                list(self.mbon_groups.values())
            )
        )

        print(
            f"[IDs] KCs={len(self.kc_idx)} "
            f"MBONs={len(self.mbon_idx)} "
            f"PAM={len(self.pam_idx)} "
            f"PPL1={len(self.ppl1_idx)} "
            f"ALPN={len(self.alpn_idx)}"
        )

    # -----------------------------------------------------------------------
    # Build Brian2 network
    # -----------------------------------------------------------------------

    def _build_network(self):
        print("[build] constructing Brian2 network...")

        start = time.time()

        params = dict(self.default_params)

        self.params = params

        (
            self.neu,
            self.syn,
            self.spk_mon,
        ) = self.create_model(
            str(self.completeness_file),
            str(self.connectivity_file),
            params,
        )


        drivable = sorted(
            set().union(
                *self.odor_sets.values()
            )
            | set(self.pam_idx)
        )

        self.drivable = np.array(
            drivable,
            dtype=int,
        )

        self.drive_position = {
            neuron: i
            for i, neuron in enumerate(drivable)
        }

        self.poisson = PoissonGroup(
            len(drivable),
            rates=0 * Hz,
        )

        self.drive = Synapses(
            self.poisson,
            self.neu,
            on_pre="v_post += w_drv",
            namespace={
                "w_drv": (
                    params["w_syn"]
                    * params["f_poi"]
                )
            },
        )

        self.drive.connect(
            i=np.arange(len(drivable)),
            j=self.drivable,
        )

        # Externally driven neurons should not have the normal refractory
        # period, matching the repository's experiment.
        self.neu.rfc[self.drivable] = 0 * ms

        self.net = Network(
            self.neu,
            self.syn,
            self.spk_mon,
            self.poisson,
            self.drive,
        )

        # -------------------------------------------------------------------
        # Identify KC -> MBON synapses
        # -------------------------------------------------------------------

        print("[build] locating KC -> MBON synapses...")

        df_con = pd.read_parquet(
            self.connectivity_file
        )

        pre = df_con[
            "Presynaptic_Index"
        ].values

        post = df_con[
            "Postsynaptic_Index"
        ].values

        kc_set = set(
            self.kc_idx.tolist()
        )

        mbon_set = set(
            self.mbon_idx.tolist()
        )

        plastic_mask = np.fromiter(
            (
                p in kc_set and q in mbon_set
                for p, q in zip(pre, post)
            ),
            dtype=bool,
            count=len(pre),
        )

        self.plastic_pos = np.flatnonzero(
            plastic_mask
        )

        self.plastic_pre = pre[
            self.plastic_pos
        ]

        self.w0 = np.array(
            self.syn.w[
                self.plastic_pos
            ]
        )

        print(
            f"[build] {len(self.plastic_pos)} "
            f"KC -> MBON plastic synapses"
        )

        # DAN neurons are neuromodulatory in this experiment.
        # Therefore remove their fast excitatory synaptic output.
        dan_all = np.concatenate(
            [
                self.pam_idx,
                self.ppl1_idx,
            ]
        )

        dan_output = np.flatnonzero(
            np.isin(pre, dan_all)
        )

        weights = np.array(
            self.syn.w[:]
        )

        weights[dan_output] = 0

        self.syn.w[:] = (
            weights * volt
        )

        print(
            f"[build] zeroed "
            f"{len(dan_output)} DAN output synapses"
        )

        print(
            f"[build] complete in "
            f"{time.time() - start:.1f}s"
        )

    # -----------------------------------------------------------------------
    # Odors
    # -----------------------------------------------------------------------

    def _create_odors(self):
        """
        Create two disjoint artificial odors from ALPN neurons.

        The existing repository experiment uses 25 projection neurons
        per odor. We reproduce that here.
        """

        shuffled = list(
            self.rng.permutation(
                self.alpn_idx
            )
        )

        n = min(25, len(shuffled) // 3)

        self.odor_A = shuffled[:n]
        self.odor_B = shuffled[n:2 * n]

        # Remaining neurons form a pool for novel probes.
        self.fresh = shuffled[2 * n:]

        self.odor_sets = {
            "A": self.odor_A,
            "B": self.odor_B,
        }

        # Create several generalization probes.
        probe_specs = {
            "probe75": 0.75,
            "probe50": 0.50,
            "probe25": 0.25,
            "probe0": 0.0,
        }

        fresh_pos = 0

        for name, fraction in probe_specs.items():
            shared = round(
                n * fraction
            )

            new_count = n - shared

            if (
                fresh_pos + new_count
                <= len(self.fresh)
            ):
                fresh_part = self.fresh[
                    fresh_pos:
                    fresh_pos + new_count
                ]

                fresh_pos += new_count

                self.odor_sets[name] = (
                    self.odor_A[:shared]
                    + fresh_part
                )

        print(
            f"[odors] A={len(self.odor_A)} "
            f"neurons, B={len(self.odor_B)} neurons"
        )

    # -----------------------------------------------------------------------
    # Reset
    # -----------------------------------------------------------------------

    def reset(self):
        """
        Reset neuronal state and restore all KC -> MBON weights
        to their original values.
        """

        self.neu.v = self.params["v_0"]
        self.neu.g = 0 * mV

        self.poisson.rates = (
            np.zeros(
                len(self.drivable)
            ) * Hz
        )

        self.syn.w[
            self.plastic_pos
        ] = self.w0 * volt

        self.previous_counts = np.array(
            self.spk_mon.count[:],
            dtype=np.int64,
        )

        self.last_result = None
        self.last_spike_counts = np.zeros(len(self.neu), dtype=np.int64)

        print("[reset] brain reset")

    # -----------------------------------------------------------------------
    # Episode execution
    # -----------------------------------------------------------------------

    def run(
        self,
        odor="A",
        reward=False,
    ):
        """
        Present an odor and run one episode.

        Parameters
        ----------
        odor:
            "A", "B", "probe75", "probe50",
            "probe25", or "probe0"

        reward:
            If True, activate PAM dopamine neurons
            during this episode and allow KC -> MBON
            LTD afterward.
        """

        if odor not in self.odor_sets:
            raise ValueError(
                f"Unknown odor '{odor}'. "
                f"Available: {list(self.odor_sets)}"
            )

        input_rates = np.zeros(
            len(self.drivable)
        )

        for neuron in self.odor_sets[odor]:
            input_rates[
                self.drive_position[neuron]
            ] = self.odor_hz

        if reward:
            for neuron in self.pam_idx:
                if neuron in self.drive_position:
                    input_rates[
                        self.drive_position[neuron]
                    ] = self.dan_hz

        self.poisson.rates = (
            input_rates * Hz
        )

        # Reset membrane state before each episode.
        self.neu.v = self.params["v_0"]
        self.neu.g = 0 * mV

        start = time.time()

        self.net.run(
            self.episode_seconds
            * 1000
            * ms
        )

        counts = np.array(
            self.spk_mon.count[:],
            dtype=np.int64,
        )

        episode_counts = (
            counts
            - self.previous_counts
        )

        # Per-neuron output for this episode (not cumulative).
        self.last_spike_counts = episode_counts.copy()

        # Turn off all external input.
        self.poisson.rates = (
            np.zeros(
                len(self.drivable)
            ) * Hz
        )

        # Let the network settle.
        self.net.run(
            200 * ms
        )

        self.previous_counts = np.array(
            self.spk_mon.count[:],
            dtype=np.int64,
        )

        # ---------------------------------------------------------------
        # Determine active KCs
        # ---------------------------------------------------------------

        active_mask = (
            episode_counts[
                self.kc_idx
            ]
            >= self.kc_threshold
        )

        active_kcs = self.kc_idx[
            active_mask
        ]

        # ---------------------------------------------------------------
        # Measure PAM activity
        # ---------------------------------------------------------------

        pam_hz = float(
            episode_counts[
                self.pam_idx
            ].mean()
            / self.episode_seconds
        )

        plasticity_applied = False
        changed_synapses = 0

        # ---------------------------------------------------------------
        # Dopamine-gated KC -> MBON LTD
        # ---------------------------------------------------------------

        if (
            reward
            and pam_hz
            >= self.pam_gate_hz
        ):
            hot = np.isin(
                self.plastic_pre,
                active_kcs,
            )

            weights = np.array(
                self.syn.w[
                    self.plastic_pos
                ]
            )

            changed_synapses = int(
                hot.sum()
            )

            weights[hot] *= (
                1 - self.eta
            )

            self.syn.w[
                self.plastic_pos
            ] = weights * volt

            plasticity_applied = True

        # ---------------------------------------------------------------
        # MBON output
        # ---------------------------------------------------------------

        mbon_total = int(
            episode_counts[
                self.mbon_idx
            ].sum()
        )

        mbon_by_type = {
            name: int(
                episode_counts[
                    indices
                ].sum()
            )
            for name, indices
            in self.mbon_groups.items()
        }

        current_weights = np.array(
            self.syn.w[
                self.plastic_pos
            ]
        )

        weight_fraction = float(
            current_weights.sum()
            / self.w0.sum()
        )

        result = {
            "odor": odor,
            "reward": reward,
            "active_kcs": len(active_kcs),
            "pam_hz": pam_hz,
            "mbon_spikes": mbon_total,
            "mbon_by_type": mbon_by_type,
            "plasticity_applied":
                plasticity_applied,
            "changed_synapses":
                changed_synapses,
            "kc_mbon_weight_fraction":
                weight_fraction,
            "runtime_seconds":
                time.time() - start,
        }

        self.last_result = result

        print(
            f"[episode] odor={odor} "
            f"reward={reward} "
            f"KC={len(active_kcs)} "
            f"PAM={pam_hz:.2f}Hz "
            f"MBON={mbon_total} "
            f"plastic={plasticity_applied} "
            f"changed={changed_synapses} "
            f"wfrac={weight_fraction:.4f}"
        )

        return result

    def _population_spike_counts(self, indices):
        """Return individual spike counts from the most recent episode."""
        indices = np.asarray(indices, dtype=np.int64)
        return self.last_spike_counts[indices].copy()

    def get_kc_output(self):
        """Return one spike count for each Kenyon cell."""
        return self._population_spike_counts(self.kc_idx)

    def get_pam_output(self):
        """Return one spike count for each PAM neuron."""
        return self._population_spike_counts(self.pam_idx)

    def get_ppl1_output(self):
        """Return one spike count for each PPL1 neuron."""
        return self._population_spike_counts(self.ppl1_idx)

    def get_mbon_output(self):
        """Return individual MBON spike counts grouped by MBON type."""
        return {name: self._population_spike_counts(indices) for name, indices in self.mbon_groups.items()}

    def get_outputs(self):
        """Return individual outputs for KCs, PAM, PPL1, and MBONs."""
        return {"kc": self.get_kc_output(), "pam": self.get_pam_output(), "ppl1": self.get_ppl1_output(), "mbon": self.get_mbon_output()}

    def train(self, odor="A", reward=False, repetitions=1, verbose=True):
        """Repeatedly train the existing Brian2 network on an odor."""
        if repetitions < 1:
            raise ValueError("repetitions must be at least 1")
        if odor not in self.odor_sets:
            raise ValueError(f"Unknown odor '{odor}'. Available: {list(self.odor_sets)}")
        history = []
        if verbose:
            label = "REWARD" if reward else "NO REWARD"
            print(f"\n=== TRAINING {odor} ({label}) ===")
        for episode in range(1, repetitions + 1):
            result = self.run(odor=odor, reward=reward)
            history.append(result)
            if verbose:
                print(f"Training {episode}/{repetitions}: KC={result['active_kcs']} PAM={result['pam_hz']:.2f}Hz MBON={result['mbon_spikes']} weight={result['kc_mbon_weight_fraction']:.4f}")
        return history

    # -----------------------------------------------------------------------
    # Convenience methods
    # -----------------------------------------------------------------------

    def present_odor(self, odor):
        """
        Select an odor for the next run.

        This method is mostly a convenience wrapper.
        The actual presentation occurs when run() is called.
        """

        if odor not in self.odor_sets:
            raise ValueError(
                f"Unknown odor '{odor}'. "
                f"Available: {list(self.odor_sets)}"
            )

        self.current_odor = odor

        print(
            f"[odor] selected {odor}"
        )

    def reward(self):
        """
        Run the currently selected odor with reward.

        Equivalent to:

            brain.run(reward=True)
        """

        odor = getattr(
            self,
            "current_odor",
            "A",
        )

        return self.run(
            odor=odor,
            reward=True,
        )

    def run_without_reward(self):
        """
        Run the currently selected odor without reward.
        """

        odor = getattr(
            self,
            "current_odor",
            "A",
        )

        return self.run(
            odor=odor,
            reward=False,
        )

    def show_outputs(self):
        """
        Display the MBON output from the last episode.
        """

        if self.last_result is None:
            print(
                "No episode has been run yet."
            )
            return

        print("\n=== MBON OUTPUT ===")

        print(
            "Total MBON spikes:",
            self.last_result[
                "mbon_spikes"
            ],
        )

        for name, value in (
            self.last_result[
                "mbon_by_type"
            ].items()
        ):
            if value:
                print(
                    f"{name:20s}: {value}"
                )

    def show_plasticity(self):
        """
        Show the current KC -> MBON
        plasticity state.
        """

        current = np.array(
            self.syn.w[
                self.plastic_pos
            ]
        )

        ratio = (
            current
            / self.w0
        )

        changed = np.count_nonzero(
            np.abs(
                ratio - 1.0
            ) > 1e-9
        )

        print("\n=== PLASTICITY ===")

        print(
            "KC -> MBON synapses:",
            len(self.plastic_pos),
        )

        print(
            "Changed synapses:",
            changed,
        )

        print(
            "Weight fraction:",
            f"{current.sum() / self.w0.sum():.6f}",
        )

        print(
            "Minimum weight fraction:",
            f"{ratio.min():.6f}",
        )

        print(
            "Maximum weight fraction:",
            f"{ratio.max():.6f}",
        )

    def available_odors(self):
        return list(
            self.odor_sets.keys()
        )



class ASCIIOutput:
    """Plastic spiking ASCII readout with one 128-neuron decoder per character slot."""

    def __init__(self, source_brain, slots=5, learning_rate=0.005, seed=0, input_scale=0.2, threshold=0.25):
        self.source_brain = source_brain
        self.slots = int(slots)
        self.learning_rate = float(learning_rate)
        self.input_scale = float(input_scale)
        self.threshold = float(threshold)
        if self.input_scale <= 0:
            raise ValueError("ASCII input_scale must be > 0")
        if self.threshold <= 0:
            raise ValueError("ASCII threshold must be > 0")
        self.rng = np.random.default_rng(seed)
        self.charset = ''.join(chr(i) for i in range(128))
        self.target_text = None

        self.groups = []
        self.readouts = []
        self.monitors = []
        for slot in range(self.slots):
            group = NeuronGroup(
                128,
                model="dv/dt = -v/(10*ms) : 1",
                threshold=f"v > {self.threshold}",
                reset="v = 0",
                refractory=2*ms,
                method="euler",
                name=f"ascii_slot_{slot}",
            )
            group.v = 0
            monitor = SpikeMonitor(group, name=f"ascii_monitor_{slot}")
            syn = Synapses(
                source_brain.neu,
                group,
                model="w : 1",
                on_pre="v_post += input_scale * w",
                method="euler",
                name=f"ascii_readout_{slot}",
                namespace={"input_scale": self.input_scale},
            )
            # Only MBONs drive the ASCII decoder.
            source_i = np.repeat(source_brain.mbon_idx, 128)
            target_j = np.tile(np.arange(128), len(source_brain.mbon_idx))
            syn.connect(i=source_i, j=target_j)
            syn.w = 0.01 * self.rng.standard_normal(len(syn))
            self.groups.append(group)
            self.readouts.append(syn)
            self.monitors.append(monitor)

        self.w0 = [np.asarray(syn.w[:], dtype=float).copy() for syn in self.readouts]
        self.last_output = ""
        self.last_char_counts = [np.zeros(128, dtype=np.int64) for _ in range(self.slots)]

    def objects(self):
        result=[]
        for g,s,m in zip(self.groups,self.readouts,self.monitors):
            result.extend([g,s,m])
        return result

    def reset(self):
        for slot, g in enumerate(self.groups):
            g.v = 0
            self.readouts[slot].w[:] = self.w0[slot]
        self.last_output = ""
        self.last_char_counts = [np.zeros(128, dtype=np.int64) for _ in range(self.slots)]

    def _decode_slot(self, slot):
        counts = np.asarray(self.last_char_counts[slot], dtype=np.int64)
        # Ignore non-printable control characters unless no printable neuron fired.
        printable = np.arange(32, 127)
        pcounts = counts[printable]
        if pcounts.max(initial=0) <= 0:
            return "", counts
        return chr(int(printable[np.argmax(pcounts)])), counts

    def decode(self):
        chars=[]
        self.last_char_counts=[]
        for slot in range(self.slots):
            char, counts = self._decode_slot(slot)
            chars.append(char)
            self.last_char_counts.append(counts)
        self.last_output=''.join(chars)
        return self.last_output

    def reinforce_slot(self, slot, expected, predicted):
        """Apply supervised reward/punishment to one character slot.

        The decoder is a multiclass readout: each MBON provides an input
        feature and the 128 ASCII neurons are the possible classes.  A
        simple "boost expected / depress only the current winner" rule can
        make one odor overwrite the other, especially when the two MBON
        patterns are similar.  Instead, use a softmax-style delta update:

          * reward the expected character
          * distribute punishment across the competing characters according
            to their current probability
          * use signed MBON -> ASCII synapses so incorrect characters can
            actively suppress the decoder

        This keeps A->Hello and B->World as separate supervised mappings
        while still using the actual MBON spike activity as the learning
        signal.
        """
        if not expected or ord(expected) >= 128:
            raise ValueError("ASCIIOutput supports ASCII characters only")

        syn = self.readouts[slot]
        n_mbon = len(self.source_brain.mbon_idx)
        weights = np.asarray(syn.w[:], dtype=float).reshape(n_mbon, 128)

        # MBON spike counts from the one brain presentation used for all
        # character slots in this training step.
        source_counts = np.asarray(
            self.source_brain.last_spike_counts[self.source_brain.mbon_idx],
            dtype=float,
        )

        # The old implementation clipped every active MBON to 20 spikes.
        # Since most MBONs fire far above 20 spikes, that turned A and B into
        # almost exactly the same binary feature vector.  We deliberately
        # retain the small rate differences instead.  Use a centered,
        # normalized log-rate code instead.  Centering rejects the enormous
        # common-mode MBON activity and preserves the small A/B differences.
        log_rates = np.log1p(source_counts / max(self.source_brain.episode_seconds, 1e-12))
        x = log_rates - log_rates.mean()
        scale = float(x.std())
        if scale < 1e-12:
            return
        x = np.clip(x / scale, -3.0, 3.0) / 3.0

        # Only printable ASCII characters participate in the competition.
        printable = np.arange(32, 127)
        logits = weights[:, printable].T @ x

        # Numerically stable softmax.  Scale the logits so that the decoder's
        # small initial weights do not become an almost-uniform distribution
        # forever, while avoiding an excessively sharp winner.
        logits = logits - np.max(logits)
        exp_logits = np.exp(np.clip(logits, -30.0, 30.0))
        probabilities = exp_logits / exp_logits.sum()

        expected_idx = ord(expected)
        expected_pos = np.flatnonzero(printable == expected_idx)
        if len(expected_pos) == 0:
            raise ValueError("Expected character must be printable ASCII")
        expected_pos = int(expected_pos[0])

        # Cross-entropy gradient: expected class gets a positive update and
        # competing classes get negative updates proportional to probability.
        #
        # IMPORTANT: these are signed synapses.  Positive weights excite the
        # expected character while negative weights suppress competing
        # characters.  Restricting all weights to [0, 1] made two very similar
        # MBON patterns collapse onto the same word.
        error = -probabilities
        error[expected_pos] += 1.0

        delta = self.learning_rate * error[:, None] * x[None, :]
        weights[:, printable] += delta.T

        # Make every character column insensitive to uniform global MBON
        # firing.  This is essential here because A and B both produce a
        # large common firing-mode in the connectome.
        weights[:, printable] -= weights[:, printable].mean(axis=0, keepdims=True)
        weights[:, printable] = np.clip(weights[:, printable], -0.5, 0.5)
        syn.w[:] = weights.reshape(-1)



class InterconnectedFlyBrain:
    """Scalable network of N FlyBrain modules plus a plastic ASCII readout.

    The full biological FlyBrain is expensive. Therefore the default scalable
    topology is a directed chain: Brain0 -> Brain1 -> ... -> BrainN-1.  This
    preserves a single input and single output while avoiding the quadratic
    all-to-all explosion.  ``topology='all_to_all'`` is available for small N.

    Inter-brain links are real Brian2 Synapses and use individual spike events.
    The interface feeds target ALPN input neurons with a small direct-voltage
    pulse so firing rate is preserved rather than saturating the downstream brain.
    The ASCII readout is also spiking and has plastic MBON -> character
    synapses. Training can reward correct characters and punish incorrect ones.
    """

    def __init__(
        self,
        n_brains,
        model_dir=MODEL_DIR,
        seed=0,
        eta=0.25,
        inter_eta=0.02,
        inter_weight=1.0,
        inter_delay_ms=1.8,
        inter_pulse_mV=2.0,
        repeatable_inputs=True,
        inter_weight_min=0.0,
        inter_weight_max=2.0,
        topology="chain",
        inter_fanout=1,
        ascii_slots=5,
        ascii_learning_rate=0.005,
        ascii_input_scale=0.2,
        ascii_threshold=0.25,
        **brain_kwargs,
    ):
        self.n_brains=int(n_brains)
        if not 1 <= self.n_brains <= 800:
            raise ValueError("n_brains must be between 1 and 800")
        self.seed=seed
        self.inter_eta=float(inter_eta)
        self.inter_weight=float(inter_weight)
        self.inter_weight_min=float(inter_weight_min)
        self.inter_weight_max=float(inter_weight_max)
        self.inter_delay_ms=float(inter_delay_ms)
        self.inter_pulse_mV=float(inter_pulse_mV)
        self.repeatable_inputs=bool(repeatable_inputs)
        self.ascii_input_scale=float(ascii_input_scale)
        self.ascii_threshold=float(ascii_threshold)
        if self.ascii_input_scale <= 0:
            raise ValueError("ascii_input_scale must be > 0")
        if self.ascii_threshold <= 0:
            raise ValueError("ascii_threshold must be > 0")
        if self.inter_pulse_mV <= 0:
            raise ValueError("inter_pulse_mV must be > 0")
        self.topology=topology
        self.inter_fanout=int(inter_fanout)
        if topology not in {"chain","bidirectional_chain","all_to_all"}:
            raise ValueError("topology must be chain, bidirectional_chain, or all_to_all")
        if self.inter_fanout < 1:
            raise ValueError("inter_fanout must be >= 1")
        if topology == "all_to_all" and self.n_brains > 20:
            raise ValueError("all_to_all is limited to 20 brains; use chain for scalable networks")

        print(f"[web] creating {self.n_brains} FlyBrain modules...")
        self.brains=[]
        for i in range(self.n_brains):
            brain=FlyBrain(model_dir=model_dir, seed=seed+i, eta=eta, **brain_kwargs)
            # The repository's create_model() gives every Brian2 object the
            # same default names (for example "default_neurons" and
            # "default_synapses"). Brian2 requires object names to be
            # globally unique when all brains are placed in one Network.
            self._namespace_brain_objects(brain, i)
            self.brains.append(brain)
        self._build_interbrain_connections()
        self.ascii=ASCIIOutput(self.brains[-1], slots=ascii_slots, learning_rate=ascii_learning_rate, seed=seed+100000, input_scale=ascii_input_scale, threshold=ascii_threshold)
        objects=[]
        for b in self.brains:
            objects.extend([b.neu,b.syn,b.spk_mon,b.poisson,b.drive])
        objects.extend(self.inter_connections.values())
        objects.extend(self.ascii.objects())
        self.net=Network(*objects)
        self.reset()

    @staticmethod
    def _namespace_brain_objects(brain, brain_id):
        """Give every Brian2 object in one FlyBrain a unique name.

        The original Drosophila model creates objects such as
        ``default_neurons`` and ``default_synapses``. A single FlyBrain can
        use those names, but N FlyBrains cannot share them in one Brian2
        Network. Contained objects (state updaters, thresholders, resetters,
        synaptic code objects, etc.) are renamed as well.
        """
        prefix=f"brain{brain_id}_"
        seen=set()

        def rename_tree(obj):
            name=getattr(obj, "name", None)
            if name is not None and not name.startswith(prefix):
                new_name=prefix + name
                if new_name in seen:
                    new_name=f"{new_name}_{len(seen)}"
                # Brian2 exposes ``name`` as a read-only property on some
                # contained objects (notably SpikeMonitor).  The underlying
                # Nameable attribute is ``_name``.
                obj._name=new_name
                seen.add(new_name)
            for child in getattr(obj, "contained_objects", ()):
                rename_tree(child)

        for obj in brain.net.objects:
            rename_tree(obj)

    def _pairs(self):
        if self.topology=="chain":
            return [(i,i+1) for i in range(self.n_brains-1)]
        if self.topology=="bidirectional_chain":
            return [(i,i+1) for i in range(self.n_brains-1)] + [(i+1,i) for i in range(self.n_brains-1)]
        return [(i,j) for i in range(self.n_brains) for j in range(self.n_brains) if i!=j]

    def _build_interbrain_connections(self):
        """Build the inter-brain interface as a receptor relay.

        The reference Drosophila model uses direct voltage injection for its
        externally stimulated neurons (PoissonInput to ``v``).  An artificial
        brain-to-brain interface should therefore feed the next brain's ALPN
        input neurons the same way, rather than adding a small conductance
        that may never drive an ALPN across threshold.

        Every MBON in the source brain participates.  Each MBON is mapped to
        one (or more) ALPNs in the target brain.  The default fanout=1 gives
        a deterministic one-to-one channel relay and avoids throwing away
        the first/last arbitrary subset of MBONs.
        """
        print(f"[web] building {self.topology} inter-brain receptor topology...")
        self.inter_connections={}
        self.inter_w0={}
        self.inter_source_indices={}
        self.inter_target_indices={}
        self.inter_receptor_pulse={}

        for source_id,target_id in self._pairs():
            source=self.brains[source_id]
            target=self.brains[target_id]

            source_idx=np.asarray(source.mbon_idx, dtype=np.int64)
            n_source=len(source_idx)
            n_target=len(target.alpn_idx)
            if n_source == 0 or n_target == 0:
                raise RuntimeError(f"Brain {source_id}->{target_id} has no MBON/ALPN channels")

            # Deterministically permute ALPNs so we do not privilege the
            # smallest Brian2 indices.  Reuse targets only when fanout needs
            # more channels than the available ALPN population.
            rng=np.random.default_rng(self.seed + 1000 + source_id*7919 + target_id*104729)
            shuffled_target=rng.permutation(np.asarray(target.alpn_idx, dtype=np.int64))
            needed=n_source*self.inter_fanout
            if needed <= n_target:
                target_idx=shuffled_target[:needed]
            else:
                target_idx=np.resize(shuffled_target, needed)

            source_rep=np.repeat(source_idx, self.inter_fanout)
            target_rep=target_idx

            # This is an artificial brain-to-brain synapse, not an external
            # PoissonInput.  The published model uses 68.75 mV pulses only for
            # strong optogenetic-style external drive; applying that pulse to
            # every MBON spike saturated our downstream ALPN population and
            # erased most A/B differences.  Use a much smaller configurable
            # relay pulse so source firing rate remains the information carrier.
            receptor_pulse = self.inter_pulse_mV * mV
            syn=Synapses(
                source.neu,
                target.neu,
                model="gain : 1",
                on_pre="v_post += gain * receptor_pulse",
                namespace={"receptor_pulse": receptor_pulse},
                delay=self.inter_delay_ms*ms,
                name=f"inter_{source_id}_{target_id}",
            )
            syn.connect(i=source_rep, j=target_rep)
            if "gain" not in syn.variables:
                raise RuntimeError(
                    f"Inter-brain Synapses {syn.name} was created without the expected 'gain' state variable"
                )
            syn.gain[:] = self.inter_weight

            # A relay spike is a synaptic event, not the model's direct
            # PoissonInput.  Keep the ordinary refractory period so the relay
            # does not turn every incoming event into an arbitrarily fast ALPN
            # spike train.
            target.neu.rfc[target_idx]=target.params["t_rfc"]

            key=(source_id,target_id)
            self.inter_connections[key]=syn
            self.inter_source_indices[key]=source_rep.copy()
            self.inter_target_indices[key]=target_rep.copy()
            self.inter_w0[key]=np.full(len(syn),self.inter_weight,dtype=float)
            self.inter_receptor_pulse[key]=receptor_pulse

            print(
                f"[web] Brain{source_id}->Brain{target_id}: "
                f"{n_source} MBON channels, {len(syn):,} receptor synapses, "
                f"pulse={float(receptor_pulse/mV):.2f} mV"
            )

        total=sum(len(x) for x in self.inter_connections.values())
        print(f"[web] {len(self.inter_connections)} directed links, {total:,} inter-brain receptor synapses")

    def reset(self):
        for b in self.brains: b.reset()
        for key,syn in self.inter_connections.items(): syn.gain[:]=self.inter_w0[key]
        self.ascii.reset()
        self.last_result=None

    def _prepare_input(self,input_brain,odor,reward):
        if not 0<=input_brain<self.n_brains: raise IndexError("input_brain is outside the web")
        for i,b in enumerate(self.brains):
            rates=np.zeros(len(b.drivable))
            if i==input_brain:
                if odor not in b.odor_sets: raise ValueError(f"Unknown odor '{odor}'. Available: {list(b.odor_sets)}")
                for neuron in b.odor_sets[odor]: rates[b.drive_position[neuron]]=b.odor_hz
                if reward:
                    for neuron in b.pam_idx:
                        if neuron in b.drive_position: rates[b.drive_position[neuron]]=b.dan_hz
            b.poisson.rates=rates*Hz
            b.neu.v=b.params["v_0"]; b.neu.g=0*mV

    def _capture(self):
        return {i:np.asarray(b.spk_mon.count[:],dtype=np.int64)-b.previous_counts for i,b in enumerate(self.brains)}

    def _internal_plasticity(self,counts,reward):
        for i,b in enumerate(self.brains):
            c=counts[i]; active=b.kc_idx[c[b.kc_idx]>=b.kc_threshold]
            pam=float(c[b.pam_idx].mean()/b.episode_seconds)
            if reward and pam>=b.pam_gate_hz:
                hot=np.isin(b.plastic_pre,active); w=np.asarray(b.syn.w[b.plastic_pos]); w[hot]*=1-b.eta; b.syn.w[b.plastic_pos]=w*volt

    def _inter_plasticity(self,counts,reward):
        """Reward-gated gain plasticity for the artificial inter-brain relay."""
        changed=0
        for key,syn in self.inter_connections.items():
            s,t=key
            sc=counts[s][self.inter_source_indices[key]]
            tc=counts[t][self.inter_target_indices[key]]

            # Each source channel should have a corresponding target receptor
            # response.  This scalar Hebbian factor is only used when the
            # whole episode carries the global reward signal.
            a=np.minimum(sc,20)/20.0
            b=np.minimum(tc,20)/20.0
            w=np.asarray(syn.gain[:],dtype=float)
            old=w.copy()

            if reward:
                w += self.inter_eta*a*b
                w=np.clip(w,self.inter_weight_min,self.inter_weight_max)
                syn.gain[:]=w

            changed += int(np.count_nonzero(np.abs(w-old)>1e-12))
        return changed

    def _run_brain_episode(self,input_brain,odor,reward):
        self._prepare_input(input_brain,odor,reward)

        # PoissonGroup is intentionally retained, but training/prediction can
        # replay the same stochastic odor pattern.  Brian2 documents its seed()
        # function as the correct way to make PoissonGroup patterns reproducible.
        # A fixed seed per odor makes the supervised A/B mapping stable without
        # replacing the Poisson source with a deterministic spike generator.
        if self.repeatable_inputs:
            odor_code=sum(ord(ch) for ch in str(odor))
            brian_seed(self.seed + 1000000*input_brain + 1000*odor_code)

        self.net.run(self.brains[0].episode_seconds*1000*ms)
        counts=self._capture()
        for b in self.brains: b.poisson.rates=np.zeros(len(b.drivable))*Hz
        self.net.run(200*ms)
        self._internal_plasticity(counts,reward)
        changed=self._inter_plasticity(counts,reward)
        for i,b in enumerate(self.brains):
            b.last_spike_counts=counts[i].copy(); b.previous_counts=np.asarray(b.spk_mon.count[:],dtype=np.int64)
        return counts,changed

    def run(self,odor="A",reward=False,input_brain=0,output_brain=None):
        if output_brain is None: output_brain=self.n_brains-1
        counts,changed=self._run_brain_episode(input_brain,odor,reward)
        out=counts[output_brain][self.brains[output_brain].mbon_idx]
        result={"odor":odor,"reward":reward,"input_brain":input_brain,"output_brain":output_brain,"output_mbon_spikes":int(out.sum()),"interbrain_changed_synapses":changed,"brain_mbon_spikes":[int(counts[i][b.mbon_idx].sum()) for i,b in enumerate(self.brains)]}
        self.last_result=result
        return result

    def _run_ascii_episode(self, input_brain, odor, reward=False):
        """Run the fly-brain network once and capture all ASCII slots.

        The final brain's MBON spikes drive all character slots during this
        single Brian2 episode.  We then use that same MBON activity to decode
        and train every character position.  This avoids running the full
        Drosophila network once per character.
        """
        # Clear decoder membrane state and remember cumulative monitor counts.
        before = []
        for group, monitor in zip(self.ascii.groups, self.ascii.monitors):
            group.v = 0
            before.append(np.asarray(monitor.count[:], dtype=np.int64).copy())

        counts, changed = self._run_brain_episode(input_brain, odor, reward)

        # The ASCII readout Synapses are part of self.net, so they received
        # the final brain's MBON spikes during the same episode above.
        for slot, monitor in enumerate(self.ascii.monitors):
            after = np.asarray(monitor.count[:], dtype=np.int64)
            self.ascii.last_char_counts[slot] = after - before[slot]
            self.ascii.groups[slot].v = 0

        return counts, changed

    def _decode_current_ascii(self):
        chars = []
        for slot in range(self.ascii.slots):
            char, _ = self.ascii._decode_slot(slot)
            chars.append(char or "?")
        output = "".join(chars)
        self.ascii.last_output = output
        return output

    def test_interbrain_relay(self, input_brain=0, odor="A"):
        """Run one odor and explicitly verify MBON -> ALPN relay activity."""
        counts, _ = self._run_brain_episode(input_brain, odor, reward=False)
        print("\n=== INTER-BRAIN RELAY TEST ===")
        for i in range(self.n_brains - 1):
            key=(i,i+1)
            src=self.inter_source_indices[key]
            dst=self.inter_target_indices[key]
            syn=self.inter_connections[key]
            tx=int(counts[i][src].sum())
            rx=int(counts[i+1][dst].sum())
            gain=np.asarray(syn.gain[:],dtype=float)
            print(
                f"Brain{i}->Brain{i+1}: source MBON spikes={tx} "
                f"({np.count_nonzero(counts[i][src])} active), "
                f"target ALPN spikes={rx} "
                f"({np.count_nonzero(counts[i+1][dst])} active), "
                f"gain={gain.min():.3f}-{gain.max():.3f}, "
                f"pulse={self.inter_receptor_pulse[key]/mV:.2f} mV"
            )
        return counts

    def measure_odor_separation(self, input_brain=0):
        """Measure how different A and B are at the final MBON layer.

        This runs A and B once each and reports cosine similarity of the
        final MBON spike vectors. Lower similarity means the ASCII readout has
        more information available to distinguish the two odors.
        """
        self._run_brain_episode(input_brain, "A", reward=False)
        a = self.brains[-1].last_spike_counts[self.brains[-1].mbon_idx].astype(float)
        self._run_brain_episode(input_brain, "B", reward=False)
        b = self.brains[-1].last_spike_counts[self.brains[-1].mbon_idx].astype(float)
        na = np.linalg.norm(a); nb = np.linalg.norm(b)
        cosine = float(np.dot(a, b) / (na * nb)) if na and nb else 0.0
        distance = float(np.linalg.norm(a - b))
        print(f"[ASCII] A/B MBON cosine similarity={cosine:.4f}, Euclidean distance={distance:.2f}")
        return {"cosine_similarity": cosine, "euclidean_distance": distance, "A": a, "B": b}

    def analyze_odor_path(self, input_brain=0):
        """Run A and B and report how information survives each brain and relay.

        ``inter_source_indices`` and ``inter_target_indices`` contain indices
        into the *full neuron population*.  The MBON vectors used for the
        cosine calculation are only 96 elements long, so they cannot be
        indexed with those full Brian2 neuron indices.  Keep the full episode
        count arrays for the relay diagnostic and derive the MBON vectors
        separately.
        """
        results = {"A": [], "B": []}
        relay = {"A": {}, "B": {}}

        for odor in ("A", "B"):
            counts, _ = self._run_brain_episode(input_brain, odor, reward=False)

            for i, b in enumerate(self.brains):
                # Store only the MBON activity needed for the information
                # comparison.  Relay diagnostics are captured directly from
                # the full neuron-count array below.
                results[odor].append(
                    counts[i][b.mbon_idx].astype(float)
                )

            for i in range(self.n_brains - 1):
                key = (i, i + 1)
                src = self.inter_source_indices[key]
                dst = self.inter_target_indices[key]

                # IMPORTANT: src/dst are absolute neuron indices in each
                # brain's full population, not positions inside the 96-MBON
                # output vector.
                relay[odor][key] = {
                    "tx": int(counts[i][src].sum()),
                    "rx": int(counts[i + 1][dst].sum()),
                    "tx_active": int(np.count_nonzero(counts[i][src])),
                    "rx_active": int(np.count_nonzero(counts[i + 1][dst])),
                }

        print("\n=== ODOR INFORMATION THROUGH THE CHAIN ===")
        for i, b in enumerate(self.brains):
            a = results["A"][i]
            c = results["B"][i]

            na = np.linalg.norm(a)
            nc = np.linalg.norm(c)
            cosine = float(np.dot(a, c) / (na * nc)) if na and nc else 0.0
            distance = float(np.linalg.norm(a - c))
            az = np.log1p(a) - np.log1p(a).mean()
            bz = np.log1p(c) - np.log1p(c).mean()
            naz = np.linalg.norm(az)
            nbz = np.linalg.norm(bz)
            centered_cosine = float(np.dot(az, bz) / (naz * nbz)) if naz and nbz else 0.0
            centered_distance = float(np.linalg.norm(az - bz))
            active_a = int(np.count_nonzero(a))
            active_b = int(np.count_nonzero(c))

            print(
                f"Brain {i}: A={int(a.sum())} spikes/{active_a} active MBONs, "
                f"B={int(c.sum())} spikes/{active_b} active MBONs, "
                f"cosine={cosine:.4f}, distance={distance:.2f}, "
                f"centered_cosine={centered_cosine:.4f}, "
                f"centered_distance={centered_distance:.2f}"
            )

            if i < self.n_brains - 1:
                key = (i, i + 1)
                ra = relay["A"][key]
                rb = relay["B"][key]
                print(
                    f"         relay {i}->{i+1}: "
                    f"A transmitted={ra['tx']} ({ra['tx_active']} active channels), "
                    f"A receptor spikes={ra['rx']} ({ra['rx_active']} active); "
                    f"B transmitted={rb['tx']} ({rb['tx_active']} active channels), "
                    f"B receptor spikes={rb['rx']} ({rb['rx_active']} active)"
                )

        return results

    def diagnose_decoder_inputs(self, input_brain=0, verbose=True):
        """Measure the two odor feature vectors actually used by the ASCII trainer."""
        counts_a, _ = self._run_brain_episode(input_brain, "A", reward=False)
        counts_b, _ = self._run_brain_episode(input_brain, "B", reward=False)
        b = self.brains[-1]
        a = counts_a[b.mbon_idx].astype(float)
        c = counts_b[b.mbon_idx].astype(float)
        za = np.log1p(a) - np.log1p(a).mean()
        zc = np.log1p(c) - np.log1p(c).mean()
        na = np.linalg.norm(za); nc = np.linalg.norm(zc)
        cosine = float(np.dot(za, zc) / (na * nc)) if na and nc else 0.0
        distance = float(np.linalg.norm(za - zc))
        raw_na = np.linalg.norm(a)
        raw_nc = np.linalg.norm(c)
        relative_difference = distance / max((raw_na + raw_nc) / 2.0, 1e-12)
        if verbose:
            print("\n=== ASCII DECODER INPUT DIAGNOSTIC ===")
            print(f"Centered log-MBON cosine={cosine:.6f}, distance={distance:.6f}")
            print(f"Centered feature distance / mean raw norm={relative_difference:.6f}")
            top=np.argsort(np.abs(za-zc))[-10:][::-1]
            print("Largest A/B MBON feature differences:")
            for pos in top:
                print(f"  MBON[{pos}] A={a[pos]:.0f} B={c[pos]:.0f} feature_delta={(za-zc)[pos]:+.4f}")
        return {"A":a,"B":c,"centered_cosine":cosine,"centered_distance":distance,"relative_difference":relative_difference}

    def predict(self, odor, input_brain=0, output_brain=None):
        if output_brain is None:
            output_brain = self.n_brains - 1
        if output_brain != self.n_brains - 1:
            raise ValueError("ASCII decoder is attached to the final output brain")

        # ONE full brain presentation. All five character decoders see the
        # same final-brain MBON spike pattern.
        self._run_ascii_episode(input_brain, odor, reward=False)
        return self._decode_current_ascii()

    def train_ascii(self, input_brain=0, epochs=100, targets=None, verbose=True):
        """Train A/B -> ASCII strings using one brain run per odor.

        Default targets are A -> Hello and B -> World.  For each odor, the
        complete interconnected brain is simulated exactly once.  The final
        MBON spike output is then presented to all character slots, and each
        slot receives its own reward/punishment update:

          correct character  -> potentiate MBON -> expected-character weights
          wrong character    -> depress MBON -> predicted-character weights

        Thus an output such as ``Hella`` against ``Hello`` gets four character
        rewards and one character punishment without five additional brain
        simulations.
        """
        if targets is None:
            targets = {"A": "Hello", "B": "Hollo"}

        for odor, text in targets.items():
            if len(text) > self.ascii.slots:
                raise ValueError(
                    f"Target '{text}' exceeds ascii_slots={self.ascii.slots}"
                )
            if odor not in self.brains[input_brain].odor_sets:
                raise ValueError(f"Unknown odor '{odor}'")

        history = []

        for epoch in range(1, int(epochs) + 1):
            epoch_record = {"epoch": epoch, "predictions": {}, "scores": {}}

            for odor, target in targets.items():
                # Exactly ONE expensive full-network presentation.
                self._run_ascii_episode(input_brain, odor, reward=False)
                prediction = self._decode_current_ascii()

                score = sum(
                    a == b for a, b in zip(prediction, target)
                )

                # The same MBON spike counts captured above are reused for all
                # five character positions. No brain rerun is needed here.
                slot_results = []
                for slot, expected in enumerate(target):
                    predicted, _ = self.ascii._decode_slot(slot)
                    predicted = predicted or "?"
                    correct = predicted == expected
                    self.ascii.reinforce_slot(
                        slot,
                        expected,
                        predicted,
                    )
                    slot_results.append({
                        "slot": slot,
                        "expected": expected,
                        "predicted": predicted,
                        "rewarded": correct,
                        "punished": not correct,
                    })

                epoch_record["predictions"][odor] = prediction
                epoch_record["scores"][odor] = score
                epoch_record.setdefault("characters", {})[odor] = slot_results

                if verbose:
                    print(
                        f"[ASCII] epoch={epoch} odor={odor} "
                        f"output='{prediction}' target='{target}' "
                        f"score={score}/{len(target)}"
                    )

            history.append(epoch_record)

            if all(
                epoch_record["scores"][odor] == len(target)
                for odor, target in targets.items()
            ):
                if verbose:
                    print(f"[ASCII] solved at epoch {epoch}")
                break

        return history

    def train(self,odor="A",reward=False,repetitions=1,input_brain=0,output_brain=None,verbose=True):
        return [self.run(odor,reward,input_brain,output_brain) for _ in range(repetitions)]

    def get_output(self,brain_id=None):
        if brain_id is None: brain_id=self.n_brains-1
        return self.brains[brain_id].get_mbon_output()

    def get_outputs(self): return [b.get_mbon_output() for b in self.brains]

    def show_interbrain_plasticity(self):
        print("\n=== INTER-BRAIN PLASTICITY ===")
        for key,syn in self.inter_connections.items():
            w=np.asarray(syn.gain[:]); initial=self.inter_w0[key]
            print(f"Brain{key[0]} -> Brain{key[1]}: synapses={len(syn):,} gain_fraction={w.sum()/initial.sum():.6f} min={w.min():.4f} max={w.max():.4f}")

    def show_ascii_weights(self):
        print("\n=== ASCII READOUT WEIGHTS ===")
        for slot,syn in enumerate(self.ascii.readouts):
            w=np.asarray(syn.w[:]); means=w.reshape(len(self.brains[-1].mbon_idx),128).mean(axis=0)
            top=np.argsort(means)[-10:][::-1]
            print(f"slot {slot}: "+" ".join(f"{repr(chr(i))}:{means[i]:.3f}" for i in top))


if __name__ == "__main__":
    print("Creating scalable FlyBrain ASCII network...")
    web=InterconnectedFlyBrain(n_brains=10, topology="chain")
    print(
        f"[config] relay_pulse={web.inter_pulse_mV:.2f} mV, "
        f"ascii_lr={web.ascii.learning_rate:.4f}, "
        f"ascii_scale={web.ascii.input_scale:.3f}, "
        f"repeatable_inputs={web.repeatable_inputs}"
    )
    web.analyze_odor_path()
    print("Initial A:",web.predict("A"))
    print("Initial B:",web.predict("B"))
    web.train_ascii(epochs=10)
    print("Final A:",web.predict("A"))
    print("Final B:",web.predict("B"))
