"""
SNNClassifier: Input spikes -> SNN layer(s) (spiking) -> readout layer (non-spiking) -> logits.

Readout: the final layer is a LeakyReadout (continuous leaky integrator, no
threshold, no spiking -- see snn/lif.py's LeakyReadout docstring for why).
Its membrane potential is summed over time per output class and used
directly as the classification logit. During training we run cross-entropy
on these summed potentials, same as any ordinary classifier's logits --
only the hidden layers upstream actually spike.
"""

import torch
import torch.nn as nn

from .layer import SNNLayer, ReadoutLayer


class SNNClassifier(nn.Module):
    def __init__(self, in_features: int, hidden_sizes: list[int], n_classes: int,
                 threshold: float = 1.0, leak: float = 0.9,
                 reset_mechanism: str = "hard", surrogate: str = "fast_sigmoid", recurrent: bool = False):
        super().__init__()
        sizes = [in_features] + hidden_sizes
        self.hidden_layers = nn.ModuleList([
            SNNLayer(sizes[i], sizes[i + 1], threshold=threshold, leak=leak,
                     reset_mechanism=reset_mechanism, surrogate=surrogate, recurrent=recurrent)
            for i in range(len(sizes) - 1)
        ])
        # Readout is a non-spiking leaky integrator, NOT another SNNLayer -- see
        # ReadoutLayer/LeakyReadout for why the final layer specifically should
        # not have a hard spike threshold.
        self.readout = ReadoutLayer(sizes[-1], n_classes, leak=leak)

    def forward(self, x: torch.Tensor, return_spikes: bool = False):
        """
        x: (batch, time, in_features) spike train
        Returns:
          logits: (batch, n_classes) = readout membrane potential summed over time
          if return_spikes: also returns (hidden_layer_spikes, readout_mem_trace)
            hidden_layer_spikes: list of (batch, time, features) spike trains, one per
                                  hidden layer (readout is NOT included -- it never spikes)
            readout_mem_trace:   (batch, time, n_classes) continuous membrane potential,
                                  for inspection/plotting (see visualize_raster.py)
        """
        hidden_layer_spikes = []
        h = x
        for layer in self.hidden_layers:
            h, _ = layer(h)
            if return_spikes:
                hidden_layer_spikes.append(h)

        readout_mem_trace = self.readout(h)  # (batch, time, n_classes), continuous
        # Average (not sum) over time: the readout's membrane potential is
        # unbounded (unlike the old spike-count readout, which was naturally
        # capped at n_time_steps), so summing all 50+ steps together produces
        # logits with a much larger, less predictable magnitude than a standard
        # classifier's -- destabilizing cross-entropy and requiring a different
        # learning rate than everything else in this codebase was tuned for.
        # Averaging keeps the logit scale comparable to a single time step's
        # potential, which behaves like an ordinary Linear layer's output.
        logits = readout_mem_trace.mean(dim=1)  # (batch, n_classes)

        if return_spikes:
            return logits, hidden_layer_spikes, readout_mem_trace
        return logits

    @classmethod
    def from_state_dict(cls, state_dict: dict) -> "SNNClassifier":
        """Rebuild the model with the layer sizes and recurrence stored in a checkpoint."""
        hidden_sizes, i = [], 0
        while f"hidden_layers.{i}.synapse.weight" in state_dict:
            hidden_sizes.append(state_dict[f"hidden_layers.{i}.synapse.weight"].shape[0])
            i += 1
        model = cls(in_features=state_dict["hidden_layers.0.synapse.weight"].shape[1],
                    hidden_sizes=hidden_sizes,
                    n_classes=state_dict["readout.synapse.weight"].shape[0],
                    recurrent="hidden_layers.0.recurrent.weight" in state_dict)
        model.load_state_dict(state_dict)
        return model

    @torch.no_grad()
    def total_spikes(self, x: torch.Tensor) -> int:
        """Total number of spikes fired across all HIDDEN layers for a batch (proxy for
        activity/cost). The readout no longer spikes, so it doesn't contribute here --
        that's expected, not a bug: there's nothing to count at a continuous layer."""
        _, hidden_layer_spikes, _ = self.forward(x, return_spikes=True)
        return sum(s.sum().item() for s in hidden_layer_spikes)
