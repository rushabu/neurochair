"""
SNNLayer: a synaptic (Linear) projection feeding a population of LIF neurons.

This is the SNN analogue of a standard nn.Linear + activation block:
  - Linear: weighted sum of inputs at each time step (synaptic current)
  - LIFNeuron: integrates that current over time and emits spikes

Input:  (batch, time, in_features)      -- e.g. spike-encoded audio features
Output: (batch, time, out_features)     -- spike train for this layer
"""

import torch
import torch.nn as nn

from .lif import LIFNeuron, LeakyReadout


class SNNLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 threshold: float = 1.0, leak: float = 0.9,
                 reset_mechanism: str = "hard", surrogate: str = "fast_sigmoid", recurrent: bool = False):
        super().__init__()
        self.synapse = nn.Linear(in_features, out_features)
        # optional spike feedback within the layer, so neurons carry context across time steps
        self.recurrent = nn.Linear(out_features, out_features, bias=False) if recurrent else None
        self.neuron = LIFNeuron(threshold=threshold, leak=leak,
                                 reset_mechanism=reset_mechanism, surrogate=surrogate)

    def forward(self, x: torch.Tensor):
        """x: (batch, time, in_features) -> spikes: (batch, time, out_features)"""
        current = self.synapse(x)  # applies the same Linear at every time step
        spikes, mem_trace = self.neuron(current, self.recurrent)
        return spikes, mem_trace


class ReadoutLayer(nn.Module):
    """
    Synapse (Linear) + non-spiking leaky integrator (LeakyReadout), used as the
    SNNClassifier's final layer instead of a second SNNLayer. See LeakyReadout's
    docstring in lif.py for why the readout specifically should not spike.
    """

    def __init__(self, in_features: int, out_features: int, leak: float = 0.9):
        super().__init__()
        self.synapse = nn.Linear(in_features, out_features)
        self.integrator = LeakyReadout(leak=leak)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, time, in_features) -> mem_trace: (batch, time, out_features), continuous"""
        current = self.synapse(x)
        return self.integrator(current)


def spike_raster(spikes_2d):
    """
    Build a simple (time, neuron) coordinate list for a raster plot from a
    single example's spike train.

    spikes_2d: (time, neurons) tensor of 0/1 spikes (already detached, on CPU)
    Returns: (times, neuron_ids) -- two 1D lists of the same length, one
             entry per spike, ready to scatter-plot.
    """
    times, neuron_ids = torch.nonzero(spikes_2d, as_tuple=True)
    return times.tolist(), neuron_ids.tolist()
