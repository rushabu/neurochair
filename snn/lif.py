"""
Leaky Integrate-and-Fire (LIF) neuron, implemented from scratch.

Dynamics (discrete-time, one step):
    mem[t] = leak * mem[t-1] + input[t]        # integrate + leak
    spike[t] = 1 if mem[t] >= threshold else 0  # fire
    mem[t] = mem[t] * (1 - spike[t])            # hard reset to 0 on spike

  - leak in (0, 1]: how much of the previous membrane potential survives
    each step. leak=1 -> pure integrator (no decay). leak close to 0 ->
    forgets almost everything each step.
  - threshold: voltage the membrane must cross to fire.
  - reset: "hard" zeroes the membrane on spike (implemented here);
    "subtract" would instead subtract `threshold` (left as an option).

No snnTorch is used -- this is a plain nn.Module with a Python/torch loop
over time steps, using the surrogate-gradient autograd Functions in
surrogate.py for the backward pass through the spike nonlinearity.
"""

import torch
import torch.nn as nn

from .surrogate import get_surrogate


class LIFNeuron(nn.Module):
    def __init__(self, threshold: float = 1.0, leak: float = 0.9,
                 reset_mechanism: str = "hard", surrogate: str = "fast_sigmoid"):
        super().__init__()
        assert 0.0 < leak <= 1.0, "leak must be in (0, 1]"
        assert reset_mechanism in ("hard", "subtract")
        self.threshold = threshold
        self.leak = leak
        self.reset_mechanism = reset_mechanism
        self.spike_fn = get_surrogate(surrogate)

    def forward(self, input_current: torch.Tensor, recurrent=None):
        """
        input_current: (batch, time, features) -- pre-synaptic current at
        each time step (e.g. output of a Linear/Conv layer applied per step).
        recurrent: optional Linear(features, features) fed with the previous step's spikes.

        Returns:
            spikes: (batch, time, features) -- 0/1 spike train
            mem_trace: (batch, time, features) -- membrane potential trace (for inspection)
        """
        batch, time_steps, features = input_current.shape
        device = input_current.device

        mem = torch.zeros(batch, features, device=device)
        spike = torch.zeros(batch, features, device=device)
        spikes = []
        mem_trace = []

        for t in range(time_steps):
            current = input_current[:, t, :]
            if recurrent is not None:
                current = current + recurrent(spike)
            mem = self.leak * mem + current
            spike = self.spike_fn(mem - self.threshold)

            if self.reset_mechanism == "hard":
                mem = mem * (1.0 - spike)
            else:  # "subtract"
                mem = mem - spike * self.threshold

            spikes.append(spike)
            mem_trace.append(mem)

        spikes = torch.stack(spikes, dim=1)
        mem_trace = torch.stack(mem_trace, dim=1)
        return spikes, mem_trace


class LeakyReadout(nn.Module):
    """
    Non-spiking leaky integrator, used as the SNN's readout layer instead of a
    second LIFNeuron.

    Dynamics -- identical leaky integration to LIFNeuron, but with NO threshold
    and NO reset:
        mem[t] = leak * mem[t-1] + input[t]

    This is standard practice for the final readout layer of a rate-coded SNN
    classifier (see e.g. Bellec et al. 2020's e-prop). Forcing the *output* layer
    through a hard spike threshold, same as the hidden layers, has two problems:
      1. It quantizes the classification signal into small integer spike counts,
         throwing away resolution the loss function could otherwise use.
      2. Early in training, weights are small and pre-threshold currents may never
         cross the threshold at all -- the readout goes completely silent, every
         class gets an identical (zero) logit, and cross-entropy has no useful
         gradient to push the weights toward the correct class. This produces
         exactly-uniform softmax output (every class at the same confidence) and
         can stall training indefinitely.

    Using the continuous membrane potential instead avoids both problems, while
    the hidden layers upstream still spike exactly as before -- this only changes
    the final layer.
    """

    def __init__(self, leak: float = 0.9):
        super().__init__()
        assert 0.0 < leak <= 1.0, "leak must be in (0, 1]"
        self.leak = leak

    def forward(self, input_current: torch.Tensor) -> torch.Tensor:
        """
        input_current: (batch, time, features)
        Returns: mem_trace (batch, time, features) -- continuous, never spikes
        """
        batch, time_steps, features = input_current.shape
        device = input_current.device

        mem = torch.zeros(batch, features, device=device)
        trace = []
        for t in range(time_steps):
            mem = self.leak * mem + input_current[:, t, :]
            trace.append(mem)

        return torch.stack(trace, dim=1)
