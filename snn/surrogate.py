"""
Surrogate gradients for the spiking (Heaviside) nonlinearity.

A spiking neuron fires with a hard threshold:
    spike = 1 if (membrane_potential - threshold) >= 0 else 0

This is the Heaviside step function. Its true derivative is a Dirac delta
(zero almost everywhere, infinite at the threshold) which is useless for
backprop. Surrogate gradients replace that derivative -- ONLY in the
backward pass -- with a smooth, well-behaved function that still points in
a sensible direction for gradient descent. The forward pass always emits a
real, hard 0/1 spike.

Three surrogates are implemented, selectable at model build time:
  - fast_sigmoid : d/dx sigmoid-like curve, cheap, widely used (Zenke & Ganguli 2018)
  - triangular   : piecewise-linear tent function around threshold
  - gaussian     : Gaussian bump centered on threshold (Bellec et al. 2020 use similar)

Each is implemented as a torch.autograd.Function so we control forward AND
backward explicitly instead of relying on autograd to differentiate a
non-differentiable step.
"""

import torch


class FastSigmoidSpike(torch.autograd.Function):
    """Forward: hard spike. Backward: derivative of a fast sigmoid 1/(1+beta|x|)."""

    beta = 5.0  # steepness of the surrogate; higher = sharper approximation to the step

    @staticmethod
    def forward(ctx, membrane_minus_threshold):
        ctx.save_for_backward(membrane_minus_threshold)
        return (membrane_minus_threshold >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        beta = FastSigmoidSpike.beta
        # d/dx [ x / (1 + beta|x|) ] = 1 / (beta|x| + 1)^2
        surrogate_grad = 1.0 / (beta * x.abs() + 1.0) ** 2
        return grad_output * surrogate_grad


class TriangularSpike(torch.autograd.Function):
    """Forward: hard spike. Backward: triangular (tent) function around threshold."""

    width = 1.0  # half-width of the tent; gradient is 0 outside [-width, width]

    @staticmethod
    def forward(ctx, membrane_minus_threshold):
        ctx.save_for_backward(membrane_minus_threshold)
        return (membrane_minus_threshold >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        width = TriangularSpike.width
        surrogate_grad = torch.clamp(1.0 - x.abs() / width, min=0.0)
        return grad_output * surrogate_grad


class GaussianSpike(torch.autograd.Function):
    """Forward: hard spike. Backward: Gaussian bump centered at the threshold crossing."""

    sigma = 0.5  # standard deviation of the Gaussian surrogate

    @staticmethod
    def forward(ctx, membrane_minus_threshold):
        ctx.save_for_backward(membrane_minus_threshold)
        return (membrane_minus_threshold >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        sigma = GaussianSpike.sigma
        surrogate_grad = torch.exp(-0.5 * (x / sigma) ** 2) / (sigma * (2 * torch.pi) ** 0.5)
        return grad_output * surrogate_grad


SURROGATES = {
    "fast_sigmoid": FastSigmoidSpike,
    "triangular": TriangularSpike,
    "gaussian": GaussianSpike,
}


def get_surrogate(name: str):
    if name not in SURROGATES:
        raise ValueError(f"Unknown surrogate '{name}'. Choose from {list(SURROGATES)}")
    return SURROGATES[name].apply
