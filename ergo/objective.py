"""Joint GAO/TAO AW update using the paper's diagonal curvature proxy."""

import torch


class AdaptiveWeights:
    def __init__(self, count, config, device):
        self.count = count
        self.config = config
        self.weights = torch.full((2 * count,), 1 / (2 * count), device=device)
        self.risks = torch.zeros_like(self.weights)

    def loss(self, geometry, texture, parameters, step):
        losses = list(geometry) + list(texture)
        if len(losses) != self.weights.numel():
            raise ValueError("Each reference needs one GAO and one TAO objective")
        if step % self.config.risk_update_interval == 0:
            risks = []
            for loss in losses:
                gradients = torch.autograd.grad(
                    loss, parameters, retain_graph=True, allow_unused=True
                )
                risk = loss.new_zeros(())
                for gradient in gradients:
                    if gradient is not None:
                        squared = gradient.detach().float().square()
                        risk += (
                            0.5
                            * (
                                squared
                                / (squared + self.config.hessian_damping).clamp_min(
                                    self.config.eps
                                )
                            ).sum()
                        )
                risks.append(risk)
            self.risks = torch.stack(risks).detach()
            if not torch.isfinite(self.risks).all():
                raise FloatingPointError("AW risk is not finite")
            with torch.no_grad():
                scores = self.risks / self.risks.sum().clamp_min(self.config.eps)
                self.weights = (
                    self.weights.clamp_min(self.config.eps).log()
                    + self.config.weight_sensitivity * scores
                ).softmax(0)
                g, t = self.weights.split(self.count)
                gm, tm = g.sum(), t.sum()
                if gm < self.config.min_geometry_weight_sum:
                    mass = self.config.min_geometry_weight_sum
                    self.weights = torch.cat(
                        [
                            g / gm.clamp_min(self.config.eps) * mass,
                            t / tm.clamp_min(self.config.eps) * (1 - mass),
                        ]
                    )
                elif tm < self.config.min_texture_weight_sum:
                    mass = self.config.min_texture_weight_sum
                    self.weights = torch.cat(
                        [
                            g / gm.clamp_min(self.config.eps) * (1 - mass),
                            t / tm.clamp_min(self.config.eps) * mass,
                        ]
                    )
        return (torch.stack(losses) * self.weights.detach()).sum()
