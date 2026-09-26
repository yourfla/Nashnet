import torch
import torch.nn as nn
import torch.nn.functional as F

class NashEquilibrium(nn.Module):

    def __init__(self, n_experts=6, max_nash_iterations=20, convergence_threshold=0.001, diversity_weight=0.3):
        super().__init__()
        self.n_experts = n_experts
        self.max_nash_iterations = max_nash_iterations
        self.convergence_threshold = convergence_threshold
        self.diversity_weight = diversity_weight
        self.best_response_net = nn.ModuleList([nn.Sequential(nn.Linear(n_experts - 1 + 3, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid()) for _ in range(n_experts)])
        self.evidence_encoder = nn.Sequential(nn.Linear(n_experts, 8), nn.ReLU(), nn.Dropout(0.2), nn.Linear(8, n_experts), nn.Softplus())
        self.spatial_quality_encoder = nn.Sequential(nn.Linear(3, 6), nn.ReLU(), nn.Dropout(0.2), nn.Linear(6, n_experts), nn.Softplus())
        self.expert_prior_logits = nn.Parameter(torch.zeros(n_experts))
        self.register_buffer('expert_performance', torch.ones(n_experts) * 0.5)
        self.register_buffer('update_count', torch.zeros(1))

    def compute_best_response(self, expert_idx, other_weights, performance_metrics):
        br_input = torch.cat([other_weights, performance_metrics.unsqueeze(0).expand(other_weights.size(0), -1)], dim=1)
        best_weight = self.best_response_net[expert_idx](br_input)
        return best_weight

    def check_nash_equilibrium(self, weights, performance_metrics, threshold=0.001):
        batch_size = weights.size(0)
        nash_violations = []
        for i in range(self.n_experts):
            other_indices = [j for j in range(self.n_experts) if j != i]
            other_weights = weights[:, other_indices]
            best_weight = self.compute_best_response(i, other_weights, performance_metrics)
            current_weight = weights[:, i:i + 1]
            violation = torch.abs(best_weight - current_weight).mean()
            nash_violations.append(violation)
        avg_violation = torch.stack(nash_violations).mean()
        is_equilibrium = avg_violation < threshold
        return (is_equilibrium, avg_violation)

    def iterative_nash_solver(self, batch_size, device, temperature, performance_metrics, base_logits=None, max_iterations=20):
        if base_logits is not None:
            weights = F.softmax(base_logits / max(float(temperature), 0.001), dim=0)
            weights = weights.unsqueeze(0).expand(batch_size, -1).contiguous()
            weights = weights + torch.randn_like(weights) * 0.01
            weights = torch.clamp(weights, min=0.0001)
        else:
            weights = torch.ones(batch_size, self.n_experts, device=device) / self.n_experts
            weights = weights + torch.randn_like(weights) * 0.01
        weights = weights / weights.sum(dim=1, keepdim=True)
        nash_achieved = False
        for iteration in range(max_iterations):
            new_weights = []
            for i in range(self.n_experts):
                other_indices = [j for j in range(self.n_experts) if j != i]
                other_weights = weights[:, other_indices]
                best_weight = self.compute_best_response(i, other_weights, performance_metrics)
                new_weights.append(best_weight)
            new_weights = torch.cat(new_weights, dim=1)
            momentum = 0.7 - 0.05 * iteration
            weights = momentum * weights + (1 - momentum) * new_weights
            weights = weights / weights.sum(dim=1, keepdim=True)
            logit = torch.log(weights + 1e-08)
            if base_logits is not None:
                logit = logit + base_logits.unsqueeze(0)
            weights = F.softmax(logit / temperature, dim=1)
            nash_achieved, violation = self.check_nash_equilibrium(weights, performance_metrics, threshold=self.convergence_threshold)
            if nash_achieved:
                break
        return (weights, nash_achieved, iteration + 1)

    def compute_diversity_bonus(self, weights):
        entropy = -(weights * torch.log(weights + 1e-08)).sum(dim=1).mean()
        max_entropy = torch.log(torch.tensor(self.n_experts, dtype=torch.float32, device=weights.device))
        normalized_entropy = entropy / max_entropy
        return normalized_entropy

    def compute_nash_equilibrium(self, batch_size, device='cuda', temperature=1.0, use_performance=True, evidence_prior=None, spatial_quality_vector=None, collective_loss=0.5, individual_losses=None, expert_iou=None, expert_dice=None):
        diversity_factor = 0.05
        if individual_losses is None:
            individual_losses = torch.ones(self.n_experts, device=device) * collective_loss
        avg_individual_loss = individual_losses.mean()
        if spatial_quality_vector is not None:
            spatial_quality = spatial_quality_vector.mean()
        else:
            spatial_quality = torch.tensor(0.5, device=device)
        performance_metrics = torch.tensor([collective_loss, avg_individual_loss, spatial_quality.item() if isinstance(spatial_quality, torch.Tensor) else spatial_quality], device=device)
        base_logits = self.expert_prior_logits.clone()
        if evidence_prior is not None:
            evidence_normalized = evidence_prior / (evidence_prior.sum() + 1e-08)
            evidence_modulation = self.evidence_encoder(evidence_normalized.unsqueeze(0)).squeeze(0)
            evidence_weight = 0.3 * min(1.0, self.update_count.item() / 500)
            base_logits = base_logits + evidence_weight * evidence_modulation
        if spatial_quality_vector is not None:
            spatial_modulation = self.spatial_quality_encoder(spatial_quality_vector.unsqueeze(0)).squeeze(0)
            spatial_weight = 0.2 * min(1.0, self.update_count.item() / 300)
            base_logits = base_logits + spatial_weight * spatial_modulation
        if use_performance and self.update_count.item() > 20:
            performance_logits = torch.log(self.expert_performance + 1e-08)
            performance_weight = 0.4
            base_logits = base_logits + performance_weight * performance_logits
        nash_weights, nash_achieved, iterations = self.iterative_nash_solver(batch_size, device, temperature, performance_metrics, base_logits=base_logits, max_iterations=self.max_nash_iterations)
        diversity = self.compute_diversity_bonus(nash_weights)
        uniform_weights = torch.ones_like(nash_weights) / self.n_experts
        if self.training:
            if self.update_count.item() < 200:
                diversity_factor = 0.05
            elif self.update_count.item() < 600:
                diversity_factor = 0.05 + 0.05 * (1 - diversity)
            else:
                diversity_factor = 0 * (1 - diversity)
        final_weights = (1 - diversity_factor) * nash_weights + diversity_factor * uniform_weights
        final_weights = final_weights / final_weights.sum(dim=1, keepdim=True)
        if self.training:
            noise_scale = 0.002 * temperature * max(0, 1 - self.update_count.item() / 500)
            noise = torch.randn_like(final_weights) * noise_scale
            final_weights = F.softmax((torch.log(final_weights + 1e-08) + noise) / temperature, dim=1)
        return final_weights
