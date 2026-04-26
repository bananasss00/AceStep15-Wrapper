import torch
from torch.optim.optimizer import Optimizer

class AdEMAMix(Optimizer):
    """
    Implements AdEMAMix algorithm.
    Paper: https://arxiv.org/abs/2409.03137
    """
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999, 0.9999),
        eps=1e-8,
        weight_decay=0,
        alpha=5.0,
        t_alpha_beta3=None,
        foreach=None,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 2: {betas[2]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= alpha:
            raise ValueError(f"Invalid alpha value: {alpha}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            alpha=alpha,
            t_alpha_beta3=t_alpha_beta3,
            foreach=foreach,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("foreach", None)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            exp_avg_slow = []
            state_steps = []

            beta1, beta2, beta3 = group["betas"]
            alpha = group["alpha"]
            t_alpha_beta3 = group["t_alpha_beta3"]

            for p in group["params"]:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError("AdEMAMix does not support sparse gradients")
                    grads.append(p.grad)

                    state = self.state[p]
                    if len(state) == 0:
                        state["step"] = torch.tensor(0.0)
                        state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state["exp_avg_slow"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state["exp_avg"])
                    exp_avg_sqs.append(state["exp_avg_sq"])
                    exp_avg_slow.append(state["exp_avg_slow"])
                    state_steps.append(state["step"])

            for i, param in enumerate(params_with_grad):
                grad = grads[i]
                exp_avg = exp_avgs[i]
                exp_avg_sq = exp_avg_sqs[i]
                exp_avg_slow_i = exp_avg_slow[i]
                step_t = state_steps[i]

                # update step
                step_t += 1
                step = step_t.item()

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                exp_avg_slow_i.mul_(beta3).add_(grad, alpha=1 - beta3)

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step

                denorm = (exp_avg_sq.sqrt() / (bias_correction2**0.5)).add_(group["eps"])
                
                # Scheduling alpha and beta3 (optional)
                if t_alpha_beta3 is not None and step > t_alpha_beta3:
                    current_alpha = min(step * alpha / t_alpha_beta3, alpha)
                    current_beta3 = min(
                        1 - (1 - beta3) * (t_alpha_beta3 / step), beta3
                    )
                    # Recalculate slow buffer if schedule is active (simplified)
                    update = (exp_avg / bias_correction1) + current_alpha * exp_avg_slow_i
                else:
                    update = (exp_avg / bias_correction1) + alpha * exp_avg_slow_i

                if group["weight_decay"] != 0:
                    update.add_(param, alpha=group["weight_decay"])

                param.addcdiv_(update, denorm, value=-group["lr"])

        return loss