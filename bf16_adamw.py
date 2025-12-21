import torch

class BF16AdamW:
    """
    Custom AdamW optimizer matching tt-train's implementation with bfloat16 optimizer state.
    This implements the exact same logic as tt-train/sources/ttml/optimizers/adamw.cpp
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.steps = 0
        
        # Initialize optimizer state as bfloat16 (matching tt-train)
        self.first_moment = [torch.zeros_like(p, dtype=torch.bfloat16) for p in self.params]
        self.second_moment = [torch.zeros_like(p, dtype=torch.bfloat16) for p in self.params]
        
        # Compatibility with PyTorch optimizer interface (for LR scheduling)
        self.param_groups = [{'lr': lr, 'params': self.params}]
    
    def zero_grad(self, set_to_none=False):
        for p in self.params:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.zero_()
    
    @torch.no_grad()
    def step(self):
        self.steps += 1
        lr = self.param_groups[0]['lr']  # Use LR from param_groups (for LR scheduling compatibility)
        
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            
            # Get gradient and cast to bf16 (matching tt-train)
            grad = p.grad.to(torch.bfloat16)
            
            # Decoupled weight decay (applied before momentum update, like tt-train)
            if self.weight_decay != 0.0:
                # weights -= weight_decay * lr * weights
                p.data.add_(p.data, alpha=-self.weight_decay * lr)
            
            # Update first moment: m = beta1 * m + (1 - beta1) * grad
            self.first_moment[i].mul_(self.beta1).add_(grad, alpha=1.0 - self.beta1)
            
            # Update second moment: v = beta2 * v + (1 - beta2) * grad^2
            self.second_moment[i].mul_(self.beta2).addcmul_(grad, grad, value=1.0 - self.beta2)
            
            # Bias correction
            bias_correction1 = 1.0 - self.beta1 ** self.steps
            bias_correction2 = 1.0 - self.beta2 ** self.steps
            
            # Compute update: -lr * m_hat / (sqrt(v_hat) + eps)
            m_hat = self.first_moment[i] / bias_correction1
            v_hat = self.second_moment[i] / bias_correction2
            
            # Apply update (cast back to param dtype for the final update)
            update = -lr * m_hat / (v_hat.sqrt() + self.eps)
            p.data.add_(update.to(p.dtype))
    
    def get_steps(self):
        return self.steps