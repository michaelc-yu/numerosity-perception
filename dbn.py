

import torch
import torch.nn as nn

class RBM(nn.Module):
    def __init__(self, n_visible, n_hidden, k=1):
        super().__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.k = k  # CD-k steps (usually k=1)

        # Weight matrix (visible → hidden)
        self.W = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.01)
        
        # Biases
        self.v_bias = nn.Parameter(torch.zeros(n_visible))
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))

    def sample_h(self, v):
        """Sample hidden units given visible units."""
        prob = torch.sigmoid(v @ self.W + self.h_bias)
        return prob, torch.bernoulli(prob)

    def sample_v(self, h):
        """Sample visible units given hidden units."""
        prob = torch.sigmoid(h @ self.W.t() + self.v_bias)
        return prob, torch.bernoulli(prob)

    def forward(self, v):
        """One CD-k step and weight update."""
        v0 = v.detach()

        # Positive phase
        ph0, h0 = self.sample_h(v0)

        # Gibbs sampling (CD-k)
        vk, hk = v0, h0
        for _ in range(self.k):
            vk_prob, vk = self.sample_v(hk)
            hk_prob, hk = self.sample_h(vk)

        # Negative phase
        phk, _ = self.sample_h(vk)

        # Compute gradients
        positive_grad = v0.t() @ ph0
        negative_grad = vk.t() @ phk

        # Update parameters manually
        self.W.grad = -(positive_grad - negative_grad) / v.size(0)
        self.v_bias.grad = -(v0 - vk).mean(0)
        self.h_bias.grad = -(ph0 - phk).mean(0)

        return vk  # reconstruction (not needed but useful)

    def reconstruct(self, v):
        """Get reconstruction without updating weights."""
        _, h = self.sample_h(v)
        v_prob, _ = self.sample_v(h)
        return v_prob


def train_rbm(rbm, data, lr=1e-3, epochs=5, batch_size=64):
    optimizer = torch.optim.SGD([rbm.W, rbm.v_bias, rbm.h_bias], lr=lr)

    for epoch in range(epochs):
        perm = torch.randperm(data.size(0))
        for i in range(0, data.size(0), batch_size):
            batch = data[perm[i:i+batch_size]]
            optimizer.zero_grad()
            rbm(batch)  # CD-1 forward pass computes grads
            optimizer.step()

        print(f"Epoch {epoch+1} done")




