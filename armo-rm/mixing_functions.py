import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import hessian


class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        hidden_dim: int = 1024,
        n_hidden: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.dropout_prob = dropout
        layers = []
        for _ in range(n_hidden):
            layers.append(nn.Linear(in_features, hidden_dim))
            in_features = hidden_dim
        layers.append(nn.Linear(in_features, out_features, bias=bias))
        self.layers = nn.ModuleList(layers)
        
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        # Apply the linear layers with ReLU and dropout
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                if self.dropout_prob > 0:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)
        return x


class LinearNetwork(nn.Module):
    """
    Context-independent linear combination of reward objectives
    
    output = sum_i (r_i * w_i) such that sum_i w_i = 1 and w_i >= 0
    """
    def __init__(self, n_objectives: int):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(n_objectives))
        
    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())
        
    def forward(self, x: torch.FloatTensor, r: torch.FloatTensor) -> torch.FloatTensor:
        normalized_weights = F.softmax(self.weights, dim=0)
        return torch.sum(r * normalized_weights, dim=-1)


class GatingNetwork(nn.Module):
    """
    Gating Network: A simple MLP with softmax output and temperature scaling
    This network learns to combine multiple reward objectives based on the input context
    
    output = sum_i (r_i * w_i(x)) such that sum_i w_i(x) = 1 and w_i(x) >= 0
    """
    def __init__(
        self,
        context_size: int,
        n_objectives: int,
        bias: bool = True,
        hidden_dim: int = 1024,
        n_hidden: int = 3,
        dropout: float = 0.2,
        temperature: float = 10,
    ):
        super().__init__()
        self.temperature = temperature
        self.mlp = MLP(context_size, n_objectives, bias=bias, hidden_dim=hidden_dim, n_hidden=n_hidden, dropout=dropout)

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def gating_weights(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = self.mlp(x)
        # Apply softmax with temperature scaling
        x = F.softmax(x / self.temperature, dim=1)
        return x
    
    def forward(self, x: torch.FloatTensor, r: torch.FloatTensor) -> torch.FloatTensor:
        """
        Calculates context-dependent weights for linear combination of rewards
        """
        gating_weights = self.gating_weights(x)
        return torch.sum(r * gating_weights, dim=-1)
    
    
class RewardScalingNetwork(nn.Module):
    """
    Implements a non-linear scaling function for reward components (used in AdditivelySeparableNetwork)
    
    output = sigmoid(MLP([x, r_i]))
    """
    def __init__(
        self, 
        context_size: int,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        dropout: float = 0.2,
        
    ):
        super().__init__()
        self.mlp = MLP(context_size + 1, 1, hidden_dim=hidden_dim, n_hidden=n_hidden, dropout=dropout)

    def forward(self, x: torch.FloatTensor, r_i: torch.FloatTensor) -> torch.FloatTensor:
        x = torch.cat([x, r_i], dim=-1)
        x = self.mlp(x)
        return torch.sigmoid(x)
    

class AdditivelySeparableNetwork(nn.Module):
    """
    Additively Separable Mixing Function
    This network learns an additively separable function of reward components, where each component is a function of the input context and reward component.
    To ensure correct output normalization:
    - Each reward component is scaled by a sigmoid function of the input context
    - The output is the weighted sum of the scaled reward components, where the weights sum to 1 and are context-dependent as in the gating network
    
    output = sum_i [g_i(x) * sigmoid(f_i(r_i, x))]
    
    1. Learn gating network g(x) to compute context-dependent weights
    2. Learn reward scaling network sigmoid(f_i(r_i, x)) to compute scaled reward components
        a. inputs (x,r) -> break into components (x,r_i)
        b. apply separate reward scaling network to each component: r'_i = sigmoid(f_i(r_i, x))
        c. output is a vector of scaled reward components r' = [r'_1, r'_2, ..., r'_n]
    3. Combine the scaled reward components using the gating network weights, ensuring normalization and additive separability
    """
    def __init__(
        self,
        context_size: int,
        n_objectives: int,
        
        # Gating network parameters
        g_bias: bool = True,
        g_hidden_dim: int = 1024,
        g_n_hidden: int = 3,
        g_dropout: float = 0.2,
        g_temperature: float = 10,
        
        # Reward scaling network parameters
        r_hidden_dim: int = 256,
        r_n_hidden: int = 2,
        r_dropout: float = 0.2,
    ):
        super().__init__()
        self.gating_network = GatingNetwork(
            context_size, n_objectives, bias=g_bias, hidden_dim=g_hidden_dim, n_hidden=g_n_hidden, dropout=g_dropout, temperature=g_temperature
        )
        self.reward_scaling_networks = nn.ModuleList(
            [RewardScalingNetwork(context_size, r_hidden_dim, r_n_hidden, r_dropout) for _ in range(n_objectives)]
        )
        
    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())
        
        
    def forward(self, x: torch.FloatTensor, r: torch.FloatTensor) -> torch.FloatTensor:
        """
        Calculate the additively separable reward function
        """
        r_split = torch.split(r, 1, dim=1) # Split the reward components
        scaled_rewards = torch.cat([reward_scaling_network(x, r_i) for reward_scaling_network, r_i in zip(self.reward_scaling_networks, r_split)], dim=1)
        return self.gating_network(x, scaled_rewards)
    
    
class GeneralNetwork(nn.Module):
    """
    Defines an arbitrary mixing function 
    
    output = sigmoid(MLP([x, r]))
    """
    def __init__(
        self,
        context_size: int,
        n_objectives: int,
        hidden_dim: int = 1024,
        n_hidden: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.mlp = MLP(context_size + n_objectives, 1, hidden_dim=hidden_dim, n_hidden=n_hidden, dropout=dropout)
        
    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())
        
    def forward(self, x: torch.FloatTensor, r: torch.FloatTensor) -> torch.FloatTensor:
        x = torch.cat([x, r], dim=-1)
        return torch.sigmoid(self.mlp(x).squeeze(-1))
    
    
def compute_avg_hessian(model, X, R):
    """
    Compute the average Hessian of the model w.r.t. the reward components
    """
    avg_hessian = torch.zeros(n_objectives, n_objectives)
    for data in zip(X, R):
        x, r = data
        x = x.unsqueeze(0)
        r = r.unsqueeze(0)

        # Compute the Hessian w.r.t. r0
        hess = hessian(model, argnums=1)(x, r).squeeze(3).squeeze(1).squeeze(0)
        avg_hessian += hess
    avg_hessian /= n_samples
    return avg_hessian


def compute_separability(hess):
    """
    Compute separability via the off-diagonal norm
    """
    hess_diag = torch.diag(hess)
    hess_off_diag = hess - torch.diag(hess_diag)
    hess_norm = torch.norm(hess, p="fro")
    hess_off_norm = torch.norm(hess_off_diag, p="fro")
    
    if hess_off_norm == 0:
        return 1.0
    else:
        return 1 - hess_off_norm / hess_norm
    

if __name__ == "__main__":
    # Test the AdditivelySeparableNetwork
    n_objectives = 5
    in_features = 128
    n_samples = 5
    X = torch.randn(n_samples, in_features)
    R = torch.rand(n_samples, n_objectives)
    
    # Test the different networks
    models = [
        LinearNetwork(n_objectives),
        GatingNetwork(in_features, n_objectives),
        AdditivelySeparableNetwork(in_features, n_objectives),
        GeneralNetwork(in_features, n_objectives),
    ]
    
    
    for model in models:
        model.eval()
        print(f"\nTesting {model.__class__.__name__}")
        print(f"Number of parameters: {model.num_parameters}")
        output = model(X, R)
        print(output)
        assert output.shape == (n_samples,)
        assert output.dtype == torch.float32
        print("Test passed!")

        # Test the Hessian computation
        avg_hessian = compute_avg_hessian(model, X, R)
        print(avg_hessian)
        
        # Separability measures
        s = compute_separability(avg_hessian)
        print(f"Separability: {s}")
    
