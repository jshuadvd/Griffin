# Description: This file contains the implementation of the Griffin model in PyTorch.
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        post_act_ln: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.post_act_ln = post_act_ln

        # Linear layers
        self.layers = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

        # Layer normalization
        if post_act_ln:
            self.layers.add_module("layer_norm", nn.LayerNorm(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class RGLRU(nn.Module):
    """
    Real-Gated Linear Recurrent Unit (RG-LRU) for 3D input tensors.
    """

    def __init__(self, dim, mult: int):
        super().__init__()
        self.dim = dim
        hidden_dim = dim * mult
        self.hidden_dim = hidden_dim
        self.c = 8  # Scalar-valued constant

        # Initialize weights
        self.Wa = nn.Parameter(torch.Tensor(hidden_dim, dim))
        self.Wx = nn.Parameter(torch.Tensor(hidden_dim, dim))
        self.ba = nn.Parameter(torch.Tensor(hidden_dim))
        self.bx = nn.Parameter(torch.Tensor(hidden_dim))
        self.Lambda = nn.Parameter(torch.Tensor(hidden_dim))  # Λ

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(
            self.Wa, mode="fan_in", nonlinearity="linear"
        )
        nn.init.kaiming_normal_(
            self.Wx, mode="fan_in", nonlinearity="linear"
        )
        nn.init.constant_(self.ba, 0)
        nn.init.constant_(self.bx, 0)
        # Initialize Λ such that a is between 0.9 and 0.999
        self.Lambda.data.uniform_(
            torch.logit(torch.tensor(0.9)),
            torch.logit(torch.tensor(0.999)),
        )

    def forward(self, x):
        """
        Forward pass for sequences.

        Parameters:
        - x (Tensor): Input tensor with shape (batch_size, sequence_length, dim)

        Returns:
        - y (Tensor): Output tensor with shape (batch_size, sequence_length, hidden_dim)
        """
        print(f"RGLRU forward input shape: {x.shape}")
        batch_size, sequence_length, _ = x.shape
        ht = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        y = []

        for t in range(sequence_length):
            xt = x[:, t, :]
            rt = torch.sigmoid(torch.matmul(xt, self.Wa) + self.ba)
            it = torch.sigmoid(torch.matmul(xt, self.Wx) + self.bx)
            a = torch.sigmoid(self.Lambda)
            at = a / self.c**rt
            ht = at * ht + ((1 - at**2) ** 0.5) * (it * xt)
            y.append(ht.unsqueeze(1))

        y = torch.cat(y, dim=1)
        print(f"RGLRU forward output shape: {y.shape}")
        return y


class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization (RMSNorm) module.

    Args:
        dim (int): The dimension of the input tensor.

    Attributes:
        scale (float): The scaling factor for the normalized output.
        g (nn.Parameter): The learnable parameter used for scaling.

    """

    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        """
        Forward pass of the RMSNorm module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized output tensor.

        """
        print(f"RMSNorm input shape: {x.shape}")
        return F.normalize(x, dim=-1) * self.scale * self.g


def output_head(x: Tensor, num_tokens: int, dim: int):
    """
    Applies a linear transformation followed by softmax activation to the input tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, dim).
        dim (int): Dimension of the input tensor.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, dim) after applying linear transformation and softmax activation.
    """
    x = RMSNorm(dim)(x)

    # Linear transformation
    x = nn.Linear(dim, num_tokens)(x)

    # Softmax
    return F.softmax(x, dim=-1)


class GriffinResidualBlock(nn.Module):
    """
    GriffinResidualBlock is a residual block used in the Griffin model.

    Args:
        dim (int): The input dimension.
        depth (int): The depth of the block.
        mlp_mult (int): The multiplier for the hidden dimension in the feedforward network.
        dropout (float, optional): The dropout rate. Defaults to 0.1.
        heads (int, optional): The number of attention heads. Defaults to 8.
        filter (int, optional): The filter size for the convolutional layer. Defaults to 4.

    Attributes:
        dim (int): The input dimension.
        depth (int): The depth of the block.
        mlp_mult (int): The multiplier for the hidden dimension in the feedforward network.
        dropout (float): The dropout rate.
        heads (int): The number of attention heads.
        filter (int): The filter size for the convolutional layer.
        norm (RMSNorm): The normalization layer.
        mlp (FeedForward): The feedforward network.

    """

    def __init__(
        self,
        dim: int,
        depth: int,
        mlp_mult: int,
        dropout: float = 0.1,
        heads: int = 8,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.mlp_mult = mlp_mult
        self.dropout = dropout
        self.heads = heads

        # Norm
        self.norm = RMSNorm(dim)

         # Define linear layers here
        self.linear_1_layer = nn.Linear(dim, dim)
        self.linear_2_layer = nn.Linear(dim, dim)

        # Feedforward
        self.mlp = FeedForward(
            dim,
            dim * mlp_mult,
            dropout=dropout,
            post_act_ln=True,
            *args,
            **kwargs,
        )

        # RG-LRU
        self.lru = RGLRU(
            dim,
            mult=4,
        )

    def forward(self, x: Tensor):
        """
        Forward pass of the GriffinResidualBlock.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.

        """
        print(f"Input shape: {x.shape}")
        print(f"GriffinResidualBlock forward input shape: {x.shape}")

        b, s, d = x.shape

        skip = x


        # Norm
        x = self.norm(x)
        print(f"After some_layer shape: {x.shape}")
        print(x.shape)

        # Temporal Mixing Block
        # linear_1, linear_2 = nn.Linear(d, d)(x), nn.Linear(d, d)(x)
         # Use the linear layers
        linear_1 = self.linear_1_layer(x)
        linear_2 = self.linear_2_layer(x)

        print(linear_1.shape, linear_2.shape)

        # Conv1d
        linear_1 = nn.Conv1d(
            in_channels=s,
            out_channels=s,
            kernel_size=3,
            padding=1,
        )(linear_1)
        print(linear_1.shape)

        # RG-LRU
        # linear_1 = self.lru(linear_1)

        # Gelu on linear 2
        linear_2 = nn.GELU()(linear_2)

        # Element wise multiplication to merge the paths
        x = linear_1 * linear_2
        print(x.shape)

        # skip
        x += skip

        # Skip2
        skip2 = x

        # Norm
        x = self.norm(x)

        # Feedforward
        x = self.mlp(x)
        print(f"RMSNorm output shape: {x.shape}")
        return x + skip2

class Griffin(nn.Module):
    """
    Griffin module for performing Griffin Residual Network operations.

    Args:
        dim (int): Dimension of the input tensor.
        num_tokens (int): Number of tokens in the vocabulary.
        seq_len (int): Maximum sequence length.
        depth (int, optional): Number of residual blocks in the network. Defaults to 8.
        mlp_mult (int, optional): Multiplier for the hidden dimension of the MLP layers. Defaults to 4.
        dropout (float, optional): Dropout probability. Defaults to 0.1.
        heads (int, optional): Number of attention heads. Defaults to 8.
        filter (int, optional): Filter size for the convolutional layers. Defaults to 4.

    Attributes:
        dim (int): Dimension of the input tensor.
        num_tokens (int): Number of tokens in the vocabulary.
        seq_len (int): Maximum sequence length.
        depth (int): Number of residual blocks in the network.
        mlp_mult (int): Multiplier for the hidden dimension of the MLP layers.
        dropout (float): Dropout probability.
        heads (int): Number of attention heads.
        filter (int): Filter size for the convolutional layers.
        max_seq_len (int): Maximum sequence length.
        layers (nn.ModuleList): List of GriffinResidualBlock layers.
        emb (nn.Embedding): Embedding layer.
        norm (RMSNorm): RMSNorm layer.

    Methods:
        forward(x: Tensor) -> Tensor:
            Perform the forward pass of the Griffin module.
        generate(inp: Tensor, generate_length: int) -> Tensor:
            Generate a sequence of tokens based on the input tensor.

    """

    def __init__(
        self,
        dim: int,
        num_tokens: int,
        seq_len: int,
        depth: int = 8,
        mlp_mult: int = 4,
        dropout: float = 0.1,
        heads: int = 8,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.num_tokens = num_tokens
        self.seq_len = seq_len
        self.depth = depth
        self.mlp_mult = mlp_mult
        self.dropout = dropout
        self.heads = heads
        self.max_seq_len = seq_len

        # Layers
        self.layers = nn.ModuleList()

        # Add layers
        self.layers.append(
            GriffinResidualBlock(
                dim,
                depth,
                mlp_mult,
                dropout,
                heads,
                *args,
                **kwargs,
            )
        )

        # Embedding layer
        self.emb = nn.Embedding(
            num_tokens,
            dim,
        )

        # Rmsnorm
        self.norm = RMSNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the Griffin module.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.

        """
        print(f"Griffin forward input shape: {x.shape}")
        # Embed the tokens
        x = self.emb(x)

        # Normilize
        x = self.norm(x)

        # Loop
        for layer in self.layers:
            x = layer(x) + x
        print(f"Griffin forward output shape: {x.shape}")
        return output_head(x, self.num_tokens, self.dim)

    def generate(self, inp, generate_length):
        # Move the input tensor to the same device as the model
        inp = inp.to(next(self.parameters()).device)

        # Initialize the generated sequence with the input tensor
        generated_seq = inp

        # Iterate generate_length times to generate the desired number of tokens
        for _ in range(generate_length):
            # Get the last token from the generated sequence
            last_token = generated_seq[:, -1:]

            # Embed the last token
            x = self.emb(last_token)

            # Normalize the embedded token
            x = self.norm(x)

            # Pass the token through the model layers
            for layer in self.layers:
                x = layer(x) + x

            # Apply the output head to get the predicted next token
            next_token_logits = output_head(x, self.num_tokens, self.dim)

            # Sample the next token from the predicted logits
            next_token = torch.multinomial(next_token_logits[:, -1, :], num_samples=1)

            # Append the predicted next token to the generated sequence
            generated_seq = torch.cat((generated_seq, next_token), dim=-1)

        return generated_seq
