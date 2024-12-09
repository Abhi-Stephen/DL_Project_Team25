import torch
import torch.nn as nn
from torch.autograd import Variable


class ConvLSTMCell(nn.Module):
    """
    Generate a convolutional LSTM cell.
    """

    def __init__(self, in_channels, hidden_channels, kernel_size):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.Gates = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=kernel_size // 2,
        )
        self.prev_state = None
        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.Gates.weight)

    def reset(self):
        self.prev_state = None

    def forward(self, input_):
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        if self.prev_state is None:
            state_size = [batch_size, self.hidden_channels] + list(spatial_size)
            self.prev_state = (
                Variable(torch.zeros(state_size, device=input_.device)),
                Variable(torch.zeros(state_size, device=input_.device)),
            )

        prev_hidden, prev_cell = self.prev_state

        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)
        cell_gate = torch.tanh(cell_gate)

        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        self.prev_state = (hidden, cell)
        return hidden


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        n_convs=3,
        kernel_size=3,
        stride=1,
        padding=1,
        downsample=True,
        dilation=1,
    ):
        super(ConvBlock, self).__init__()
        self.modules_list = []
        c_in = in_channels
        c_out = out_channels
        for i in range(n_convs):
            self.modules_list.append(
                nn.Conv2d(
                    in_channels=c_in,
                    out_channels=c_out,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    bias=False,
                    dilation=dilation,
                )
            )
            self.modules_list.append(nn.BatchNorm2d(num_features=out_channels))
            self.modules_list.append(nn.LeakyReLU(0.1))
            c_in = c_out

        if downsample:
            self.modules_list.append(
                nn.Conv2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=2,
                    stride=2,
                )
            )
            self.modules_list.append(nn.ReLU())

        self.model = nn.Sequential(*self.modules_list)
        self.init_weights()

    def init_weights(self):
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        return self.model(x)
