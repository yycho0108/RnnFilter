#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.jit as jit
import warnings
from collections import namedtuple
from typing import List, Tuple
from torch import Tensor
import numbers


def _update(m0, ls0, m1, ls1):
    """ gaussian update rule in log-std form """
    v0 = torch.exp2(2 * ls0)
    v1 = torch.exp2(2 * ls1)
    vsum = (v0 + v1)
    m = (m0 * v1 + m1 * v0) / (vsum + 1e-6)
    ls = ls0 + ls1 - 0.5 * torch.log2(vsum)
    return (m, ls)


class BayesGRUCell(jit.ScriptModule):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.weight_ih = Parameter(torch.randn(2 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(2 * hidden_size, hidden_size))
        self.bias = Parameter(torch.randn(2 * hidden_size))

        self.weight_ih2 = Parameter(torch.randn(1 * hidden_size, input_size))
        self.weight_hh2 = Parameter(torch.randn(1 * hidden_size, hidden_size))
        self.bias2 = Parameter(torch.randn(1 * hidden_size))

    @jit.script_method
    def forward(self, input: Tensor,
                state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        # state, log(sqrt(variance)) == log_std
        h0, ls0 = state

        gates = (
            torch.mm(
                input,
                self.weight_ih.t()) +
            torch.mm(
                h0,
                self.weight_hh.t()) +
            self.bias)

        r, ls1 = gates.chunk(2, 1)

        # exp2(`ls1`)**2 constrained to 0.25 ~ 4.0
        ls1 = torch.tanh(ls1)

        reset_gate = torch.sigmoid(r)

        h1 = torch.tanh(
            torch.mm(input, self.weight_ih2.t()) +
            torch.mm(reset_gate * h0, self.weight_hh2.t()) + self.bias2)

        # Multivariate gaussian update
        # This explicit form basically means
        # hidden state initialization basically does not matter
        # as long as v0 == large
        h2, ls2 = _update(h0, ls0, h1, ls1)

        return h2, (h2, ls2)


class BayesGRULayer(jit.ScriptModule):
    def __init__(self, cell: BayesGRUCell, batch_first: bool, *cell_args):
        super().__init__()
        self.axis = (1 if batch_first else 0)
        self.cell = cell(*cell_args)

    @jit.script_method
    def forward(self, inputs: Tensor,
                state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        # idk wtf this is doing
        inputs = inputs.unbind(self.axis)  # unbind == unstack
        outputs = jit.annotate(List[Tensor], [])
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs += [out]
        return torch.stack(outputs, dim=self.axis), state


class BayesGRU(jit.ScriptModule):
    __constant__ = ['layers']

    def __init__(self, input_size, hidden_size, num_layers, batch_first: bool):
        super().__init__()
        self.layers = nn.ModuleList(
            [BayesGRULayer(BayesGRUCell, batch_first, input_size, hidden_size)] +
            [BayesGRULayer(BayesGRUCell, batch_first, hidden_size, hidden_size)
             for _ in range(num_layers - 1)])

    @jit.script_method
    def forward(self, input: Tensor,
                states: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        # states = (c,h) I guess BUT we will use it as
        # (hidden(GRU), variance(GRU))
        # output_states = jit.annotate(List[Tuple[Tensor, Tensor]], [])

        hs = jit.annotate(List[Tensor], [])
        lss = jit.annotate(List[Tensor], [])

        output = input
        i = 0
        for layer in self.layers:
            state = (states[0][i], states[1][i])
            output, (h, ls) = layer(output, state)
            hs.append(h)
            lss.append(ls)
            i += 1
        hs = torch.stack(hs, dim=0)
        lss = torch.stack(lss, dim=0)
        return output, (hs, lss)


def main():
    batch_size = 64
    sequence_length = 8
    latent_size = 16
    input_size = 4
    num_layers = 2
    output_size = 2

    model = BayesGRU(input_size, latent_size, num_layers, True)
    #model = nn.LSTM(input_size=input_size,
    #                hidden_size=latent_size,
    #                num_layers=num_layers,
    #                batch_first=True)

    dummy_x = torch.zeros(
        (batch_size,
         sequence_length,
         input_size),
        dtype=torch.float32)

    dummy_h = torch.zeros(
        (num_layers,
         batch_size,
         latent_size),
        dtype=torch.float32)

    dummy_ls = torch.zeros(
        (num_layers,
         batch_size,
         latent_size),
        dtype=torch.float32)

    dummy_y, dummy_state = model(dummy_x, (dummy_h, dummy_ls))

    print(dummy_x.shape)  # 4
    print(dummy_y.shape)  # 16

    pass


if __name__ == '__main__':
    main()
