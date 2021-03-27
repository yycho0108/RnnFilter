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


def _update(m0, v0, m1, v1):
    # NOTE(ycho): v0/v1 might be better specified in log form
    # i.e. log(std0), log(std1)
    # as seen in e.g. PPO
    vsum = v0 + v1
    m = (v1 * m0 + v0 * m1) / vsum
    v = (v0 * v1) / vsum


class BayesGRUCell(jit.ScriptModule):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.weight_ih = Parameter(torch.randn(3 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(3 * hidden_size, hidden_size))
        self.bias = Parameter(torch.randn(3 * hidden_size))

    @jit.script_method
    def forward(self, input: Tensor,
                state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        h0, v0 = state

        gates = (
            torch.mm(
                input,
                self.weight_ih.t()) +
            torch.mm(
                h0,
                self.weight_hh.t()) +
            self.bias)

        r, v1 = gates.chunk(3, 1)
        h1 = torch.tanh(

        ig = torch.sigmoid(ig)
        fg = torch.sigmoid(fg)
        cg = torch.tanh(cg)
        og = torch.sigmoid(og)

        # 
        cy = (fg*cx) + (ig*cg) # "cell state"

        hy = (og * torch.tanh(cy)) # "output"

        # Multivariate gaussian update
        # This explicit form basically means
        # hidden state initialization basically does not matter
        # as long as v0 == large
        h2, v2 = _update(h0, v0, h1, v1)

        return (h2, v2)


class BayesGRULayer(jit.ScriptModule):
    def __init__(self, cell, *cell_args):
        super().__init__()
        self.cell = cell(*cell_args)

    @jit.script_method
    def forward(self, input: Tensor,
                state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        inputs = inputs.unbind(0)  # idk wtf this is doing
        outputs = jit.annotate(List[Tensor], [])
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs += [out]
        return torch.stack(outputs), state


class BayesGRU(jit.ScriptModule):
    __constant__ = ['layers']

    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([BayesGRULayer()] +
                                    [BayesGRULayer()
                                     for _ in range(num_layers - 1)])

    @jit.script_method
    def forward(self, input: Tensor,
                states: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]:
        # states = (c,h) I guess BUT we will use it as
        # (hidden(GRU), variance(GRU))
        output_states = jit.annotate(List[Tuple[Tensor, Tensor]], [])

        output = input
        i = 0
        for layer in self.layers:
            state = states[i]
            output, out_state = rnn_layer(output, state)
            output_states += [out_state]
            i += 1
        return output, output_states


def main():
    pass


if __name__ == '__main__':
    main()
