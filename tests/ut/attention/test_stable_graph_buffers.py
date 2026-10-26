# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for the fixed-address FIA length buffers.

These exercise the buffer arithmetic on CPU, so they need no NPU: the point is
that the buffers keep their address while the values move, and that the padding
rows keep the values the FIA TND layout expects.
"""

import torch

from vllm_ascend.attention.attention_v1 import _StableGraphBuffers


def test_buffers_keep_their_address_across_fills():
    buffers = _StableGraphBuffers(torch.device("cpu"), capacity=4, max_blocks=8)
    q_ptr, kv_ptr, bt_ptr = buffers.q_lens.data_ptr(), buffers.kv_lens.data_ptr(), buffers.block_table.data_ptr()

    block_table = torch.arange(8, dtype=torch.int32).reshape(1, 8)
    first = buffers.fill([4], [16], block_table)
    second = buffers.fill([8], [32], block_table * 2)

    assert first[0].data_ptr() == second[0].data_ptr() == q_ptr
    assert first[1].data_ptr() == second[1].data_ptr() == kv_ptr
    assert first[2].data_ptr() == second[2].data_ptr() == bt_ptr


def test_values_are_written_in_place():
    buffers = _StableGraphBuffers(torch.device("cpu"), capacity=4, max_blocks=8)
    block_table = torch.full((2, 8), 7, dtype=torch.int32)
    q_lens, kv_lens, bt = buffers.fill([4, 4], [16, 17], block_table)

    assert q_lens.tolist() == [4, 4]
    assert kv_lens.tolist() == [16, 17]
    assert bt.shape == (2, 8)
    assert bt.tolist() == [[7] * 8, [7] * 8]


def test_padding_rows_keep_dummy_values():
    buffers = _StableGraphBuffers(torch.device("cpu"), capacity=4, max_blocks=8)
    # One real request, one synthetic padding request appended by build().
    q_lens, kv_lens, bt = buffers.fill([4, 4], [16], torch.ones((1, 8), dtype=torch.int32))

    assert q_lens.tolist() == [4, 4]
    # Padding row keeps the "any positive KV length" value written at init.
    assert kv_lens.tolist() == [16, 1]
    # Padding row keeps pointing at block 0.
    assert bt[1].tolist() == [0] * 8


def test_overflow_falls_back_to_the_update_path():
    buffers = _StableGraphBuffers(torch.device("cpu"), capacity=2, max_blocks=8)
    assert buffers.fill([4, 4, 4], [16, 16, 16], torch.zeros((3, 8), dtype=torch.int32)) is None


def test_block_table_absent_is_tolerated():
    buffers = _StableGraphBuffers(torch.device("cpu"), capacity=4, max_blocks=8)
    q_lens, kv_lens, bt = buffers.fill([4], [16], None)

    assert q_lens.tolist() == [4]
    assert kv_lens.tolist() == [16]
    assert bt.tolist() == [[0] * 8]
