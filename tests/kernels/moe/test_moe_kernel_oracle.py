# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the MoEKernelOracle ABC introduced in PR series for #37753.

This file contains a single canonical demonstration that
`UnquantizedMoEKernelOracle` methods delegate one-to-one to the
existing module-level functions in `oracle/unquantized.py`. Each method
on `UnquantizedMoEKernelOracle` follows the same `return module_fn(args)`
pattern, so verifying delegation for one method (`make_kernel`) gives
high confidence in the rest.
"""

from dataclasses import replace
from unittest.mock import patch

import pytest

from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
from vllm.model_executor.layers.fused_moe.oracle import UnquantizedMoEKernelOracle
from vllm.model_executor.layers.fused_moe.oracle import fp8 as fp8_oracle
from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
    UnquantizedMoeBackend,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Dynamic128Sym,
    kFp8Static128BlockSym,
)


class TestUnquantizedDelegation:
    """UnquantizedMoEKernelOracle methods must delegate to the existing
    module-level functions; behaviour is bit-identical."""

    def test_make_kernel_delegates(self) -> None:
        quant_config = object()
        moe_config = object()
        experts_cls = TritonExperts
        sentinel_kernel = object()

        with patch(
            "vllm.model_executor.layers.fused_moe.oracle.unquantized."
            "make_unquantized_moe_kernel",
            return_value=sentinel_kernel,
        ) as mocked:
            out = UnquantizedMoEKernelOracle().make_kernel(
                quant_config,
                moe_config,
                UnquantizedMoeBackend.TRITON,
                experts_cls,
            )

        mocked.assert_called_once_with(
            quant_config,
            moe_config,
            UnquantizedMoeBackend.TRITON,
            experts_cls,
            None,  # routing_tables default
        )
        assert out is sentinel_kernel


class _AlwaysSupported:
    @staticmethod
    def is_supported_config(*args, **kwargs):
        return True, None


@pytest.mark.parametrize(
    ("excluded", "moe_backend", "env", "expected"),
    [
        (False, "auto", None, "DEEPGEMM"),
        (True, "auto", None, "TRITON"),
        (True, "auto", "1", "TRITON"),
        (True, "deep_gemm", None, "DEEPGEMM"),
    ],
)
def test_fp8_oracle_deep_gemm_model_exclusion(
    monkeypatch: pytest.MonkeyPatch,
    excluded: bool,
    moe_backend: str,
    env: str | None,
    expected: str,
) -> None:
    Backend = fp8_oracle.Fp8MoeBackend
    monkeypatch.setattr(fp8_oracle, "deep_gemm_disabled_for_model", lambda: excluded)
    monkeypatch.setattr(
        fp8_oracle,
        "_get_priority_backends",
        lambda *a, **k: [
            Backend.DEEPGEMM,
            Backend.VLLM_CUTLASS,
            Backend.TRITON,
            Backend.BATCHED_DEEPGEMM,
            Backend.BATCHED_VLLM_CUTLASS,
        ],
    )
    monkeypatch.setattr(
        fp8_oracle, "backend_to_kernel_cls", lambda backend: [_AlwaysSupported]
    )
    monkeypatch.delenv("VLLM_USE_DEEP_GEMM", raising=False)
    monkeypatch.delenv("VLLM_MOE_USE_DEEP_GEMM", raising=False)
    if env is not None:
        monkeypatch.setenv("VLLM_USE_DEEP_GEMM", env)

    backend, _ = fp8_oracle.select_fp8_moe_backend(
        config=replace(make_dummy_moe_config(), moe_backend=moe_backend),
        weight_key=kFp8Static128BlockSym,
        activation_key=kFp8Dynamic128Sym,
    )
    assert backend == Backend[expected]
