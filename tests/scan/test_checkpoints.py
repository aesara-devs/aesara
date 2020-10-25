import numpy as np
import pytest

import theano
import theano.tensor as tt
from theano.scan.basic import scan
from theano.scan.checkpoints import scan_checkpoints


class TestScanCheckpoint:
    def setup_method(self):
        self.k = tt.iscalar("k")
        self.A = tt.vector("A")
        result, _ = scan(
            fn=lambda prior_result, A: prior_result * A,
            outputs_info=tt.ones_like(self.A),
            non_sequences=self.A,
            n_steps=self.k,
        )
        result_check, _ = scan_checkpoints(
            fn=lambda prior_result, A: prior_result * A,
            outputs_info=tt.ones_like(self.A),
            non_sequences=self.A,
            n_steps=self.k,
            save_every_N=100,
        )
        self.result = result[-1]
        self.result_check = result_check[-1]
        self.grad_A = tt.grad(self.result.sum(), self.A)
        self.grad_A_check = tt.grad(self.result_check.sum(), self.A)

    def test_forward_pass(self):
        # Test forward computation of A**k.
        f = theano.function(
            inputs=[self.A, self.k], outputs=[self.result, self.result_check]
        )
        out, out_check = f(range(10), 101)
        assert np.allclose(out, out_check)

    def test_backward_pass(self):
        # Test gradient computation of A**k.
        f = theano.function(
            inputs=[self.A, self.k], outputs=[self.grad_A, self.grad_A_check]
        )
        out, out_check = f(range(10), 101)
        assert np.allclose(out, out_check)

    def test_taps_error(self):
        # Test that an error rises if we use taps in outputs_info.
        with pytest.raises(RuntimeError):
            scan_checkpoints(lambda: None, [], {"initial": self.A, "taps": [-2]})
