import numpy as np
import pytest

import theano
import theano.sandbox.rng_mrg
from tests import unittest_tools as utt
from tests.gpuarray.config import mode_with_gpu, test_ctx_name
from theano import gpuarray, tensor
from theano.gpuarray.basic_ops import GpuFromHost, HostFromGpu
from theano.gpuarray.elemwise import GpuElemwise
from theano.scan.basic import scan
from theano.scan.checkpoints import scan_checkpoints
from theano.scan.op import Scan


pygpu_gpuarray = pytest.importorskip("pygpy.gpuarray")
GpuArrayException = pygpu_gpuarray.GpuArrayException


if theano.config.mode == "FAST_COMPILE":
    mode_with_opt = theano.compile.mode.get_mode("FAST_RUN")
else:
    mode_with_opt = theano.compile.mode.get_default_mode()
if theano.config.mode in ("DEBUG_MODE", "DebugMode"):
    mode_nodebug = theano.compile.mode.get_mode("FAST_RUN")
else:
    mode_nodebug = mode_with_opt


class TestScan:
    def setup_method(self):
        utt.seed_rng()

    def test_one_sequence_one_output_weights_gpu1(self):
        def f_rnn(u_t, x_tm1, W_in, W):
            return u_t * W_in + x_tm1 * W

        u = theano.tensor.fvector("u")
        x0 = theano.tensor.fscalar("x0")
        W_in = theano.tensor.fscalar("win")
        W = theano.tensor.fscalar("w")

        mode = mode_with_gpu.excluding("InputToGpuOptimizer")
        output, updates = scan(
            f_rnn,
            u,
            x0,
            [W_in, W],
            n_steps=None,
            truncate_gradient=-1,
            go_backwards=False,
            mode=mode,
        )

        output = GpuFromHost(test_ctx_name)(output)
        f2 = theano.function(
            [u, x0, W_in, W],
            output,
            updates=updates,
            allow_input_downcast=True,
            mode=mode,
        )

        rng = np.random.RandomState(utt.fetch_seed())
        v_u = rng.uniform(size=(4,), low=-5.0, high=5.0)
        v_x0 = rng.uniform()
        W = rng.uniform()
        W_in = rng.uniform()

        v_u = np.asarray(v_u, dtype="float32")
        v_x0 = np.asarray(v_x0, dtype="float32")
        W = np.asarray(W, dtype="float32")
        W_in = np.asarray(W_in, dtype="float32")

        # compute the output in numpy
        v_out = np.zeros((4,))
        v_out[0] = v_u[0] * W_in + v_x0 * W
        for step in range(1, 4):
            v_out[step] = v_u[step] * W_in + v_out[step - 1] * W

        theano_values = f2(v_u, v_x0, W_in, W)
        utt.assert_allclose(theano_values, v_out)

        # TO DEL
        topo = f2.maker.fgraph.toposort()
        scan_node = [node for node in topo if isinstance(node.op, scan.op.Scan)]
        assert len(scan_node) == 1
        scan_node = scan_node[0]

        topo = f2.maker.fgraph.toposort()
        assert sum([isinstance(node.op, HostFromGpu) for node in topo]) == 0
        assert sum([isinstance(node.op, GpuFromHost) for node in topo]) == 4

        scan_node = [node for node in topo if isinstance(node.op, scan.op.Scan)]
        assert len(scan_node) == 1
        scan_node = scan_node[0]
        scan_node_topo = scan_node.op.fn.maker.fgraph.toposort()

        # check that there is no gpu transfer in the inner loop.
        assert any([isinstance(node.op, GpuElemwise) for node in scan_node_topo])
        assert not any([isinstance(node.op, HostFromGpu) for node in scan_node_topo])
        assert not any([isinstance(node.op, GpuFromHost) for node in scan_node_topo])

    # This second version test the second case in the optimizer to the gpu.
    def test_one_sequence_one_output_weights_gpu2(self):
        def f_rnn(u_t, x_tm1, W_in, W):
            return u_t * W_in + x_tm1 * W

        u = theano.tensor.fvector("u")
        x0 = theano.tensor.fscalar("x0")
        W_in = theano.tensor.fscalar("win")
        W = theano.tensor.fscalar("w")
        output, updates = scan(
            f_rnn,
            u,
            x0,
            [W_in, W],
            n_steps=None,
            truncate_gradient=-1,
            go_backwards=False,
            mode=mode_with_gpu,
        )

        f2 = theano.function(
            [u, x0, W_in, W],
            output,
            updates=updates,
            allow_input_downcast=True,
            mode=mode_with_gpu,
        )

        # get random initial values
        rng = np.random.RandomState(utt.fetch_seed())
        v_u = rng.uniform(size=(4,), low=-5.0, high=5.0)
        v_x0 = rng.uniform()
        W = rng.uniform()
        W_in = rng.uniform()

        # compute the output in numpy
        v_out = np.zeros((4,))
        v_out[0] = v_u[0] * W_in + v_x0 * W
        for step in range(1, 4):
            v_out[step] = v_u[step] * W_in + v_out[step - 1] * W

        theano_values = f2(v_u, v_x0, W_in, W)
        utt.assert_allclose(theano_values, v_out)

        topo = f2.maker.fgraph.toposort()
        assert sum([isinstance(node.op, HostFromGpu) for node in topo]) == 1
        assert sum([isinstance(node.op, GpuFromHost) for node in topo]) == 4

        scan_node = [node for node in topo if isinstance(node.op, scan.op.Scan)]
        assert len(scan_node) == 1
        scan_node = scan_node[0]
        scan_node_topo = scan_node.op.fn.maker.fgraph.toposort()

        # check that there is no gpu transfer in the inner loop.
        assert any([isinstance(node.op, GpuElemwise) for node in scan_node_topo])
        assert not any([isinstance(node.op, HostFromGpu) for node in scan_node_topo])
        assert not any([isinstance(node.op, GpuFromHost) for node in scan_node_topo])

    # This third test checks that scan can deal with a mixture of dtypes as
    # outputs when is running on GPU
    def test_gpu3_mixture_dtype_outputs(self):
        def f_rnn(u_t, x_tm1, W_in, W):
            return (u_t * W_in + x_tm1 * W, theano.tensor.cast(u_t + x_tm1, "int64"))

        u = theano.tensor.fvector("u")
        x0 = theano.tensor.fscalar("x0")
        W_in = theano.tensor.fscalar("win")
        W = theano.tensor.fscalar("w")
        output, updates = scan(
            f_rnn,
            u,
            [x0, None],
            [W_in, W],
            n_steps=None,
            truncate_gradient=-1,
            go_backwards=False,
            mode=mode_with_gpu,
        )

        f2 = theano.function(
            [u, x0, W_in, W],
            output,
            updates=updates,
            allow_input_downcast=True,
            mode=mode_with_gpu,
        )

        # get random initial values
        rng = np.random.RandomState(utt.fetch_seed())
        v_u = rng.uniform(size=(4,), low=-5.0, high=5.0)
        v_x0 = rng.uniform()
        W = rng.uniform()
        W_in = rng.uniform()

        # compute the output in numpy
        v_out1 = np.zeros((4,))
        v_out2 = np.zeros((4,), dtype="int64")
        v_out1[0] = v_u[0] * W_in + v_x0 * W
        v_out2[0] = v_u[0] + v_x0
        for step in range(1, 4):
            v_out1[step] = v_u[step] * W_in + v_out1[step - 1] * W
            v_out2[step] = np.int64(v_u[step] + v_out1[step - 1])

        theano_out1, theano_out2 = f2(v_u, v_x0, W_in, W)
        utt.assert_allclose(theano_out1, v_out1)
        utt.assert_allclose(theano_out2, v_out2)

        topo = f2.maker.fgraph.toposort()
        scan_node = [node for node in topo if isinstance(node.op, scan.op.Scan)]
        assert len(scan_node) == 1
        scan_node = scan_node[0]
        assert scan_node.op.gpua

        scan_node_topo = scan_node.op.fn.maker.fgraph.toposort()

        # check that there is no gpu transfer in the inner loop.
        assert not any([isinstance(node.op, HostFromGpu) for node in scan_node_topo])
        assert not any([isinstance(node.op, GpuFromHost) for node in scan_node_topo])

    def test_gpu4_gibbs_chain(self):
        rng = np.random.RandomState(utt.fetch_seed())
        v_vsample = np.array(
            rng.binomial(
                1,
                0.5,
                size=(3, 20),
            ),
            dtype="float32",
        )
        vsample = theano.shared(v_vsample)
        trng = theano.sandbox.rng_mrg.MRG_RandomStreams(utt.fetch_seed())

        def f(vsample_tm1):
            return (
                trng.binomial(vsample_tm1.shape, n=1, p=0.3, dtype="float32")
                * vsample_tm1
            )

        theano_vsamples, updates = scan(
            f,
            [],
            vsample,
            [],
            n_steps=10,
            truncate_gradient=-1,
            go_backwards=False,
            mode=mode_with_gpu,
        )
        my_f = theano.function(
            [],
            theano_vsamples[-1],
            updates=updates,
            allow_input_downcast=True,
            mode=mode_with_gpu,
        )

        # I leave this to tested by debugmode, this test was anyway
        # more of does the graph compile kind of test
        my_f()


class ScanGpuTests:
    """
    This class defines a number of tests for Scan on GPU as well as a few
    helper functions for these tests. The GPU tests defined in this class are
    independent of the GPU backend used. Because of this, a class inheriting
    from ScanGpuTests should define the following attributes and methods to
    make the tests run on a specific backend :
    - self.gpu_backend : Reference to the backend module
    - self.mode_with_opt : Compilation mode to force usage of the gpu backend
    - self.is_scan_on_gpu(node) : Method to determine is a scan node has been
                                  moved to run on a gpu under the specific
                                  backend. Returns a boolean.
    """

    def test_one_sequence_one_output_weights_gpu1(self):
        def f_rnn(u_t, x_tm1, W_in, W):
            return u_t * W_in + x_tm1 * W

        u = theano.tensor.fvector("u")
        x0 = theano.tensor.fscalar("x0")
        W_in = theano.tensor.fscalar("win")
        W = theano.tensor.fscalar("w")

        # The following line is needed to have the first case being used
        # Otherwise, it is the second that is tested.
        mode = self.mode_with_gpu.excluding("InputToGpuOptimizer")
        output, updates = scan(
            f_rnn,
            u,
            x0,
            [W_in, W],
            n_steps=None,
            truncate_gradient=-1,
            go_backwards=False,
            mode=mode,
        )

        output = self.gpu_backend.gpu_from_host(output)
        f2 = theano.function(
            [u, x0, W_in, W],
            output,
            updates=updates,
            allow_input_downcast=True,
            mode=self.mode_with_gpu,
        )

        # get random initial values
        rng = np.random.RandomState(utt.fetch_seed())
        v_u = rng.uniform(size=(4,), low=-5.0, high=5.0)
        v_x0 = rng.uniform()
        W = rng.uniform()
        W_in = rng.uniform()

        v_u = np.asarray(v_u, dtype="float32")
        v_x0 = np.asarray(v_x0, dtype="float32")
        W = np.asarray(W, dtype="float32")
        W_in = np.asarray(W_in, dtype="float32")

        # compute the output in numpy
        v_out = np.zeros((4,))
        v_out[0] = v_u[0] * W_in + v_x0 * W
        for step in range(1, 4):
            v_out[step] = v_u[step] * W_in + v_out[step - 1] * W
        theano_values = f2(v_u, v_x0, W_in, W)
        utt.assert_allclose(theano_values, v_out)

        # TO DEL
        topo = f2.maker.fgraph.toposort()
        scan_node = [node for node in topo if isinstance(node.op, Scan)]
        assert len(scan_node) == 1
        scan_node = scan_node[0]

        topo = f2.maker.fgraph.toposort()
        assert (
            sum([isinstance(node.op, self.gpu_backend.HostFromGpu) for node in topo])
            == 0
        )
        assert (
            sum([isinstance(node.op, self.gpu_backend.GpuFromHost) for node in topo])
            == 4
        )

        scan_node = [node for node in topo if isinstance(node.op, Scan)]
        assert len(scan_node) == 1
        scan_node = scan_node[0]
        scan_node_topo = scan_node.op.fn.maker.fgraph.toposort()

        # check that there is no gpu transfer in the inner loop.
        assert any(
            [
                isinstance(node.op, self.gpu_backend.GpuElemwise)
                for node in scan_node_topo
            ]
        )
        assert not any(
            [
                isinstance(node.op, self.gpu_backend.HostFromGpu)
                for node in scan_node_topo
            ]
        )
        assert not any(
            [
                isinstance(node.op, self.gpu_backend.GpuFromHost)
                for node in scan_node_topo
            ]
        )

    # This second version test the second case in the optimizer to the gpu.
    def test_one_sequence_one_output_weights_gpu2(self):
        def f_rnn(u_t, x_tm1, W_in, W):
            return u_t * W_in + x_tm1 * W

        u = theano.tensor.fvector("u")
        x0 = theano.tensor.fscalar("x0")
        W_in = theano.tensor.fscalar("win")
        W = theano.tensor.fscalar("w")
        output, updates = scan(
            f_rnn,
            u,
            x0,
            [W_in, W],
            n_steps=None,
            truncate_gradient=-1,
            go_backwards=False,
            mode=self.mode_with_gpu,
        )

        f2 = theano.function(
            [u, x0, W_in, W],
            output,
            updates=updates,
            allow_input_downcast=True,
            mode=self.mode_with_gpu,
        )

        # get random initial values
        rng = np.random.RandomState(utt.fetch_seed())
        v_u = rng.uniform(size=(4,), low=-5.0, high=5.0)
        v_x0 = rng.uniform()
        W = rng.uniform()
        W_in = rng.uniform()

        # compute the output in numpy
        v_out = np.zeros((4,))
        v_out[0] = v_u[0] * W_in + v_x0 * W
        for step in range(1, 4):
            v_out[step] = v_u[step] * W_in + v_out[step - 1] * W
        theano_values = f2(v_u, v_x0, W_in, W)
        utt.assert_allclose(theano_values, v_out)

        topo = f2.maker.fgraph.toposort()
        assert (
            sum([isinstance(node.op, self.gpu_backend.HostFromGpu) for node in topo])
            == 1
        )
        assert (
            sum([isinstance(node.op, self.gpu_backend.GpuFromHost) for node in topo])
            == 4
        )

        scan_node = [node for node in topo if isinstance(node.op, Scan)]
        assert len(scan_node) == 1
        scan_node = scan_node[0]
        scan_node_topo = scan_node.op.fn.maker.fgraph.toposort()

        # check that there is no gpu transfer in the inner loop.
        assert any(
            [
                isinstance(node.op, self.gpu_backend.GpuElemwise)
                for node in scan_node_topo
            ]
        )
        assert not any(
            [
                isinstance(node.op, self.gpu_backend.HostFromGpu)
                for node in scan_node_topo
            ]
        )
        assert not any(
            [
                isinstance(node.op, self.gpu_backend.GpuFromHost)
                for node in scan_node_topo
            ]
        )

    # This third test checks that scan can deal with a mixture of dtypes as
    # outputs when is running on GPU
    def test_gpu3_mixture_dtype_outputs(self):
        def f_rnn(u_t, x_tm1, W_in, W):
            return (u_t * W_in + x_tm1 * W, tensor.cast(u_t + x_tm1, "int64"))

        u = theano.tensor.fvector("u")
        x0 = theano.tensor.fscalar("x0")
        W_in = theano.tensor.fscalar("win")
        W = theano.tensor.fscalar("w")
        output, updates = scan(
            f_rnn,
            u,
            [x0, None],
            [W_in, W],
            n_steps=None,
            truncate_gradient=-1,
            go_backwards=False,
            mode=self.mode_with_gpu,
        )

        f2 = theano.function(
            [u, x0, W_in, W],
            output,
            updates=updates,
            allow_input_downcast=True,
            mode=self.mode_with_gpu,
        )

        # get random initial values
        rng = np.random.RandomState(utt.fetch_seed())
        v_u = rng.uniform(size=(4,), low=-5.0, high=5.0)
        v_x0 = rng.uniform()
        W = rng.uniform()
        W_in = rng.uniform()

        # compute the output in numpy
        v_out1 = np.zeros((4,))
        v_out2 = np.zeros((4,), dtype="int64")
        v_out1[0] = v_u[0] * W_in + v_x0 * W
        v_out2[0] = v_u[0] + v_x0
        for step in range(1, 4):
            v_out1[step] = v_u[step] * W_in + v_out1[step - 1] * W
            v_out2[step] = np.int64(v_u[step] + v_out1[step - 1])

        theano_out1, theano_out2 = f2(v_u, v_x0, W_in, W)
        utt.assert_allclose(theano_out1, v_out1)
        utt.assert_allclose(theano_out2, v_out2)

        topo = f2.maker.fgraph.toposort()
        scan_node = [node for node in topo if isinstance(node.op, Scan)]
        assert len(scan_node) == 1
        scan_node = scan_node[0]
        assert self.is_scan_on_gpu(scan_node)

    def test_gibbs_chain(self):
        rng = np.random.RandomState(utt.fetch_seed())
        v_vsample = np.array(
            rng.binomial(
                1,
                0.5,
                size=(3, 20),
            ),
            dtype="float32",
        )
        vsample = theano.shared(v_vsample)
        trng = theano.sandbox.rng_mrg.MRG_RandomStreams(utt.fetch_seed())

        def f(vsample_tm1):
            return (
                trng.binomial(vsample_tm1.shape, n=1, p=0.3, dtype="float32")
                * vsample_tm1
            )

        theano_vsamples, updates = scan(
            f,
            [],
            vsample,
            [],
            n_steps=10,
            truncate_gradient=-1,
            go_backwards=False,
            mode=self.mode_with_gpu,
        )
        my_f = theano.function(
            [],
            theano_vsamples[-1],
            updates=updates,
            allow_input_downcast=True,
            mode=self.mode_with_gpu,
        )

        # I leave this to tested by debugmode, this test was anyway more of
        # doest the graph compile kind of test
        my_f()

    def test_gpu_memory_usage(self):
        # This test validates that the memory usage of the defined theano
        # function is reasonnable when executed on the GPU. It checks for
        # a bug in which one of scan's optimization was not applied which
        # made the scan node compute large and unnecessary outputs which
        # brought memory usage on the GPU to ~12G.

        # Dimensionality of input and output data (not one-hot coded)
        n_in = 100
        n_out = 100
        # Number of neurons in hidden layer
        n_hid = 4000

        # Number of minibatches
        mb_size = 2
        # Time steps in minibatch
        mb_length = 200

        # Define input variables
        xin = tensor.ftensor3(name="xin")
        yout = tensor.ftensor3(name="yout")

        # Initialize the network parameters
        U = theano.shared(np.zeros((n_in, n_hid), dtype="float32"), name="W_xin_to_l1")
        V = theano.shared(np.zeros((n_hid, n_hid), dtype="float32"), name="W_l1_to_l1")
        W = theano.shared(np.zeros((n_hid, n_out), dtype="float32"), name="W_l1_to_l2")
        nparams = [U, V, W]

        # Build the forward pass
        l1_base = tensor.dot(xin, U)

        def scan_l(baseline, last_step):
            return baseline + tensor.dot(last_step, V)

        zero_output = tensor.alloc(np.asarray(0.0, dtype="float32"), mb_size, n_hid)

        l1_out, _ = scan(
            scan_l,
            sequences=[l1_base],
            outputs_info=[zero_output],
            mode=self.mode_with_gpu_nodebug,
        )

        l2_out = tensor.dot(l1_out, W)

        # Compute the cost and take the gradient wrt params
        cost = tensor.sum((l2_out - yout) ** 2)
        grads = tensor.grad(cost, nparams)
        updates = list(zip(nparams, (n - g for n, g in zip(nparams, grads))))

        # Compile the theano function
        feval_backprop = theano.function(
            [xin, yout], cost, updates=updates, mode=self.mode_with_gpu_nodebug
        )

        # Validate that the PushOutScanOutput optimization has been applied
        # by checking the number of outputs of the grad Scan node in the
        # compiled function.
        nodes = feval_backprop.maker.fgraph.toposort()
        scan_nodes = [n for n in nodes if isinstance(n.op, Scan)]

        # The grad scan is always the 2nd one according to toposort. If the
        # optimization has been applied, it has 2 outputs, otherwise 3.
        grad_scan_node = scan_nodes[1]
        assert len(grad_scan_node.outputs) == 2, len(grad_scan_node.outputs)

        # Call the theano function to ensure the absence of a memory error
        feval_backprop(
            np.zeros((mb_length, mb_size, n_in), dtype="float32"),
            np.zeros((mb_length, mb_size, n_out), dtype="float32"),
        )

    def test_memory_reuse_gpudimshuffle(self):
        # Test the memory pre-allocation feature in scan when one output is
        # the result of a GpuDimshuffle (because an optimization in
        # GpuDimshuffle can cause issues with the memory pre-allocation
        # where it falsely thinks that a pre-allocated memory region has
        # been used when it hasn't).
        def inner_fn(seq1, recurrent_out):
            temp = seq1 + recurrent_out.sum()
            output1 = temp.dimshuffle(1, 0)
            output2 = temp.sum() + recurrent_out
            return output1, output2

        input1 = theano.tensor.ftensor3()
        init = theano.tensor.ftensor3()
        outputs_info = [None, init]

        out, _ = scan(
            inner_fn,
            sequences=[input1],
            outputs_info=outputs_info,
            mode=self.mode_with_gpu,
        )

        out1 = out[0].flatten()
        out2 = out[1].flatten()

        fct = theano.function([input1, init], [out1, out2], mode=self.mode_with_gpu)

        output = fct(
            np.ones((2, 1, 1), dtype="float32"), np.ones((1, 1, 1), dtype="float32")
        )

        expected_output = (
            np.array([2, 4], dtype="float32"),
            np.array([3, 7], dtype="float32"),
        )
        utt.assert_allclose(output, expected_output)


class TestScanGpuarray(ScanGpuTests):
    """
    This class takes the gpu tests for scan that are defined in
    class ScanGpuTests and runs them using the gpuarray backend.
    """

    def setup_method(self):

        self.gpu_backend = gpuarray

        # This is unfortunate, but required
        def gpu_from_host(v):
            return gpuarray.GpuFromHost(None)(v)

        self.gpu_backend.gpu_from_host = gpu_from_host

        self.mode_with_gpu = mode_with_opt.including("gpuarray", "scan")
        self.mode_with_gpu_nodebug = mode_nodebug.including("gpuarray", "scan")

        # Skip the test if pygpu is not available
        if not self.gpu_backend.pygpu_activated:
            pytest.skip("Optional package pygpu disabled")

        utt.seed_rng()

    def is_scan_on_gpu(self, node):
        return node.op.info.get("gpua", False)


class TestScanCheckpoint:
    def setup_method(self):
        self.k = tensor.iscalar("k")
        self.A = tensor.vector("A")
        result, _ = scan(
            fn=lambda prior_result, A: prior_result * A,
            outputs_info=tensor.ones_like(self.A),
            non_sequences=self.A,
            n_steps=self.k,
        )
        result_check, _ = scan_checkpoints(
            fn=lambda prior_result, A: prior_result * A,
            outputs_info=tensor.ones_like(self.A),
            non_sequences=self.A,
            n_steps=self.k,
            save_every_N=100,
        )
        self.result = result[-1]
        self.result_check = result_check[-1]
        self.grad_A = tensor.grad(self.result.sum(), self.A)
        self.grad_A_check = tensor.grad(self.result_check.sum(), self.A)

    def test_memory(self):
        from tests.gpuarray.config import mode_with_gpu  # noqa

        f = theano.function(
            inputs=[self.A, self.k], outputs=self.grad_A, mode=mode_with_gpu
        )
        f_check = theano.function(
            inputs=[self.A, self.k], outputs=self.grad_A_check, mode=mode_with_gpu
        )
        free_gmem = theano.gpuarray.type._context_reg[None].free_gmem
        data = np.ones(free_gmem // 3000, dtype=np.float32)
        # Check that it works with the checkpoints
        size = 1000
        if isinstance(mode_with_gpu, theano.compile.DebugMode):
            size = 100
        f_check(data, size)
        # Check that the basic scan fails in that case
        # Skip that check in DebugMode, as it can fail in different ways
        if not isinstance(mode_with_gpu, theano.compile.DebugMode):
            with pytest.raises(GpuArrayException):
                f(data, 1000)
