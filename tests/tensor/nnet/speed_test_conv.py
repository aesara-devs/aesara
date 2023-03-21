import time

import numpy as np

from aesara import function
from aesara.compile.mode import Mode
from aesara.tensor.nnet.conv import ConvOp
from aesara.tensor.type import TensorType, dmatrix


def flip(kern, kshp):
    "flip the kernel as scipy.convolv2d do it flipped."
    flip = np.zeros(kern.shape)
    if len(kern.shape) == 2:
        kern = kern.reshape(-1)
        it = reversed(kern)
        for i in range(kshp[0]):
            for j in range(kshp[1]):
                flip[i, j] = next(it)
    elif len(kern.shape) == 3:
        kern = kern.reshape(kern.shape[0], -1)
        for k in range(kern.shape[0]):
            it = reversed(kern[k, :])
            for i in range(kshp[0]):
                for j in range(kshp[1]):
                    flip[k, i, j] = next(it)
    elif len(kern.shape) == 4:
        kern = kern.reshape(kern.shape[0], kern.shape[1], -1)
        for k in range(kern.shape[0]):
            for m in range(kern.shape[1]):
                it = reversed(kern[k, m, :])
                for i in range(kshp[0]):
                    for j in range(kshp[1]):
                        flip[k, m, i, j] = next(it)
    else:
        raise NotImplementedError()
    return flip


global_rng = np.random.default_rng(3423489)

dmatrix4 = TensorType("float64", shape=(None, None, None, None))


def exec_multilayer_conv_nnet_old(
    conv_mode,
    ss,
    bsize,
    imshp,
    kshps,
    nkerns,
    unroll_batch=0,
    unroll_kern=0,
    img=None,
    validate=True,
    conv_op_py=False,
    do_print=True,
    repeat=1,
    unroll_patch=False,
    unroll_patch_size=False,
    verbose=0,
):
    if img is None:
        img = dmatrix()

    # build actual input images
    imgval = global_rng.random((bsize, imshp[0], imshp[1], imshp[2]))

    a = dmatrix()
    kerns = [a for i in nkerns]
    inputs4 = dmatrix4()
    kerns4 = dmatrix4()

    # for each layer
    ntot = 0
    tctot = 0
    tpytot = 0

    for kshp, kern, nkern, n_layer in zip(kshps, kerns, nkerns, range(len(nkerns))):
        if do_print:
            print("************* layer %i ***************" % n_layer)
            print(conv_mode, ss, n_layer, kshp, nkern)

        # actual values
        w = global_rng.random(np.r_[nkern, imshp[0], kshp])
        w_flip = flip(w, kshp).reshape(w.shape)

        # manual implementation
        # check first stage
        padimg = imgval
        if conv_mode == "full":
            padimg_shp = np.array(imshp[1:]) + 2 * (np.array(kshp) - np.array([1, 1]))
            padimg = np.zeros(np.r_[bsize, imshp[0], padimg_shp])
            padimg[
                :, :, kshp[0] - 1 : -kshp[0] + 1, kshp[1] - 1 : -kshp[1] + 1
            ] = imgval

        outshp = np.hstack(
            (nkern, ConvOp.getOutputShape(imshp[1:], kshp, ss, conv_mode))
        )

        time1 = time.perf_counter()
        outval = np.zeros(np.r_[bsize, outshp])
        if validate:
            # causes an atexit problem

            try:
                from scipy.signal.signaltools import _bvalfromboundary, _valfrommode
                from scipy.signal.sigtools import _convolve2d
            except ImportError:
                from scipy.signal._signaltools import _bvalfromboundary, _valfrommode
                from scipy.signal._sigtools import _convolve2d

            val = _valfrommode(conv_mode)
            bval = _bvalfromboundary("fill")
            for b in range(bsize):  # loop over batches
                for n in range(nkern):  # loop over filters
                    for i in range(imshp[0]):  # loop over input feature maps
                        outval[b, n, ...] += _convolve2d(
                            imgval[b, i, ...], w_flip[n, i, ...], 1, val, bval, 0
                        )[0 :: ss[0], 0 :: ss[1]]
            ntot += time.perf_counter() - time1

        # ConvOp
        if unroll_patch and not unroll_patch_size:
            conv_op = ConvOp(
                dx=ss[0],
                dy=ss[1],
                output_mode=conv_mode,
                unroll_patch=unroll_patch,
                verbose=verbose,
            )(inputs4, kerns4)
        else:
            conv_op = ConvOp(
                imshp,
                kshp,
                nkern,
                bsize,
                ss[0],
                ss[1],
                conv_mode,
                unroll_batch=unroll_batch,
                unroll_kern=unroll_kern,
                unroll_patch=unroll_patch,
                verbose=verbose,
            )(inputs4, kerns4)
        # l1shp = np.hstack((nkern,
        #                ConvOp.getOutputShape(imshp[1:], kshp, ss, conv_mode)))
        propup2 = function([inputs4, kerns4], conv_op)
        propup3 = function([inputs4, kerns4], conv_op, mode=Mode(linker="py"))

        time1 = time.perf_counter()
        for i in range(repeat):
            hidval2_ = propup2(imgval, w_flip)
        hidval2 = hidval2_  # [:,:,0::ss[0],0::ss[1]]
        tctot += time.perf_counter() - time1

        if conv_op_py:
            time1 = time.perf_counter()
            for i in range(repeat):
                hidval3_ = propup3(imgval, w_flip)
            hidval3 = hidval3_  # [:,:,0::ss[0],0::ss[1]]
            tpytot += time.perf_counter() - time1
            assert (np.abs(hidval2 - hidval3) < 1e-5).all()
        else:
            tpytot += 0

        if validate:
            temp = np.abs(outval - hidval2)
            assert (temp < 1e-5).all()
        if validate and conv_op_py:
            temp = np.abs(outval - hidval3)
            assert (temp < 1e-5).all()

        imshp = tuple(outshp)
        imgval = outval.reshape(bsize, outshp[0], outshp[1], outshp[2])

    return tctot, tpytot, ntot


def exec_multilayer_conv_nnet(
    conv_mode,
    ss,
    bsize,
    imshp,
    kshps,
    nkerns,
    unroll_batch=0,
    unroll_kern=0,
    img=None,
    do_print=True,
    repeat=1,
    unroll_patch=False,
    unroll_patch_size=False,
    verbose=0,
):
    if img is None:
        img = dmatrix()

    # build actual input images
    imgval = global_rng.random((bsize, imshp[0], imshp[1], imshp[2]))

    a = dmatrix()
    kerns = [a for i in nkerns]
    inputs4 = dmatrix4()
    kerns4 = dmatrix4()

    # for each layer
    ntot = 0
    tctot = 0
    tpytot = 0

    for kshp, kern, nkern, n_layer in zip(kshps, kerns, nkerns, range(len(nkerns))):
        if do_print:
            print("************* layer %i ***************" % n_layer)
            print(conv_mode, ss, n_layer, kshp, nkern)

        # actual values
        w = global_rng.random(np.r_[nkern, imshp[0], kshp])
        w_flip = flip(w, kshp).reshape(w.shape)

        outshp = np.hstack(
            (nkern, ConvOp.getOutputShape(imshp[1:], kshp, ss, conv_mode))
        )

        time1 = time.perf_counter()
        # outval = np.zeros(np.r_[bsize, outshp])

        # ConvOp
        if unroll_patch and not unroll_patch_size:
            conv_op = ConvOp(
                dx=ss[0],
                dy=ss[1],
                output_mode=conv_mode,
                unroll_patch=unroll_patch,
                verbose=verbose,
            )(inputs4, kerns4)
        else:
            conv_op = ConvOp(
                imshp,
                kshp,
                nkern,
                bsize,
                ss[0],
                ss[1],
                conv_mode,
                unroll_batch=unroll_batch,
                unroll_kern=unroll_kern,
                unroll_patch=unroll_patch,
                verbose=verbose,
            )(inputs4, kerns4)
        # l1shp = np.hstack((nkern,
        #                ConvOp.getOutputShape(imshp[1:], kshp, ss, conv_mode)))
        propup2 = function([inputs4, kerns4], conv_op)

        time1 = time.perf_counter()
        for i in range(repeat):
            propup2(imgval, w_flip)
        tctot += time.perf_counter() - time1

        imshp = tuple(outshp)
        # imgval = outval.reshape(bsize, outshp[0], outshp[1], outshp[2])

    return tctot, tpytot, ntot


def speed_multilayer_conv():
    # calculate the speed up of different combination of unroll
    # put the parameter to the same you will try.
    # validate = False  # we don't validate the result to have it much faster!
    repeat = 3
    verbose = 1
    unroll_batch = [1, 2, 3, 4, 5, 6, 10]  # 15, 30, 60 always much slower
    unroll_kern = [1, 2, 3, 4, 5, 6, 10]  # 15, 30, 60 always much slower
    # unroll_batch = [1,4,5]
    # unroll_kern = [1,4,5]
    # unroll_batch = [1,4]
    # unroll_kern = [1,4]
    # unroll_patch = [True, False]
    bsize = 60  # batch size
    imshp_start = (1, 48, 48)  # un square shape to test more corner case.
    kshps = ([11, 12],)  # un square shape to test more corner case.
    nkerns = [60]  # per output pixel
    ssizes = [
        (1, 1),
    ]  # (1,1)]#(2,2) bugged
    convmodes = ["valid", "full"]
    # do_convolve2 = False
    a = dmatrix()
    kerns = [a for i in nkerns]

    assert len(kshps) == len(nkerns) == len(kerns)
    timing = np.zeros(
        (len(unroll_batch), len(unroll_kern), 3, len(convmodes) * len(ssizes))
    )
    t_b_k = []
    # calculate the timing with unrolling

    print("time unroll batch kern")
    best = []
    worst = []
    t_ = []
    for unroll_b, n_b in zip(unroll_batch, range(len(unroll_batch))):
        for unroll_k, n_k in zip(unroll_kern, range(len(unroll_kern))):
            t_b_k.append(str(unroll_b) + "/" + str(unroll_k))
            if not t_:
                tctot, tpytot, ntot = [], [], []
                for conv_mode, n_mode in zip(convmodes, range(len(convmodes))):
                    for ss, n_ss in zip(ssizes, range(len(ssizes))):
                        # tctot_, tpytot_, ntot_ = exec_multilayer_conv_nnet_old(conv_mode, ss, bsize, imshp_start, kshps, nkerns, unroll_batch=unroll_b, unroll_kern=unroll_k, validate=validate, verbose=verbose,do_print=False)
                        tctot_, tpytot_, ntot_ = exec_multilayer_conv_nnet(
                            conv_mode,
                            ss,
                            bsize,
                            imshp_start,
                            kshps,
                            nkerns,
                            unroll_batch=unroll_b,
                            unroll_kern=unroll_k,
                            verbose=verbose,
                            do_print=False,
                            repeat=repeat,
                        )
                        tctot += [tctot_]
                        tpytot += [tpytot_]
                        ntot += [ntot_]
                if unroll_b == 4 and unroll_k == 4:
                    # print "unroll 4/4",tctot
                    best = tctot
                if unroll_b == 1 and unroll_k == 1:
                    # print "unroll 1/1",tctot
                    worst = tctot
                timing[n_b, n_k] = [
                    tctot,
                    tpytot,
                    ntot,
                ]  # [sum(tctot), sum(tpytot), sum(ntot)]
    if not t_:
        t = timing[:, :, 0, :]  # We select only the c timing.
    else:
        t = t_
    t = np.asarray(t)
    # calculate the old timing
    print("time old version")
    tctot, tpytot, ntot = [], [], []
    tctot_ = []
    if not tctot_:
        for conv_mode, n_mode in zip(convmodes, range(len(convmodes))):
            for ss, n_ss in zip(ssizes, range(len(ssizes))):
                # tctot_, tpytot_, ntot_ = exec_multilayer_conv_nnet_old(conv_mode, ss, bsize, imshp_start, kshps, nkerns, unroll_batch=0, unroll_kern=0, validate=validate, verbose=verbose,do_print=False)
                tctot_, tpytot_, ntot_ = exec_multilayer_conv_nnet(
                    conv_mode,
                    ss,
                    bsize,
                    imshp_start,
                    kshps,
                    nkerns,
                    unroll_batch=0,
                    unroll_kern=0,
                    verbose=verbose,
                    do_print=False,
                    repeat=repeat,
                )
                tctot += [tctot_]
                tpytot += [tpytot_]
                ntot += [ntot_]
    else:
        tctot = np.asarray(tctot_)
    print(f"old code timing {sum(tctot):.3f}s", tctot)
    best = np.asarray(best)
    worst = np.asarray(worst)
    print("timing for unrolled version")
    print("unroll_batch/unroll_kern valid_mode full_mode")
    for n_b in range(len(unroll_batch)):
        for n_k in range(len(unroll_kern)):
            print((unroll_batch[n_b], unroll_kern[n_k]) + tuple(t[n_b, n_k]), ",")
    # t_detail = t
    t = t.sum(axis=2)
    print(
        f"max {t.max():.3f}s",
        "max param(batch unloop size/kernel unloop size)",
        t_b_k[t.argmax()],
    )
    print(
        f"min {t.min():.3f}s",
        "min param(batch unloop size/kernel unloop size)",
        t_b_k[t.argmin()],
    )
    print(
        f"speedup vs (1/1){t.max() / t.min():.3f}x, vs old {sum(tctot) / t.min():.3f}x"
    )
    print(worst / best, tctot / best)

    # calculate the timing of unroll_patch
    print("time unroll_patch")
    tctot_patch = []
    tctot_patch_size = []
    for conv_mode, n_mode in zip(convmodes, range(len(convmodes))):
        for ss, n_ss in zip(ssizes, range(len(ssizes))):
            # tctot_, tpytot_, ntot_ = exec_multilayer_conv_nnet_old(conv_mode, ss, bsize, imshp_start, kshps, nkerns, unroll_batch=0, unroll_kern=0, validate=validate,unroll_patch=True,verbose=verbose,do_print=False)
            tctot_, tpytot_, ntot_ = exec_multilayer_conv_nnet(
                conv_mode,
                ss,
                bsize,
                imshp_start,
                kshps,
                nkerns,
                unroll_batch=0,
                unroll_kern=0,
                unroll_patch=True,
                verbose=verbose,
                do_print=False,
                repeat=repeat,
            )
            tctot_patch += [tctot_]
            # tctot_, tpytot_, ntot_ = exec_multilayer_conv_nnet_old(conv_mode, ss, bsize, imshp_start, kshps, nkerns, unroll_batch=0, unroll_kern=0, validate=validate,unroll_patch=True,verbose=verbose,do_print=False,unroll_patch_size=True)
            tctot_, tpytot_, ntot_ = exec_multilayer_conv_nnet(
                conv_mode,
                ss,
                bsize,
                imshp_start,
                kshps,
                nkerns,
                unroll_batch=0,
                unroll_kern=0,
                unroll_patch=True,
                verbose=verbose,
                do_print=False,
                unroll_patch_size=True,
                repeat=repeat,
            )
            tctot_patch_size += [tctot_]

    t_patch = sum(tctot_patch)
    print("unroll_patch without shape time", tctot_patch)
    print(
        f"speedup vs (1/1){t.max() / t_patch:.3f}x, vs old {sum(tctot) / t_patch:.3f}x"
    )
    print(best / tctot_patch, worst / tctot_patch)
    t_patch_size = sum(tctot_patch_size)
    print("unroll_patch with shape time", tctot_patch_size)
    print(
        "speedup vs (1/1)%.3fx, vs old %.3fx"
        % (t.max() / t_patch_size, sum(tctot) / t_patch_size)
    )
    print(best / tctot_patch_size, worst / tctot_patch_size)
    return


if __name__ == "__main__":
    speed_multilayer_conv()
