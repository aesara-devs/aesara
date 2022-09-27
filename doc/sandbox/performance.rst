
===========
Performance
===========

Aesara uses several tricks to obtain good performance:
 * common sub-expression elimination
 * [custom generated] C code for many operations
 * pre-allocation of temporary storage
 * loop fusion (which gcc normally can't do)

On my neural net experiments for my course projects, I was getting around 10x
speed improvements over basic numpy by using aesara.
[More specific speed tests would be nice.]


With a little work, Aesara could also implement more sophisticated
rewrites:

 * automatic ordering of matrix multiplications
 * profile-based memory layout decisions (e.g. row-major vs. col-major)
 * gcc intrinsics to use MMX, SSE2 parallelism for faster element-wise arithmetic
 * conditional expressions
