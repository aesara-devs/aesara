from theano import printing
from theano.printing import pprint
from theano.tensor.elemwise import DimShuffle, _scal_elemwise


@_scal_elemwise
def lt_inplace(a, b):
    """a < b (inplace on a)"""


@_scal_elemwise
def gt_inplace(a, b):
    """a > b (inplace on a)"""


@_scal_elemwise
def le_inplace(a, b):
    """a <= b (inplace on a)"""


@_scal_elemwise
def ge_inplace(a, b):
    """a >= b (inplace on a)"""


@_scal_elemwise
def eq_inplace(a, b):
    """a == b (inplace on a)"""


@_scal_elemwise
def neq_inplace(a, b):
    """a != b (inplace on a)"""


@_scal_elemwise
def and__inplace(a, b):
    """bitwise a & b (inplace on a)"""


@_scal_elemwise
def or__inplace(a, b):
    """bitwise a | b (inplace on a)"""


@_scal_elemwise
def xor_inplace(a, b):
    """bitwise a ^ b (inplace on a)"""


@_scal_elemwise
def invert_inplace(a):
    """bitwise ~a (inplace on a)"""


@_scal_elemwise
def abs__inplace(a):
    """|`a`| (inplace on `a`)"""


@_scal_elemwise
def exp_inplace(a):
    """e^`a` (inplace on `a`)"""


@_scal_elemwise
def exp2_inplace(a):
    """2^`a` (inplace on `a`)"""


@_scal_elemwise
def expm1_inplace(a):
    """e^`a` - 1 (inplace on `a`)"""


@_scal_elemwise
def neg_inplace(a):
    """-a (inplace on a)"""


@_scal_elemwise
def inv_inplace(a):
    """1.0/a (inplace on a)"""


@_scal_elemwise
def log_inplace(a):
    """base e logarithm of a (inplace on a)"""


@_scal_elemwise
def log1p_inplace(a):
    """log(1+a)"""


@_scal_elemwise
def log2_inplace(a):
    """base 2 logarithm of a (inplace on a)"""


@_scal_elemwise
def log10_inplace(a):
    """base 10 logarithm of a (inplace on a)"""


@_scal_elemwise
def sgn_inplace(a):
    """sign of `a` (inplace on `a`)"""


@_scal_elemwise
def ceil_inplace(a):
    """ceil of `a` (inplace on `a`)"""


@_scal_elemwise
def floor_inplace(a):
    """floor of `a` (inplace on `a`)"""


@_scal_elemwise
def trunc_inplace(a):
    """trunc of `a` (inplace on `a`)"""


@_scal_elemwise
def round_half_to_even_inplace(a):
    """round_half_to_even_inplace(a) (inplace on `a`)"""


@_scal_elemwise
def round_half_away_from_zero_inplace(a):
    """round_half_away_from_zero_inplace(a) (inplace on `a`)"""


@_scal_elemwise
def sqr_inplace(a):
    """square of `a` (inplace on `a`)"""


@_scal_elemwise
def sqrt_inplace(a):
    """square root of `a` (inplace on `a`)"""


@_scal_elemwise
def deg2rad_inplace(a):
    """convert degree `a` to radian(inplace on `a`)"""


@_scal_elemwise
def rad2deg_inplace(a):
    """convert radian `a` to degree(inplace on `a`)"""


@_scal_elemwise
def cos_inplace(a):
    """cosine of `a` (inplace on `a`)"""


@_scal_elemwise
def arccos_inplace(a):
    """arccosine of `a` (inplace on `a`)"""


@_scal_elemwise
def sin_inplace(a):
    """sine of `a` (inplace on `a`)"""


@_scal_elemwise
def arcsin_inplace(a):
    """arcsine of `a` (inplace on `a`)"""


@_scal_elemwise
def tan_inplace(a):
    """tangent of `a` (inplace on `a`)"""


@_scal_elemwise
def arctan_inplace(a):
    """arctangent of `a` (inplace on `a`)"""


@_scal_elemwise
def arctan2_inplace(a, b):
    """arctangent of `a` / `b` (inplace on `a`)"""


@_scal_elemwise
def cosh_inplace(a):
    """hyperbolic cosine of `a` (inplace on `a`)"""


@_scal_elemwise
def arccosh_inplace(a):
    """hyperbolic arc cosine of `a` (inplace on `a`)"""


@_scal_elemwise
def sinh_inplace(a):
    """hyperbolic sine of `a` (inplace on `a`)"""


@_scal_elemwise
def arcsinh_inplace(a):
    """hyperbolic arc sine of `a` (inplace on `a`)"""


@_scal_elemwise
def tanh_inplace(a):
    """hyperbolic tangent of `a` (inplace on `a`)"""


@_scal_elemwise
def arctanh_inplace(a):
    """hyperbolic arc tangent of `a` (inplace on `a`)"""


@_scal_elemwise
def erf_inplace(a):
    """error function"""


@_scal_elemwise
def erfc_inplace(a):
    """complementary error function"""


@_scal_elemwise
def erfcx_inplace(a):
    """scaled complementary error function"""


@_scal_elemwise
def gamma_inplace(a):
    """gamma function"""


@_scal_elemwise
def gammaln_inplace(a):
    """log gamma function"""


@_scal_elemwise
def psi_inplace(a):
    """derivative of log gamma function"""


@_scal_elemwise
def tri_gamma_inplace(a):
    """second derivative of the log gamma function"""


@_scal_elemwise
def chi2sf_inplace(x, k):
    """chi squared survival function"""


@_scal_elemwise
def j0_inplace(x):
    """Bessel function of the first kind of order 0."""


@_scal_elemwise
def j1_inplace(x):
    """Bessel function of the first kind of order 1."""


@_scal_elemwise
def jv_inplace(v, x):
    """Bessel function of the first kind of order v (real)."""


@_scal_elemwise
def i0_inplace(x):
    """Modified Bessel function of the first kind of order 0."""


@_scal_elemwise
def i1_inplace(x):
    """Modified Bessel function of the first kind of order 1."""


@_scal_elemwise
def iv_inplace(v, x):
    """Modified Bessel function of the first kind of order v (real)."""


@_scal_elemwise
def second_inplace(a):
    """Fill `a` with `b`"""


fill_inplace = second_inplace
pprint.assign(fill_inplace, printing.FunctionPrinter("fill="))


@_scal_elemwise(symbolname="scalar_maximum_inplace")
def maximum_inplace(a, b):
    """elementwise addition (inplace on `a`)"""


@_scal_elemwise(symbolname="scalar_minimum_inplace")
def minimum_inplace(a, b):
    """elementwise addition (inplace on `a`)"""


@_scal_elemwise
def add_inplace(a, b):
    """elementwise addition (inplace on `a`)"""


@_scal_elemwise
def sub_inplace(a, b):
    """elementwise subtraction (inplace on `a`)"""


@_scal_elemwise
def mul_inplace(a, b):
    """elementwise multiplication (inplace on `a`)"""


@_scal_elemwise
def true_div_inplace(a, b):
    """elementwise division (inplace on `a`)"""


@_scal_elemwise
def int_div_inplace(a, b):
    """elementwise division (inplace on `a`)"""


@_scal_elemwise
def mod_inplace(a, b):
    """elementwise modulo (inplace on `a`)"""


@_scal_elemwise
def pow_inplace(a, b):
    """elementwise power (inplace on `a`)"""


@_scal_elemwise
def conj_inplace(a):
    """elementwise conjugate (inplace on `a`)"""


pprint.assign(add_inplace, printing.OperatorPrinter("+=", -2, "either"))
pprint.assign(mul_inplace, printing.OperatorPrinter("*=", -1, "either"))
pprint.assign(sub_inplace, printing.OperatorPrinter("-=", -2, "left"))
pprint.assign(neg_inplace, printing.OperatorPrinter("-=", 0, "either"))
pprint.assign(true_div_inplace, printing.OperatorPrinter("/=", -1, "left"))
pprint.assign(int_div_inplace, printing.OperatorPrinter("//=", -1, "left"))
pprint.assign(pow_inplace, printing.OperatorPrinter("**=", 1, "right"))


def transpose_inplace(x, **kwargs):
    "Perform a transpose on a tensor without copying the underlying storage"
    dims = list(range(x.ndim - 1, -1, -1))
    return DimShuffle(x.broadcastable, dims, inplace=True)(x)
