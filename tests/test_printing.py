"""
Tests of printing functionality
"""
import logging
from io import StringIO

import pytest

import aesara
from aesara.printing import (
    debugprint,
    min_informative_str,
    pp,
    pydot_imported,
    pydotprint,
)
from aesara.tensor.type import dvector, matrix


@pytest.mark.skipif(not pydot_imported, reason="pydot not available")
def test_pydotprint_cond_highlight():
    # This is a REALLY PARTIAL TEST.
    # I did them to help debug stuff.
    x = dvector()
    f = aesara.function([x], x * 2)
    f([1, 2, 3, 4])

    s = StringIO()
    new_handler = logging.StreamHandler(s)
    new_handler.setLevel(logging.DEBUG)
    orig_handler = aesara.logging_default_handler

    aesara.aesara_logger.removeHandler(orig_handler)
    aesara.aesara_logger.addHandler(new_handler)
    try:
        pydotprint(f, cond_highlight=True, print_output_file=False)
    finally:
        aesara.aesara_logger.addHandler(orig_handler)
        aesara.aesara_logger.removeHandler(new_handler)

    assert (
        s.getvalue() == "pydotprint: cond_highlight is set but there"
        " is no IfElse node in the graph\n"
    )


@pytest.mark.skipif(not pydot_imported, reason="pydot not available")
def test_pydotprint_return_image():
    x = dvector()
    ret = pydotprint(x * 2, return_image=True)
    assert isinstance(ret, (str, bytes))


@pytest.mark.skipif(not pydot_imported, reason="pydot not available")
def test_pydotprint_long_name():
    # This is a REALLY PARTIAL TEST.
    # It prints a graph where there are variable and apply nodes whose long
    # names are different, but not the shortened names.
    # We should not merge those nodes in the dot graph.
    x = dvector()
    mode = aesara.compile.mode.get_default_mode().excluding("fusion")
    f = aesara.function([x], [x * 2, x + x], mode=mode)
    f([1, 2, 3, 4])

    pydotprint(f, max_label_size=5, print_output_file=False)
    pydotprint([x * 2, x + x], max_label_size=5, print_output_file=False)


@pytest.mark.skipif(
    not pydot_imported or aesara.config.mode in ("DebugMode", "DEBUG_MODE"),
    reason="Can't profile in DebugMode",
)
def test_pydotprint_profile():
    A = matrix()
    prof = aesara.compile.ProfileStats(atexit_print=False, gpu_checks=False)
    f = aesara.function([A], A + 1, profile=prof)
    pydotprint(f, print_output_file=False)
    f([[1]])
    pydotprint(f, print_output_file=False)


def test_min_informative_str():
    # evaluates a reference output to make sure the
    # min_informative_str function works as intended

    A = matrix(name="A")
    B = matrix(name="B")
    C = A + B
    C.name = "C"
    D = matrix(name="D")
    E = matrix(name="E")

    F = D + E
    G = C + F

    mis = min_informative_str(G).replace("\t", "        ")

    reference = """A. Elemwise{add,no_inplace}
 B. C
 C. Elemwise{add,no_inplace}
  D. D
  E. E"""

    if mis != reference:
        print("--" + mis + "--")
        print("--" + reference + "--")

    assert mis == reference


def test_debugprint():
    A = matrix(name="A")
    B = matrix(name="B")
    C = A + B
    C.name = "C"
    D = matrix(name="D")
    E = matrix(name="E")

    F = D + E
    G = C + F
    mode = aesara.compile.get_default_mode().including("fusion")
    g = aesara.function([A, B, D, E], G, mode=mode)

    # just test that it work
    s = StringIO()
    debugprint(G, file=s)

    # test ids=int
    s = StringIO()
    debugprint(G, file=s, ids="int")
    s = s.getvalue()
    # The additional white space are needed!
    reference = (
        "\n".join(
            [
                "Elemwise{add,no_inplace} [id 0] ''   ",
                " |Elemwise{add,no_inplace} [id 1] 'C'   ",
                " | |A [id 2]",
                " | |B [id 3]",
                " |Elemwise{add,no_inplace} [id 4] ''   ",
                "   |D [id 5]",
                "   |E [id 6]",
            ]
        )
        + "\n"
    )

    if s != reference:
        print("--" + s + "--")
        print("--" + reference + "--")

    assert s == reference

    # test ids=CHAR
    s = StringIO()
    debugprint(G, file=s, ids="CHAR")
    s = s.getvalue()
    # The additional white space are needed!
    reference = (
        "\n".join(
            [
                "Elemwise{add,no_inplace} [id A] ''   ",
                " |Elemwise{add,no_inplace} [id B] 'C'   ",
                " | |A [id C]",
                " | |B [id D]",
                " |Elemwise{add,no_inplace} [id E] ''   ",
                "   |D [id F]",
                "   |E [id G]",
            ]
        )
        + "\n"
    )

    if s != reference:
        print("--" + s + "--")
        print("--" + reference + "--")

    assert s == reference

    # test ids=CHAR, stop_on_name=True
    s = StringIO()
    debugprint(G, file=s, ids="CHAR", stop_on_name=True)
    s = s.getvalue()
    # The additional white space are needed!
    reference = (
        "\n".join(
            [
                "Elemwise{add,no_inplace} [id A] ''   ",
                " |Elemwise{add,no_inplace} [id B] 'C'   ",
                " |Elemwise{add,no_inplace} [id C] ''   ",
                "   |D [id D]",
                "   |E [id E]",
            ]
        )
        + "\n"
    )

    if s != reference:
        print("--" + s + "--")
        print("--" + reference + "--")

    assert s == reference

    # test ids=
    s = StringIO()
    debugprint(G, file=s, ids="")
    s = s.getvalue()
    # The additional white space are needed!
    reference = (
        "\n".join(
            [
                "Elemwise{add,no_inplace}  ''   ",
                " |Elemwise{add,no_inplace}  'C'   ",
                " | |A ",
                " | |B ",
                " |Elemwise{add,no_inplace}  ''   ",
                "   |D ",
                "   |E ",
            ]
        )
        + "\n"
    )
    if s != reference:
        print("--" + s + "--")
        print("--" + reference + "--")

    assert s == reference

    # test print_storage=True
    s = StringIO()
    debugprint(g, file=s, ids="", print_storage=True)
    s = s.getvalue()
    # The additional white space are needed!
    reference = (
        "\n".join(
            [
                "Elemwise{add,no_inplace}  ''   0 [None]",
                " |A  [None]",
                " |B  [None]",
                " |D  [None]",
                " |E  [None]",
            ]
        )
        + "\n"
    )
    if s != reference:
        print("--" + s + "--")
        print("--" + reference + "--")

    assert s == reference


def test_subtensor():
    x = dvector()
    y = x[1]
    assert pp(y) == "<TensorType(float64, vector)>[ScalarConstant{1}]"
