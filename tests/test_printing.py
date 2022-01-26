"""
Tests of printing functionality
"""
import logging
from io import StringIO

import pytest

import aesara
from aesara.printing import (
    PatternPrinter,
    PPrinter,
    debugprint,
    default_printer,
    get_node_by_id,
    min_informative_str,
    pp,
    pydot_imported,
    pydotprint,
)
from aesara.tensor.type import dmatrix, dvector, matrix
from tests.graph.utils import MyInnerGraphOp, MyOp, MyVariable


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
    with pytest.raises(TypeError):
        debugprint("blah")

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


def test_debugprint_ids():
    a_at = dvector()
    b_at = dmatrix()

    d_at = b_at.dot(a_at)
    e_at = d_at + a_at

    s = StringIO()
    debugprint(e_at, ids="auto", file=s)
    s = s.getvalue()

    exp_res = f"""Elemwise{{add,no_inplace}} [id {e_at.auto_name}] ''
 |dot [id {d_at.auto_name}] ''
 | |<TensorType(float64, (None, None))> [id {b_at.auto_name}]
 | |<TensorType(float64, (None,))> [id {a_at.auto_name}]
 |<TensorType(float64, (None,))> [id {a_at.auto_name}]
    """

    assert [l.strip() for l in s.split("\n")] == [
        l.strip() for l in exp_res.split("\n")
    ]


def test_pprint():
    x = dvector()
    y = x[1]
    assert pp(y) == "<TensorType(float64, (None,))>[1]"


def test_debugprint_inner_graph():
    r1, r2 = MyVariable("1"), MyVariable("2")
    o1 = MyOp("op1")(r1, r2)
    o1.name = "o1"

    # Inner graph
    igo_in_1 = MyVariable("4")
    igo_in_2 = MyVariable("5")
    igo_out_1 = MyOp("op2")(igo_in_1, igo_in_2)
    igo_out_1.name = "igo1"

    igo = MyInnerGraphOp([igo_in_1, igo_in_2], [igo_out_1])

    r3, r4 = MyVariable("3"), MyVariable("4")
    out = igo(r3, r4)

    output_str = debugprint(out, file="str")
    lines = output_str.split("\n")

    exp_res = """MyInnerGraphOp [id A] ''
 |3 [id B]
 |4 [id C]

Inner graphs:

MyInnerGraphOp [id A] ''
 >op2 [id D] 'igo1'
 > |4 [id E]
 > |5 [id F]
    """

    for exp_line, res_line in zip(exp_res.split("\n"), lines):
        assert exp_line.strip() == res_line.strip()

    # Test nested inner-graph `Op`s
    igo_2 = MyInnerGraphOp([r3, r4], [out])

    r5 = MyVariable("5")
    out_2 = igo_2(r5)

    output_str = debugprint(out_2, file="str")
    lines = output_str.split("\n")

    exp_res = """MyInnerGraphOp [id A] ''
 |5 [id B]

Inner graphs:

MyInnerGraphOp [id A] ''
 >MyInnerGraphOp [id C] ''
 > |3 [id D]
 > |4 [id E]

MyInnerGraphOp [id C] ''
 >op2 [id F] 'igo1'
 > |4 [id G]
 > |5 [id H]
    """

    for exp_line, res_line in zip(exp_res.split("\n"), lines):
        assert exp_line.strip() == res_line.strip()


def test_get_var_by_id():

    r1, r2 = MyVariable("v1"), MyVariable("v2")
    o1 = MyOp("op1")(r1, r2)
    o1.name = "o1"

    # Inner graph
    igo_in_1 = MyVariable("v4")
    igo_in_2 = MyVariable("v5")
    igo_out_1 = MyOp("op2")(igo_in_1, igo_in_2)
    igo_out_1.name = "igo1"

    igo = MyInnerGraphOp([igo_in_1, igo_in_2], [igo_out_1])

    r3 = MyVariable("v3")
    o2 = igo(r3, o1)

    # import aesara; aesara.dprint([o1, o2])
    # op1 [id A] 'o1'
    #  |1 [id B]
    #  |2 [id C]
    # MyInnerGraphOp [id D] ''
    #  |3 [id E]
    #  |op1 [id A] 'o1'
    #
    # Inner graphs:
    #
    # MyInnerGraphOp [id D] ''
    #  >op2 [id F] 'igo1'
    #  > |4 [id G]
    #  > |5 [id H]

    res = get_node_by_id(o1, "blah")

    assert res is None

    res = get_node_by_id([o1, o2], "C")

    assert res == r2

    res = get_node_by_id([o1, o2], "F")

    assert res == igo_out_1.owner


def test_PatternPrinter():
    r1, r2 = MyVariable("1"), MyVariable("2")
    op1 = MyOp("op1")
    o1 = op1(r1, r2)
    o1.name = "o1"

    pprint = PPrinter()
    pprint.assign(op1, PatternPrinter(("|%(0)s - %(1)s|", -1000)))
    pprint.assign(lambda pstate, r: True, default_printer)

    res = pprint(o1)

    assert res == "|1 - 2|"


def test_uncomputed_inputs():
    from tests.link.test_link import OpWithIgnoredInput

    r1, r2 = matrix("r1"), matrix("r2")
    o1 = OpWithIgnoredInput()(r1, r2)

    output_str = debugprint(o1, file="str")
    lines = output_str.split("\n")

    exp_res = """OpWithIgnoredInput [id A] ''
    \\r1 [id B]
    |r2 [id C]
    """

    for exp_line, res_line in zip(exp_res.split("\n"), lines):
        assert exp_line.strip() == res_line.strip()
