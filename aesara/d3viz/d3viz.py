"""Dynamic visualization of Aesara graphs.

Author: Christof Angermueller <cangermueller@gmail.com>
"""


import json
import os
import shutil

from aesara.d3viz.formatting import PyDotFormatter


__path__ = os.path.dirname(os.path.realpath(__file__))


def replace_patterns(x, replace):
    """Replace `replace` in string `x`.

    Parameters
    ----------
    s : str
        String on which function is applied
    replace : dict
        `key`, `value` pairs where key is a regular expression and `value` a
        string by which `key` is replaced
    """
    for from_, to in replace.items():
        x = x.replace(str(from_), str(to))
    return x


def safe_json(obj):
    """Encode `obj` to JSON so that it can be embedded safely inside HTML.

    Parameters
    ----------
    obj : object
        object to serialize
    """
    return json.dumps(obj).replace("<", "\\u003c")


def d3viz(fct, outfile, copy_deps=True, *args, **kwargs):
    """Create HTML file with dynamic visualizing of an Aesara function graph.

    In the HTML file, the whole graph or single nodes can be moved by drag and
    drop. Zooming is possible via the mouse wheel. Detailed information about
    nodes and edges are displayed via mouse-over events. Node labels can be
    edited by selecting Edit from the context menu.

    Input nodes are colored in green, output nodes in blue. Apply nodes are
    ellipses, and colored depending on the type of operation they perform. Red
    ellipses are transfers from/to the GPU (ops with names GpuFromHost,
    HostFromGpu).

    Edges are black by default. If a node returns a view of an
    input, the input edge will be blue. If it returns a destroyed input, the
    edge will be red.

    Parameters
    ----------
    fct : aesara.compile.function.types.Function
        A compiled Aesara function, variable, apply or a list of variables.
    outfile : str
        Path to output HTML file.
    copy_deps : bool, optional
        Copy javascript and CSS dependencies to output directory.

    Notes
    -----
    This function accepts extra parameters which will be forwarded to
    :class:`aesara.d3viz.formatting.PyDotFormatter`.

    """

    # Create DOT graph
    formatter = PyDotFormatter(*args, **kwargs)
    graph = formatter(fct)
    dot_graph = graph.create_dot()
    dot_graph = dot_graph.decode("utf8")

    # Create output directory if not existing
    outdir = os.path.dirname(outfile)
    if outdir != "" and not os.path.exists(outdir):
        os.makedirs(outdir)

    # Read template HTML file
    template_file = os.path.join(__path__, "html", "template.html")
    with open(template_file) as f:
        template = f.read()

    # Copy dependencies to output directory
    src_deps = __path__
    if copy_deps:
        dst_deps = "d3viz"
        for d in ("js", "css"):
            dep = os.path.join(outdir, dst_deps, d)
            if not os.path.exists(dep):
                shutil.copytree(os.path.join(src_deps, d), dep)
    else:
        dst_deps = src_deps

    # Replace patterns in template
    replace = {
        "%% JS_DIR %%": os.path.join(dst_deps, "js"),
        "%% CSS_DIR %%": os.path.join(dst_deps, "css"),
        "%% DOT_GRAPH %%": safe_json(dot_graph),
    }
    html = replace_patterns(template, replace)

    # Write HTML file
    with open(outfile, "w") as f:
        f.write(html)


def d3write(fct, path, *args, **kwargs):
    """Convert Aesara graph to pydot graph and write to dot file.

    Parameters
    ----------
    fct : aesara.compile.function.types.Function
        A compiled Aesara function, variable, apply or a list of variables.
    path: str
        Path to output file

    Notes
    -----
    This function accepts extra parameters which will be forwarded to
    :class:`aesara.d3viz.formatting.PyDotFormatter`.

    """

    formatter = PyDotFormatter(*args, **kwargs)
    graph = formatter(fct)
    graph.write_dot(path)
