Aesara Mission Statement
========================

Introduction
------------

We, the Aesara Core Team ("we"), have created this mission statement to provide
users and developers of the Aesara Project (the "Project") with a clear
description of the Project's goals and guiding principles. The Project includes
multiple software libraries including Aesara, AeHMC, AeMCMC, and AePPL, so any
references to the Project apply to these libraries jointly.

Background
----------

The overall goal of the Aesara Project is to provide a rich, Python-first
toolset for the symbolic representation, manipulation, and compilation of
mathematical expressions involving arrays.

More specifically, we want Aesara to facilitate the prototyping and development
of higher-level reasoning and automation within mathematical modeling
disciplines such as statistical modeling and machine learning.  We also want
Aesara to defer productively to more specialized external libraries for
specialized and lower-level computations.

In general, three basic principles guide our work:

* Modular design
* Composition of different ecosystems
* Separation of concerns

Project Goals
-------------

The higher-level goals of the Aesara Project are:

* Produce code that is hackable in pure-Python. This includes Aesara’s graph
  elements and compilation process.
* Reduce the development overhead between high-level prototyping and the
  construction of computable expressions.
* Enable expressions represented in Aesara to be lowered to multiple compilation
  targets, and make lowering highly customizable.
* Make computable representations of mathematics easier to construct, and
  facilitate connections with symbolic mathematics.
* Facilitate the development of domain-specific compilers by domain experts who
  are comfortable programming in Python.
* Provide tools to implement “meta”-level logic that includes a framework and
  base implementations for defining and executing arbitrary rewrites of Aesara
  expressions.
* Offer the ability to choose between computational stability, speed, and any
  other custom metric during compilation.
* Avoid unnecessarily restrictive choices that prevent reasonable uses of the
  Project that were not foreseen by the developers.
* Use our work to demonstrate the utility of symbolic reasoning outside of its
  already established uses within the Python community.
* Establish a diverse group of contributors who share similar interests and
  provide unique perspectives on how we can accomplish these goals.

The Project also commits to producing high-quality software by

* prioritizing testing and its automation;
* maintaining a well-defined and tested feature surface with a higher weight
  towards stability and the improvement of existing functionality over adding new
  features;
* prioritizing design consistency, simplicity, and the ability to extend core
  functionality;
* prioritizing work on core bugs, performance, and stability so that others can
  reliably build upon the Project.
