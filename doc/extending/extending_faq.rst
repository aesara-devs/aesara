
.. _extend_faq:

=========================================
Extending Aesara: FAQ and Troubleshooting
=========================================

I wrote a new `Op`\/`Type`, and weird stuff is happening...
-----------------------------------------------------------

First, check the :ref:`op_contract` and the :ref:`type_contract`
and make sure you're following the rules.
Then try running your program in :ref:`using_debugmode`.  `DebugMode` might catch
something that you're not seeing.


I wrote a new rewrite, but it's not getting used...
---------------------------------------------------

Remember that you have to register rewrites with the :ref:`optdb`
for them to get used by the normal modes like FAST_COMPILE, FAST_RUN,
and `DebugMode`.


I wrote a new rewrite, and it changed my results even though I'm pretty sure it is correct.
-------------------------------------------------------------------------------------------

First, check the :ref:`op_contract` and make sure you're following the rules.
Then try running your program in :ref:`using_debugmode`.  `DebugMode` might
catch something that you're not seeing.
