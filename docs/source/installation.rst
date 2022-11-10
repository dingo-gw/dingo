Installation
============

Development Installation
------------------------

To install dingo, along with the tools for development and testing, do the following:

Create and activate a virtual environment. By convention, the environment is called `venv`` and is located in the `dingo-devel` directory. Some unit tests depend on this convention.

.. code-block:: console 

    python3 -m venv venv
    source venv/bin/activate

In this virtual environment, install dingo

.. code-block:: console
 
    pip install wheel
    python setup.py bdist_wheel
    pip install -e ."[dev]"


Documentation Instructions
--------------------------

To build documentation navigate to the `dingo-devel` directory and run 

.. code-block:: console 

    sphinx-build -b html docs/source/ docs/build/

If you add new modules you can also autopopulate .rst files for each module by running

.. code-block:: console

    sphinx-apidoc -o docs/source dingo
    
from the `dingo-devel` directory

To use the autodoc feature (works for pycharm and numpy docstrings) in a .rst file run 


`.. autofunction:: dingo.core.utils.trainutils.write_history`

This will render like so: 

.. autofunction:: dingo.core.utils.trainutils.write_history