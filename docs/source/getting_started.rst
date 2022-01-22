Development Installation 
------------------------

To install dingo, along with the tools for development and testing, do the following:

Create and activate a virtual environment. By convention, the environment is called `venv` and is located in the dingo-devel directory. Some unit tests depend on this convention.


.. code-block:: console

   python3 -m venv venv
   source venv/bin/activate 

In this virtual environment, install dingo. 

.. code-block:: console 

    pip install wheel
    python setup.py bdist_wheel
    pip install -e ."[dev]"


