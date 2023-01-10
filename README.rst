========
lime_tbx
========

The LIME TBX is a python package with a toolbox for using the LIME (Lunar Irradiance Model of ESA) model
to simulate lunar observations and compare to remote sensing observations of the moon.

.. image:: https://img.shields.io/badge/version-0.0.5-informational
  :alt: Version 0.0.5


Documentation
=============

Documentation is available on: https://eco.gitlab-docs.npl.co.uk/eo/lime_tbx/


Usage
=====

Installation
------------

The LIME TBX is designed as a standalone application, but you can install the python package and its dependencies by using::

    pip install -e .


Development
-----------

For developing the package, you'll want to install the pre-commit hooks as well. Type::

    pre-commit install

Note that from now on when you commit, `black` will check your code for styling
errors. If it finds any it will correct them, but the commit will be aborted.
This is so that you can check its work before you continue. If you're happy,
just commit again.

The project dependencies can be installed with::

    pip install -r requirements.txt


Testing
-------

To perform the unit tests one must type::

    python3 -m unittest

To perform the coverage tests one must type::

    ./coverage_run.sh


Deployment
----------

Requirements:

- python 3.8 (Windows and Linux) or python 3.9 (Mac).
- pyinstaller (installed outside of the virtual environment).

It is strongly recommended to create the app-bundle using a virtual environment (venv). Inside of it,
the project dependencies can be installed with::

    pip install -r requirements.txt

Create a desktop app-bundle for your OS by using::
  
    pyinstaller lime_tbx.spec

Now you can deactivate the virtual environment. You may create an installer for your OS under the installer directory.
For Windows you must use "InnoSetup", for Mac and Linux you must execute the scripts "build_mac_installer.sh"
and "build_linux_installer.sh" respectively, and for Debian you must execute the "build_deb.sh" script.


Compatibility
-------------

Licence
-------

Authors
-------

`Pieter De Vis <pieter.de.vis@npl.co.uk>`_.
`Jacob Fahy <jacob.fahy@npl.co.uk>`_.
`Javier Gatón Herguedas <gaton@goa.uva.es>`_.
`Ramiro González Catón <ramiro@goa.uva.es>`_.
`Carlos Toledano <toledano@goa.uva.es>`_.
