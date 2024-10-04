# LIME Toolbox

![Version 1.0.3](https://img.shields.io/badge/version-1.0.3-informational)
[![License: LGPL v3](https://img.shields.io/badge/License-LGPL_v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)

The *lime_tbx* is a Python package that provides a comprehensive toolbox
for utilizing the Lunar Irradiance Model of ESA (LIME) to simulate lunar
observations and compare them with remote sensing data of the Moon.


<style>
.center {
  display: block;
  margin-left: auto;
  margin-right: auto;
  width: 50%;
}
.column {
  float: left;
  width: 30%;
  padding: 5px;
}
.row::after {
  content: "";
  clear: both;
  display: table;
}
</style>

<img src="../images/lime_logo.png" alt="LIME logo" style="width:250px;" class="center"/>

This project is managed, financed and supported by the European Space
Agency (ESA).


<img src="../images/esa.png" alt="ESA logo" style="width:250px;" class="center"/>

More information about the LIME can be found on [lime.uva.es](https://lime.uva.es>).


Documentation
=============

Documentation is available on: https://eco.gitlab-docs.npl.co.uk/eo/lime_tbx/


Usage
=====

Installation
------------

The LIME TBX is designed as a standalone application, but you can
install the python package and its dependencies by using::

    pip install -e .


Development
-----------

For developing the package, you'll want to install the pre-commit
hooks as well. Type::

    pre-commit install

Note that from now on when you commit, `black` will check your code for styling
errors. If it finds any it will correct them, but the commit will be aborted.
This is so that you can check its work before you continue. If you're happy,
just commit again.

The project dependencies can be installed with::

    pip install -r requirements.txt

The project composite structure is explained in quality
documentation/uml/composite_structure.png:

<a href="../../quality_documentation/uml/composite_structure.png">
  <img src="../../quality_documentation/uml/composite_structure.png"
alt="UML diagram of the composite structure of lime_tbx" style="width:1000px;" class="center"/>
</a>


Testing
-------

To perform the unit tests one must run::

    python3 -m unittest

To perform the coverage tests one must type::

    ./coverage_run.sh


Deployment
----------

Requirements:

- python 3.8 (Linux) or python 3.9 (Mac and Windows).
- pyinstaller (installed outside of the virtual environment).

It is strongly recommended to create the app-bundle using a virtual
environment (venv) in order to minimize the application size. Inside of
it, the project dependencies can be installed with::

    pip install -r requirements.txt

Create a desktop app-bundle for your OS by using::

    pyinstaller lime_tbx.spec

Now you can deactivate the virtual environment. You may create an installer
for your OS under the installer directory. For Windows you must use
"InnoSetup", for Mac and Linux you must execute the scripts
"build_mac_installer.sh" and "build_linux_installer.sh" respectively,
and for Debian you must execute the "build_deb.sh" script.

For more information about the recommended environments for the production
of TBX binaries please check the installer directory.


Compatibility
-------------

- Windows 10 with x86_64 arch.
- Linux with GLIBC >= 2.23 and x86_64 arch.
- Mac with x86_64 arch. or with ARM64 arch. and Rosetta interpreter.

License
-------

[LGPL v3](../../LICENSE)

Authors
-------

* [Javier Gatón Herguedas](gaton@goa.uva.es) - [GOA-UVa](https://goa.uva.es)
* [Pieter De Vis](pieter.de.vis@npl.co.uk) - [NPL](https://npl.co.uk)
* [Stefan Adriaensen](stefan.adriaensen@vito.be) - [VITO](https://vito.be)
* [Jacob Fahy](jacob.fahy@npl.co.uk) - [NPL](https://npl.co.uk)
* [Ramiro González Catón](ramiro@goa.uva.es) - [GOA-UVa](https://goa.uva.es)
* [Carlos Toledano](toledano@goa.uva.es) - [GOA-UVa](https://goa.uva.es)

<div class="row">
  <div class="column">
    <img src="../images/uva_sello.png" alt="Logo of UVa" style="width:100%">
  </div>
  <div class="column">
    <img src="../images/npl.png" alt="Logo of NPL" style="width:100%">
  </div>
  <div class="column">
    <img src="../images/vito.png" alt="Logo of VITO" style="width:100%">
  </div>
</div>
