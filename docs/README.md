# Docs

Adding documentation for a module:
* All modules are located in `../api`
* When creating a new module, add this line to `conf.py`: `sys.path.insert(0, os.path.abspath('../src/MODULE_FOLDER'))`
* Build docs by running `sphinx-apidoc -o src/MODULE_FOLDER ../api/MODULE_FOLDER` (destination and source) then `make clean && make html`, both from the `docs` folder

To change the format of the table of contents on the docs webpage, mess with the `.rst` files in `docs/src` and in `docs/index.rst`.
