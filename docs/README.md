# Documentation

## Building documentation

To build documentation first generate the API docs using `autodoc`. From the present directory execute
```sh
sphinx-apidoc -o source ../dingo
```
This will create `dingo.*.rst.` files in `source/` corresponding to the various modules.

Next, the main docs can be generated using
```sh
make html
```
This creates a directory `build/` containing HTML doc pages. The main index is at [build/html/index.html](build/html/index.html).

### Cleanup

To remove generated docs, execute
```sh
make clean
rm source/dingo.* source/modules.rst
```
