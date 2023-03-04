# export LDFLAGS="-L/usr/local/opt/libomp/lib"
# export CPPFLAGS="-I/usr/local/opt/libomp/include"

rm -rf build
rm DM_compute.c
rm DM_compute.cpython*.so
rm DM_compute.html
python setup.py build_ext --inplace