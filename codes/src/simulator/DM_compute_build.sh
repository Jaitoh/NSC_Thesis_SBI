cd ~/tmp/NSC/codes/src/simulator
rm -rf build
rm DM_compute.c
rm DM_compute.cpython*.so
rm DM_compute.html
python3 DM_compute_setup.py build_ext --inplace
# /home/wehe/data/conda/envs/sbi/bin/python3 DM_compute_setup.py build_ext --inplace
# rm -rf ./src/simulator/build
# rm ./src/simulator/DM_compute.c
# rm ./src/simulator/DM_compute.cpython*.so
# rm ./src/simulator/DM_compute.html
# python3 ./src/simulator/DM_compute_setup.py build_ext --inplace

# export LDFLAGS="-L/usr/local/opt/libomp/lib"
# export CPPFLAGS="-I/usr/local/opt/libomp/include"
