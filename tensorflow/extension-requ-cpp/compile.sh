rm -rf build
mkdir build
cd build
cmake ..
make

cd ..
python requ_manual_test.py 