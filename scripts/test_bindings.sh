#!/usr/bin/env bash
case $(pwd) in *scripts) cd ..;; esac
if ! command -v cargo &> /dev/null
then
    echo "Rust is not installed, please install it first."
    echo "https://www.rust-lang.org/tools/install"
    exit
fi

if ! command -v clang &> /dev/null
then
    c_compiler="gcc"
    cxx_compiler="g++"
else
    c_compiler="clang"
    cxx_compiler="clang++"
fi

cargo build --release
$c_compiler -Wall -O2 -o c_test ./bindings/tests/test.c ./target/release/libstonnx_api.so
$cxx_compiler -Wall -O2 -std=c++20 -o cpp_test ./bindings/tests/test.cpp ./target/release/libstonnx_api.so
cp ./bindings/py/stonnx.py .
cp ./bindings/tests/test.py .
cp ./target/release/libstonnx_api.so .
echo "Running C test..."
./c_test &> /dev/null
if [ $? -eq 0 ]; then
    echo "C test passed."
else
    echo "C test failed."
fi
echo "Running C++ test..."
./cpp_test &> /dev/null
if [ $? -eq 0 ]; then
    echo "C++ test passed."
else
    echo "C++ test failed."
fi
echo "Running Python test..."
python3 test.py &> /dev/null
if [ $? -eq 0 ]; then
    echo "Python test passed."
else
    echo "Python test failed."
fi
rm c_test cpp_test
rm ./stonnx.py
rm ./test.py
rm ./*.so
