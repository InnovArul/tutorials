cmake_minimum_required(VERSION 2.8)

# tensorflow include dirs
execute_process(COMMAND python -c "import tensorflow as tf; print(tf.sysconfig.get_include())" OUTPUT_VARIABLE Tensorflow_INCLUDE_DIRS OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND python -c "import tensorflow as tf; print(tf.sysconfig.get_lib())" OUTPUT_VARIABLE Tensorflow_INCLUDE_LIBRARY OUTPUT_STRIP_TRAILING_WHITESPACE)

execute_process(COMMAND python -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_compile_flags()))" OUTPUT_VARIABLE Tensorflow_CFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND python -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_link_flags()))" OUTPUT_VARIABLE Tensorflow_LFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)

# C++11 required for tensorflow
set(CMAKE_CXX_FLAGS "-std=c++11 -fPIC -O2 -Wl,--no-as-needed ${Tensorflow_CFLAGS} ${Tensorflow_LFLAGS} ${CMAKE_CXX_FLAGS}")
message(${CMAKE_CXX_FLAGS})

include_directories(${Tensorflow_INCLUDE_DIRS})
add_library(requ SHARED requ.cc)
add_library(requ_grad SHARED requ_grad.cc)