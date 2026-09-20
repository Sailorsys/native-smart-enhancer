cmake_minimum_required(VERSION 3.22.1)
project("smart_enhancer")

set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/opencv-android-sdk/sdk/native/jni)
find_package(OpenCV REQUIRED COMPONENTS core imgproc)

add_library(smart_enhancer SHARED smart_enhancer.cpp)

target_link_libraries(
    smart_enhancer
    ${OpenCV_LIBS}
    jnigraphics
    log
)

