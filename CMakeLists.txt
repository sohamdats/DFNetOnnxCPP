cmake_minimum_required(VERSION 3.28)
project(audioCPPNoiseRemoval)

set(CMAKE_CXX_STANDARD 17)
#execute_process(
#        COMMAND python -c "import torch;print(torch.utils.cmake_prefix_path)"
#        OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
#        OUTPUT_STRIP_TRAILING_WHITESPACE
#)

# Append the paths obtained from the Python script to CMAKE_PREFIX_PATH
#list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX_PATH})
# find_package(Torch REQUIRED)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(audioCPPNoiseRemoval main.cpp)
target_include_directories(audioCPPNoiseRemoval PUBLIC ${TORCH_INCLUDE_DIRS})
# Link libraries
target_link_libraries(audioCPPNoiseRemoval PUBLIC ${TORCH_LIBRARIES})

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if(APPLE)
    if(DEFINED ENV{ONNXRUNTIME_ROOT})
        set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/externals/$ENV{ONNXRUNTIME_ROOT}")
    else()
        set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/externals/onnxruntime-osx-arm64-1.17.3")
    endif()
elseif(WIN32)
    set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/externals/onnxruntime-win-x64-1.12.1")
endif()

find_package(ONNXRUNTIME REQUIRED)
target_link_libraries(audioCPPNoiseRemoval PUBLIC ${ONNXRUNTIME_LIBRARY})
target_include_directories(audioCPPNoiseRemoval PUBLIC ${ONNXRUNTIME_INCLUDE_DIR})

if(WIN32)
    add_custom_command(TARGET ${APP_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
            "${CMAKE_CURRENT_SOURCE_DIR}/externals/onnxruntime-win-x64-1.12.1/lib/onnxruntime.dll"
            "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime.dll"
    )
endif()

find_package(SndFile REQUIRED)
target_link_libraries(audioCPPNoiseRemoval PUBLIC ${SndFile_LIBRARIES})
target_include_directories(audioCPPNoiseRemoval PUBLIC ${SndFile_INCLUDE_DIRS})

# find_package(PkgConfig REQUIRED)
# pkg_check_modules(LIBSAMPLERATE REQUIRED samplerate)
# target_link_directories(audioCPPNoiseRemoval PUBLIC ${LIBSAMPLERATE_INCLUDE_DIRS})
# target_link_libraries(audioCPPNoiseRemoval PUBLIC ${LIBSAMPLERATE_LIBRARIES})