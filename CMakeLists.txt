cmake_minimum_required(VERSION 3.28)
project(audioCPPNoiseRemoval)

set(CMAKE_CXX_STANDARD 17)


add_executable(audioCPPNoiseRemoval main.cpp)

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