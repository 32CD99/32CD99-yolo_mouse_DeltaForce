 cmake_minimum_required(VERSION 3.23)
project(dd_trt_yolo_demo LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- EDIT THESE PATHS ----
set(TENSORRT_DIR "D:/NVIDIA/TensorRT-10.14.1.48")
set(CUDA_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8")

# 4060 Ti = Ada SM89
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)

include_directories("${TENSORRT_DIR}/include")
include_directories("${CUDA_DIR}/include")




# Use static MSVC runtime: Release=/MT, Debug=/MTd
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")





add_executable(dd_trt_yolo WIN32
  src/main.cpp
  src/dxgi_dup.cpp
  src/trt_runner.cpp
  src/postprocess.cpp
  src/preprocess.cu
  src/mouse_controller.cpp
  src/roi_debugger.cpp
  src/fa.cpp
 src/backend_worker.cpp
 src/tracking.cpp
 src/cuda_utils.cpp
 src/ui_main.h 
 src/ui_main.cpp )

link_directories("${TENSORRT_DIR}/lib")
link_directories("${CUDA_DIR}/lib/x64")

function(link_trt target)
  foreach(name IN ITEMS nvinfer_10 nvinfer)
    find_library(L_${name} NAMES ${name} PATHS "${TENSORRT_DIR}/lib" NO_DEFAULT_PATH)
    if(L_${name})
      target_link_libraries(${target} PRIVATE ${L_${name}})
      break()
    endif()
  endforeach()

  foreach(name IN ITEMS nvinfer_plugin_10 nvinfer_plugin)
    find_library(L_${name} NAMES ${name} PATHS "${TENSORRT_DIR}/lib" NO_DEFAULT_PATH)
    if(L_${name})
      target_link_libraries(${target} PRIVATE ${L_${name}})
      break()
    endif()
  endforeach()
endfunction()

link_trt(dd_trt_yolo)

find_library(CUDART_LIB NAMES cudart PATHS "${CUDA_DIR}/lib/x64" NO_DEFAULT_PATH)
if(NOT CUDART_LIB)
  message(FATAL_ERROR "Could not find cudart.lib under CUDA_DIR/lib/x64. Check CUDA_DIR.")
endif()
target_link_libraries(dd_trt_yolo PRIVATE ${CUDART_LIB})

target_link_libraries(dd_trt_yolo PRIVATE d3d11 dxgi user32)
target_link_libraries(dd_trt_yolo PRIVATE comctl32 user32 gdi32)
target_link_libraries(dd_trt_yolo PRIVATE setupapi hid)



set_target_properties(dd_trt_yolo PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_definitions(dd_trt_yolo PRIVATE
  _CRT_SECURE_NO_WARNINGS
  NOMINMAX
  UNICODE
  _UNICODE
)

