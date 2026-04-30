cmake_minimum_required(VERSION 3.24)
project(moss-tts-cli LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

include(FetchContent)

# FIX: Bypass the version compatibility error for SentencePiece
set(CMAKE_POLICY_DEFAULT_CMP0000 NEW)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

# --- ONNX Runtime Download ---
if(WIN32)
    set(ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/v1.17.1/onnxruntime-win-x64-1.17.1.zip")
elseif(APPLE)
    set(ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/v1.17.1/onnxruntime-osx-universal2-1.17.1.tgz")
else()
    set(ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/v1.17.1/onnxruntime-linux-x64-1.17.1.tgz")
endif()

FetchContent_Declare(onnxruntime URL ${ORT_URL})
FetchContent_MakeAvailable(onnxruntime)

# --- SentencePiece Download ---
FetchContent_Declare(
  sentencepiece
  GIT_REPOSITORY https://github.com/google/sentencepiece.git
  GIT_TAG        v0.2.0
)
set(SPM_ENABLE_SHARED OFF CACHE BOOL "" FORCE)
set(SPM_ENABLE_TCMALLOC OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(sentencepiece)

# --- Locate Files ---
file(GLOB_RECURSE ORT_INC_DIR "${onnxruntime_SOURCE_DIR}/onnxruntime_cxx_api.h")
get_filename_component(ORT_INC_DIR "${ORT_INC_DIR}" DIRECTORY)

file(GLOB ORT_LIBS "${onnxruntime_SOURCE_DIR}/lib/*.lib" "${onnxruntime_SOURCE_DIR}/lib/*.so" "${onnxruntime_SOURCE_DIR}/lib/*.dylib")
list(FILTER ORT_LIBS INCLUDE REGEX ".*\\.lib$|.*\\.so$|.*\\.dylib$")

# --- Build Target ---
add_executable(moss-tts-nano src/main.cpp)

target_include_directories(moss-tts-nano PRIVATE
    ${ORT_INC_DIR}
    ${sentencepiece_SOURCE_DIR}/src  # Crucial for finding sentencepiece headers
)

target_link_libraries(moss-tts-nano PRIVATE
    ${ORT_LIBS}
    sentencepiece-static
)

# --- Probe Model Tool ---
add_executable(probe_model src/probe_model.cpp)
target_include_directories(probe_model PRIVATE ${ORT_INC_DIR})
target_link_libraries(probe_model PRIVATE ${ORT_LIBS})

# --- Post-Build (Windows DLL Setup) ---
if(WIN32)
    file(GLOB_RECURSE ORT_DLLS "${onnxruntime_SOURCE_DIR}/*.dll")
    foreach(DLL_PATH ${ORT_DLLS})
        add_custom_command(TARGET moss-tts-nano POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DLL_PATH}" $<TARGET_FILE_DIR:moss-tts-nano>
            COMMENT "Copying ${DLL_PATH} to output directory"
        )
    endforeach()
endif()