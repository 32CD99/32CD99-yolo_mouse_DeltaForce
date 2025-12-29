#pragma once
#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <string>
#include <vector>

class TrtRunner {
public:
  bool load_engine(const std::string& path);
  void shutdown();

  int roi() const { return m_roi; }

  void* d_input() const { return m_dInput; }
  void* d_output() const { return m_dOutput; }
  size_t output_elems() const { return m_outElems; }
  nvinfer1::DataType input_dtype() const { return m_inType; }
  nvinfer1::DataType output_dtype() const { return m_outType; }

  const char* input_name() const { return "images"; }
  const char* output_name() const { return "output0"; }

  bool infer(cudaStream_t stream);

private:
  struct Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override;
  } m_logger;

  nvinfer1::IRuntime* m_runtime = nullptr;
  nvinfer1::ICudaEngine* m_engine = nullptr;
  nvinfer1::IExecutionContext* m_ctx = nullptr;

  void* m_dInput = nullptr;
  void* m_dOutput = nullptr;

  int m_roi = 640;
  size_t m_inElems = 0;
  size_t m_outElems = 0;
  std::vector<char> m_blob;

  nvinfer1::Dims m_inDims{};
  nvinfer1::Dims m_outDims{};

  nvinfer1::DataType m_inType = nvinfer1::DataType::kFLOAT;
  nvinfer1::DataType m_outType = nvinfer1::DataType::kFLOAT;

  bool alloc_buffers();
};
