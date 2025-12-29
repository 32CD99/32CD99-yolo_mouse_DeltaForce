#include "trt_runner.h"
#include <NvInferPlugin.h>

#include <fstream>
#include <iostream>

void TrtRunner::Logger::log(Severity severity, const char* msg) noexcept {
  if (severity <= Severity::kWARNING) {
    std::cout << "[TRT] " << msg << "\n";
  }
}

static std::vector<char> read_all(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("Cannot open engine: " + path);
  f.seekg(0, std::ios::end);
  size_t n = (size_t)f.tellg();
  f.seekg(0, std::ios::beg);
  std::vector<char> buf(n);
  f.read(buf.data(), (std::streamsize)n);
  return buf;
}

static size_t vol(const nvinfer1::Dims& d) {
  size_t v = 1;
  for (int i = 0; i < d.nbDims; i++) v *= (size_t)d.d[i];
  return v;
}

bool TrtRunner::load_engine(const std::string& path)
{
  shutdown();

  try { m_blob = read_all(path); }
  catch (const std::exception& e) { std::cerr << e.what() << "\n"; return false; }

  initLibNvInferPlugins(&m_logger, "");

  m_runtime = nvinfer1::createInferRuntime(m_logger);
  if (!m_runtime) return false;

  m_engine = m_runtime->deserializeCudaEngine(m_blob.data(), m_blob.size());
  if (!m_engine) return false;

  m_ctx = m_engine->createExecutionContext();
  if (!m_ctx) return false;

  int nb = m_engine->getNbIOTensors();
  std::cout << "[TRT] NbIOTensors=" << nb << "\n";
  for (int i = 0; i < nb; i++) {
    const char* name = m_engine->getIOTensorName(i);
    auto mode = m_engine->getTensorIOMode(name);
    auto shape = m_engine->getTensorShape(name);
    auto dtype = m_engine->getTensorDataType(name);
    std::cout << (mode == nvinfer1::TensorIOMode::kINPUT ? "INPUT " : "OUTPUT ")
              << "name=" << name << " dims=[";
    for (int k = 0; k < shape.nbDims; k++) {
      std::cout << shape.d[k] << (k + 1 < shape.nbDims ? "x" : "");
    }
    std::cout << "] dtype=" << (int)dtype << "\n";
  }

  m_inDims = m_engine->getTensorShape(input_name());
  m_outDims = m_engine->getTensorShape(output_name());
  m_inType = m_engine->getTensorDataType(input_name());
  m_outType = m_engine->getTensorDataType(output_name());

  // If dynamic, user must set; we attempt to set now for static/dynamic.
  if (!m_ctx->setInputShape(input_name(), m_inDims)) {
    std::cerr << "[TRT] setInputShape failed for " << input_name() << "\n";
  }
  if (!m_ctx->allInputDimensionsSpecified()) {
    std::cerr << "[TRT] input dimensions not fully specified\n";
  }

  m_inElems = vol(m_inDims);
  m_outElems = vol(m_outDims);

  if (m_inDims.nbDims == 4) m_roi = m_inDims.d[2];

  if (!alloc_buffers()) return false;

  std::cout << "[TRT] engine loaded OK. roi=" << m_roi
            << " inElems=" << m_inElems
            << " outElems=" << m_outElems << "\n";
  return true;
}

bool TrtRunner::alloc_buffers()
{
  if (m_inElems == 0 || m_outElems == 0) return false;

  auto typeSize = [](nvinfer1::DataType t)->size_t{
    switch(t){
      case nvinfer1::DataType::kFLOAT: return 4;
      case nvinfer1::DataType::kHALF:  return 2;
      case nvinfer1::DataType::kINT8:  return 1;
      case nvinfer1::DataType::kINT32: return 4;
      case nvinfer1::DataType::kINT64: return 8;
      case nvinfer1::DataType::kBOOL:  return 1;
      default: return 4;
    }
  };

  size_t inBytes = m_inElems * typeSize(m_inType);
  size_t outBytes = m_outElems * typeSize(m_outType);

  if (cudaMalloc(&m_dInput, inBytes) != cudaSuccess) return false;
  if (cudaMalloc(&m_dOutput, outBytes) != cudaSuccess) return false;
  return true;
}

bool TrtRunner::infer(cudaStream_t stream)
{
  if (!m_ctx) return false;

  if (!m_ctx->setTensorAddress(input_name(), m_dInput)) return false;
  if (!m_ctx->setTensorAddress(output_name(), m_dOutput)) return false;

  return m_ctx->enqueueV3(stream);
}

void TrtRunner::shutdown()
{
  if (m_dOutput) { cudaFree(m_dOutput); m_dOutput = nullptr; }
  if (m_dInput)  { cudaFree(m_dInput);  m_dInput  = nullptr; }

  if (m_ctx)     { delete m_ctx;     m_ctx = nullptr; }
  if (m_engine)  { delete m_engine;  m_engine = nullptr; }
  if (m_runtime) { delete m_runtime; m_runtime = nullptr; }

  m_blob.clear();
}
