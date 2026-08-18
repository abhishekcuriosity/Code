#define NOMINMAX
#include <onnxruntime_cxx_api.h>
#include <windows.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace fs = std::filesystem;

static fs::path ExeDir()
{
    std::wstring buf(32768, L'\0');
    DWORD n = GetModuleFileNameW(nullptr, buf.data(), static_cast<DWORD>(buf.size()));
    if (n == 0 || n >= buf.size()) throw std::runtime_error("GetModuleFileNameW failed");
    buf.resize(n);
    return fs::path(buf).parent_path();
}

static std::vector<float> ReadFloats(const fs::path& p, size_t count)
{
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + p.string());
    std::vector<float> v(count);
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(count * sizeof(float)));
    if (f.gcount() != static_cast<std::streamsize>(count * sizeof(float)))
        throw std::runtime_error("Unexpected file size: " + p.string());
    return v;
}

static float MaxError(const std::vector<float>& a, const std::vector<float>& b)
{
    float e = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) e = std::max(e, std::abs(a[i] - b[i]));
    return e;
}

struct Result {
    std::vector<float> output;
    double median_us;
};

static Result RunCase(
    Ort::Env& env,
    const fs::path& model,
    const fs::path& optimized_model,
    GraphOptimizationLevel level,
    bool save_optimized,
    std::vector<float>& input)
{
    Ort::SessionOptions so;
    so.SetIntraOpNumThreads(1);
    so.SetGraphOptimizationLevel(level);
    if (save_optimized) so.SetOptimizedModelFilePath(optimized_model.c_str());

    Ort::Session session(env, model.c_str(), so);
    Ort::AllocatorWithDefaultOptions allocator;

    auto in_name = session.GetInputNameAllocated(0, allocator);
    auto out_name = session.GetOutputNameAllocated(0, allocator);

    std::cout << "Input  : " << in_name.get() << "\n";
    std::cout << "Output : " << out_name.get() << "\n";

    std::array<int64_t,4> in_shape{1,1,8,8};
    std::array<int64_t,2> out_shape{1,3};

    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    auto in_tensor = Ort::Value::CreateTensor<float>(
        mem, input.data(), input.size(), in_shape.data(), in_shape.size());

    std::vector<float> output(3, 0.0f);
    auto out_tensor = Ort::Value::CreateTensor<float>(
        mem, output.data(), output.size(), out_shape.data(), out_shape.size());

    const char* ins[] = { in_name.get() };
    const char* outs[] = { out_name.get() };

    session.Run(Ort::RunOptions{nullptr}, ins, &in_tensor, 1, outs, &out_tensor, 1);

    constexpr int warmup = 1000;
    for (int i = 0; i < warmup; ++i)
        session.Run(Ort::RunOptions{nullptr}, ins, &in_tensor, 1, outs, &out_tensor, 1);

    constexpr int trials = 7;
    constexpr int iterations = 10000;
    std::vector<double> times;

    for (int t = 0; t < trials; ++t) {
        auto s = std::chrono::steady_clock::now();
        for (int i = 0; i < iterations; ++i)
            session.Run(Ort::RunOptions{nullptr}, ins, &in_tensor, 1, outs, &out_tensor, 1);
        auto e = std::chrono::steady_clock::now();

        double us = std::chrono::duration<double, std::micro>(e - s).count() / iterations;
        times.push_back(us);
        std::cout << "Trial " << (t+1) << " : "
                  << std::fixed << std::setprecision(3)
                  << us << " us/inference\n";
    }

    std::sort(times.begin(), times.end());
    return { output, times[times.size()/2] };
}

int main()
{
    try {
        std::cout << "============================================================\n";
        std::cout << "Qualcomm Edge ML - Experiment 08\n";
        std::cout << "ONNX -> ONNX Runtime CPU EP -> Oryon\n";
        std::cout << "============================================================\n";

#ifdef _M_ARM64
        std::cout << "Build architecture : ARM64\n";
#else
        std::cout << "Build architecture : NOT ARM64\n";
#endif

        std::cout << "ONNX Runtime       : "
            << Ort::GetVersionString()
            << "\n";
        std::cout << "Execution Provider : CPU\n";

        fs::path dir = ExeDir();
        fs::path model = dir / L"tiny_cnn_fp32.onnx";
        fs::path optimized = dir / L"tiny_cnn_optimized.onnx";

        auto input = ReadFloats(dir / L"input.bin", 64);
        auto golden = ReadFloats(dir / L"golden.bin", 3);

        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "Experiment08");

        std::cout << "\n--- Optimization OFF ---\n";
        Result off = RunCase(env, model, optimized, GraphOptimizationLevel::ORT_DISABLE_ALL, false, input);
        float off_err = MaxError(off.output, golden);

        std::cout << "Output:\n";
        for (size_t i=0; i<off.output.size(); ++i)
            std::cout << "class[" << i << "] = " << std::setprecision(9) << off.output[i] << "\n";
        std::cout << "Max error     : " << std::scientific << off_err << "\n";
        std::cout << "Verification  : " << (off_err < 1e-5f ? "PASS" : "FAIL") << "\n";
        std::cout << "Median        : " << std::fixed << std::setprecision(3) << off.median_us << " us\n";

        std::cout << "\n--- Optimization ON ---\n";
        Result on = RunCase(env, model, optimized, GraphOptimizationLevel::ORT_ENABLE_ALL, true, input);
        float on_err = MaxError(on.output, golden);

        std::cout << "Output:\n";
        for (size_t i=0; i<on.output.size(); ++i)
            std::cout << "class[" << i << "] = " << std::setprecision(9) << on.output[i] << "\n";
        std::cout << "Max error     : " << std::scientific << on_err << "\n";
        std::cout << "Verification  : " << (on_err < 1e-5f ? "PASS" : "FAIL") << "\n";
        std::cout << "Median        : " << std::fixed << std::setprecision(3) << on.median_us << " us\n";

        std::cout << "\n================ SUMMARY ================\n";
        std::cout << "Optimization OFF : " << off.median_us << " us\n";
        std::cout << "Optimization ON  : " << on.median_us << " us\n";
        std::cout << "Speedup          : " << (off.median_us / on.median_us) << "x\n";
        std::cout << "Optimized graph  : " << optimized.string() << "\n";

        return (off_err < 1e-5f && on_err < 1e-5f) ? 0 : 1;
    }
    catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << "\n";
        return 1;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
