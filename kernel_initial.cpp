__kernel void wave_update(__global const float* current, __global const float* previous, 
                          __global float* next, __global const float* elevation,
                          const int WIDTH, const int HEIGHT, const float dt_dx2) {
    // Get the global work ID
    int i = get_global_id(1); // Row index
    int j = get_global_id(0); // Column index

    // Calculate linear index for 2D array
    int idx = i * WIDTH + j;

    // Boundary and edge condition checks
    if (i < 0 || i >= HEIGHT || j < 0 || j >= WIDTH) {
        return; // Out of bounds
    }

    // Reflect wave on land (elevation > 0)
    if (elevation[idx] > 0.0f) {
        next[idx] = current[idx];
    } 
    else if (i == 0 || i == HEIGHT - 1 || j == 0 || j == WIDTH - 1) {
        // Absorb wave on edges
        next[idx] = 0.0f;
    } 
    else {
        // Wave propagation for water (elevation <= 0)
        next[idx] = 2.0f * current[idx] - previous[idx] + 
                    dt_dx2 * (current[idx - WIDTH] + current[idx + WIDTH] + 
                              current[idx - 1] + current[idx + 1] - 4.0f * current[idx]);
    }
}


#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>

// Helper function to read kernel source
std::string readKernelFile(const std::string& fileName) {
    std::ifstream file(fileName);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open kernel file: " + fileName);
    }
    return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

int main() {
    try {
        const int WIDTH = 512;
        const int HEIGHT = 512;
        const int TIMESTEPS = 2500;
        const float WAVE_SPEED = 1.0f;
        const float DT = 0.1f;
        const float DX = 1.0f;
        const float dt_dx2 = (WAVE_SPEED * WAVE_SPEED * DT * DT) / (DX * DX);

        size_t grid_size = WIDTH * HEIGHT;

        // Host data initialization
        std::vector<float> current(grid_size, 0.0f);
        std::vector<float> previous(grid_size, 0.0f);
        std::vector<float> next(grid_size, 0.0f);
        std::vector<float> elevation(grid_size, -100.0f);

        // Initialize grid (elevation, current, previous)
        for (int i = 0; i < HEIGHT; i++) {
            for (int j = 0; j < WIDTH; j++) {
                int idx = i * WIDTH + j;
                if ((i - HEIGHT / 2) * (i - HEIGHT / 2) + (j - WIDTH / 2) * (j - WIDTH / 2) <= 4) {
                    current[idx] = 10.0f;
                    previous[idx] = 10.0f;
                }
                if ((i - 400) * (i - 400) + (j - 400) * (j - 400) <= 50 * 50) {
                    elevation[idx] = 100.0f;
                }
            }
        }

        // Setup OpenCL
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        cl::Platform platform = platforms[0];

        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        cl::Device device = devices[0];

        cl::Context context(device);
        cl::CommandQueue queue(context, device);

        // Load and build kernel
        std::string kernelSource = readKernelFile("kernel.cl");
        cl::Program::Sources sources(1, std::make_pair(kernelSource.c_str(), kernelSource.length()));
        cl::Program program(context, sources);
        program.build("-cl-std=CL1.2");
        cl::Kernel kernel(program, "wave_update");

        // Create buffers
        cl::Buffer bufferCurrent(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * grid_size, current.data());
        cl::Buffer bufferPrevious(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * grid_size, previous.data());
        cl::Buffer bufferNext(context, CL_MEM_READ_WRITE, sizeof(float) * grid_size);
        cl::Buffer bufferElevation(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * grid_size, elevation.data());

        // Set kernel arguments
        kernel.setArg(0, bufferCurrent);
        kernel.setArg(1, bufferPrevious);
        kernel.setArg(2, bufferNext);
        kernel.setArg(3, bufferElevation);
        kernel.setArg(4, WIDTH);
        kernel.setArg(5, HEIGHT);
        kernel.setArg(6, dt_dx2);

        // Time-stepping loop
        auto start = std::chrono::high_resolution_clock::now();

        for (int t = 0; t < TIMESTEPS; t++) {
            queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(WIDTH, HEIGHT), cl::NullRange);
            queue.enqueueCopyBuffer(bufferNext, bufferPrevious, 0, 0, sizeof(float) * grid_size);
            queue.enqueueCopyBuffer(bufferCurrent, bufferNext, 0, 0, sizeof(float) * grid_size);
        }

        queue.finish();
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> elapsed = end - start;
        std::cout << "OpenCL execution time: " << elapsed.count() << " seconds." << std::endl;

        // Read back results (if needed for validation)
        queue.enqueueReadBuffer(bufferCurrent, CL_TRUE, 0, sizeof(float) * grid_size, current.data());



    } catch (const cl::Error& err) {
        std::cerr << "OpenCL error: " << err.what() << "(" << err.err() << ")" << std::endl;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
    }

    return 0;
}

