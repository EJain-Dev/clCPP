/**
 * Copyright 2025 Ekansh Jain
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#ifndef CLCPP_CL_HPP_
#define CLCPP_CL_HPP_

#define CL_HPP_TARGET_OPENCL_VERSION 300
#define CL_HPP_ENABLE_EXCEPTIONS

#include <CL/opencl.hpp>
#include <functional>
#include <utility>
#include <vector>

namespace clcpp {
/**
 * @brief Interface for holding and getting some information from OpenCL buffers allocated via
 * Device::malloc
 *
 */
class Memory {
 public:
  ~Memory() { _dev.free(*this); }

  /**
   * @brief Get the OpenCL buffer object associated with this Memory
   *
   * @return cThe internal OpenCL buffer object
   *
   * @warning THIS WILL RETURN AN INVALID cl_mem OBJECT IF THE DEVICE DOES NOT USE SVM
   */
  [[nodiscard]] cl_mem getMemObj() const noexcept { return (cl_mem)_data; }
  [[nodiscard]] size_t getSize() const noexcept { return _size; }

 private:
  friend class Device;

  Memory() = delete;
  Memory(Device &dev, void *data, const size_t &size) : _dev(dev), _data(data), _size(size) {};

  void read(const cl_event &event) {
    clRetainEvent(event);
    _events.push_back(event);
  }

  void write(const cl_event &event) {
    clRetainEvent(event);
    _events.clear();
    _events.push_back(event);
  }

  [[nodiscard]] std::vector<cl_event> get() { return _events; }

  void wait() {
    cl::detail::errHandler(clWaitForEvents(_events.size(), _events.data()), "clWaitForEvents");
    for (auto &event : _events) {
      clReleaseEvent(event);
    }
  }

  Device &_dev;
  void *_cpu;
  void *_data;
  size_t _size;
  std::vector<cl_event> _events;
};

/**
 * @brief Manages OpenCL Device, Context, and Queue whilst using auto dependency management with
 * OutOfOrder Queue
 *
 */
class Device {
 public:
  enum class Type { FLOPS, MEM, BOTH };

  /**
   * @brief Construct a new Device object
   *
   * @tparam type What to base the device selection on
   * @param enable_profiling Whether to enable profiling for the created queue
   * @param points_per_gflop The amount of points per gigaflop on the device. Does nothing if type
   * is not BOTH.
   * @param points_per_gbyte The amount of points per gigabyte on the device. Does nothing if type
   * is not BOTH.
   */
  template <Type type>
  Device(const bool &enable_profiling = false, const int &points_per_gflop = 5,
         const int &points_per_gbyte = 1)
      : _device(getBestDevice<type>(points_per_gflop, points_per_gbyte)), _context(_device) {
    if (enable_profiling) {
      _queue = cl::CommandQueue(_context, _device,
                                cl::QueueProperties::OutOfOrder | cl::QueueProperties::Profiling);
    } else {
      _queue = cl::CommandQueue(_context, _device, cl::QueueProperties::OutOfOrder);
    }

    auto svm = _device.getInfo<CL_DEVICE_SVM_CAPABILITIES>();
    if (svm & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM) {
      // Malloc wrapper
      createMem = [](const size_t &size) { return Memory(*this, malloc(size), size); };

      // Free wrapper
      deleteMem = [](Memory &data) {
        data.wait();
        free(data._data);
      };

      // No sync required (always synced)
      syncMem = [](Memory &data) { data._cpu = data._data; };

      // No desync required (always synced)
      desyncMem = [](Memory &) {};
    } else if (svm & CL_DEVICE_SVM_FINE_GRAIN_BUFFER || svm & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER) {
      // Wrapper for clEnqueueSVMFree (lazy freeing)
      deleteMem = [&_queue](Memory &data) {
        cl_event event;
        auto wait_list = data.get();
        cl::detail::errHandler(clEnqueueSVMFree(_queue(), 1, data._data, nullptr, nullptr,
                                                wait_list.size(), wait_list.data(), &event));
        data.write(event);
        clReleaseEvent(event);
      };

      if (svm & CL_DEVICE_SVM_FINE_GRAIN_BUFFER) {
        // Wrapper for clSVMAlloc
        createMem = [&_context](const size_t &size) {
          return clSVMAlloc(_context(), CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER, size, 1);
        };
        // No sync required (always synced)
        syncMem = [](Memory &data) { data._cpu = data._data; };

        // No desync required (always synced)
        desyncMem = [](Memory &) {};
      } else {
        // Wrapper for clSVMAlloc
        createMem = [&_context](const size_t &size) {
          return clSVMAlloc(_context(), CL_MEM_READ_WRITE, size, 1);
        };

        // clEnqueueSVMMap is used to sync
        syncMem = [&_queue](Memory &data) {
          cl_event event;
          auto wait_list = data.get();
          cl::detail::errHandler(
              clEnqueueSVMMap(_queue(), CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, data._data, data._size,
                              wait_list.size(), wait_list.data(), &event),
              "clEnqueueSVMMap");
          data.write(event);
          clReleaseEvent(event);
          data._cpu = data._data;
        };

        // clEnqueueSVMUnmap is used to desync
        desyncMem = [](Memory &data) {
          cl_event event;
          auto wait_list = data.get();
          cl::detail::errHandler(
              clEnqueueSVMUnmap(_queue(), data._cpu, wait_list.size(), wait_list.data(), &event));
          data.write(event);
          clReleaseEvent(event);
        };
      }
    } else {
      // Wrapper for clCreateBuffer
      createMem = [&_context](const size_t &size) {
        cl_int err;
        void *data = (void *)clCreateBuffer(_context(), CL_MEM_READ_WRITE, size, nullptr, &_err);
        cl::detail::errHandler(err, "clCreateBuffer");
        return data;
      };

      // Wrapper for clReleaseMemObject
      deleteMem = [](Memory &data) {
        data.wait();
        cl::detail::errHandler(clReleaseMemObject(data._data));
      };

      // Wrapper for clEnqueueMapBuffer
      syncMem = [&_queue](Memory &data) {
        cl_event event;
        cl_int err;
        auto wait_list = data.get();
        void *mem =
            clEnqueueMapBuffer(_queue(), data._data, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0,
                               data._size, wait_list.size(), wait_list.data(), &event, &err);
        cl::detail::errHandler(err, "clEnqueueMapBuffer");
        data.write(event);
        clReleaseEvent(event);
        data._cpu = mem;
      };

      // Wrapper for clEnqueueUnmapMemObject
      desyncMem = [&_queue](Memory &data) {
        cl_event event;
        auto wait_list = data.get();
        cl::detail::errHandler(clEnqueueUnmapMemObject(_queue(), data._cpu, mem, wait_list.size(),
                                                       wait_list.data(), &event));
        data.write(event);
        clReleaseEvent(event);
      }
    }

    if (svm & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM || svm & CL_DEVICE_SVM_FINE_GRAIN_BUFFER ||
        svm & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER) {
      _uses_svm = true;
      cpyMem = [&_queue](const Memory &src, Memory &dst) {
        cl_event event;
        auto wait_list = dst.get();
        cl::detail::errHandler(clEnqueueSVMMemcpy(_queue(), CL_FALSE, dst._data, src._data,
                                                  src._size, wait_list.size(), wait_list.data(),
                                                  &event));
        src._data.read(event);
        dst.write(event);
        return cl::Event(event);
      };
      fillMem = [&_queue](Memory &data, const void *pattern, const size_t &pattern_size) {
        cl_event event;
        auto wait_list = data.get();
        cl::detail::errHandler(clEnqueueSVMMemFill(_queue(), data._data, pattern, pattern_size,
                                                   data._size, wait_list.size(), wait_list.data(),
                                                   &event));
        data.write(event);
        return cl::Event(event);
      };
    } else {
      _uses_svm = false;

      cpyMem = [&_queue](const Memory &src, Memory &dst) {
        cl_event event;
        auto wait_list = dst.get();
        cl::detail::errHandler(clEnqueueCopyBuffer(_queue(), src._data, dst._data, 0, 0, src._size,
                                                   wait_list.size(), wait_list.data(), &event));
        src._data.read(event);
        dst.write(event);
        return cl::Event(event);
      };
      fillMem = [&_queue](Memory &data, const void *pattern, const size_t &pattern_size) {
        cl_event event;
        auto wait_list = data.get();
        cl::detail::errHandler(clEnqueueFillBuffer(_queue(), data._data, pattern, pattern_size, 0,
                                                   data._size, wait_list.size(), wait_list.data(),
                                                   &event));
        data.write(event);
        return cl::Event(event);
      };
    }
  }

  /**
   * @brief Allocate a buffer on the device
   *
   * @param size The size of the buffer
   * @return The allocated buffer
   */
  Memory malloc(const size_t &size) { return Memory(*this, createMem(size), size); }
  /**
   * @brief Frees a buffer allocated with malloc from this Device object
   *
   * @param data The allocated buffer to free
   */
  void free(Memory &data) { deleteMem(data); }
  /**
   * @brief Copies one buffer into another
   *
   * @param src The buffer to copy from
   * @param dst The buffer to copy to
   * @return The OpenCL HPP Event Object wrapper for the copy command
   */
  cl::Event copy(const Memory &src, Memory &dst) { return cpyMem(src, dst); }
  /**
   * @brief Gets a read/write accessible CPU pointer to the buffer
   *
   * @param data The buffer to get access to
   * @return The CPU pointer
   */
  void *load(Memory &data) {
    syncMem(data);
    return data._cpu;
  }
  /**
   * @brief Deallocates the CPU pointer which will allow blocked GPU commands to run
   *
   * @param data The buffer to deallocate the CPU pointer from
   */
  void unload(Memory &data) { desyncMem(data); }

  [[nodiscard]] cl::CommandQueue &getQueue() { return _queue; }

  [[nodiscard]] cl::Context &getContext() { return _context; }

  [[nodiscard]] cl::Device &getDevice() { return _device; }

  [[nodiscard]] bool usesSVM() { return _uses_svm; }

 private:
  template <Type type>
  cl::Device getBestDevice(const int &points_per_gflop, const int &points_per_gbyte) {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    std::pair<size_t, cl::Device> best_device{0, {}};

    for (const auto &platform : platforms) {
      constexpr size_t GFLOPS_TO_FLOPS = 1'000'000'000;

      std::vector<cl::Device> devices;
      platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
      for (const auto &device : devices) {
        if constexpr (type == Type::FLOPS) {
          const size_t gflops = (device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() *
                                 device.getInfo<CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT>() *
                                 device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() *
                                 device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>()) /
                                GFLOPS_TO_FLOPS;
          if (gflops > best_device.first) {
            best_device.first = gflops;
            best_device.second = device;
          } else if constexpr (type == Type::MEM) {
            const size_t mem = device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() / GBYTES_TO_BYTES;
            if (mem > best_device.first) {
              best_device.first = mem;
              best_device.second = device;
            }
          } else {
            const size_t score =
                ((device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() *
                  device.getInfo<CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT>() *
                  device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() *
                  device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>()) /
                 GFLOPS_TO_FLOPS) *
                    points_per_gflop +
                (device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() / GBYTES_TO_BYTES) * points_per_gbyte;
            if (score > best_device.first) {
              best_device.first = score;
              best_device.second = device;
            }
          }
        }
      }
    }

    return best_device.second;
  }

  std::function<void *(const size_t &)> createMem;
  std::function<void(Memory &)> deleteMem;
  std::function<cl::Event(const Memory &, Memory &)> cpyMem;
  std::function<cl::Event(Memory &, const void *, const size_t &)> fillMem;
  std::function<void(Memory &)> syncMem;
  std::function<void(Memory &)> desyncMem;
  cl::Device _device;
  cl::CommandQueue _queue;
  cl::Context _context;
  bool _uses_svm;
};

}  // namespace clcpp

#endif