# Profiling in IPPL {#Profiling}

In certain applications, you might want to use profiling tools for debugging and testing. Since IPPL uses **Kokkos** as a backend, you can leverage Kokkos' built-in profiling tools.

This guide explains how to use Kokkos' profiling tools, using the **MemoryEvents** tool as an example.

### Description of MemoryEvents

MemoryEvents tracks a timeline of allocation and deallocation events in Kokkos Memory Spaces. It records time, pointer, size, memory-space-name, and allocation-name. This is in particular useful for debugging purposes to understand where all the memory is going.

Additionally, the tool provides a timeline of memory usage for each individual Kokkos Memory Space.

The tool is located at: https://github.com/kokkos/kokkos-tools/tree/develop/profiling/memory-events


---

## Steps to Use Kokkos Profiling Tools

### 1. Clone the Kokkos Tools Repository

First, clone the Kokkos tools repository, which contains a variety of profiling tools:
```
git clone https://github.com/kokkos/kokkos-tools
```

### 2. Build and Install the Tools

Navigate into the repository and build the tools using CMake:

```
cd kokkos-tools
cmake ..
make -j
sudo make install
```

### 3. Set Up the Profiling Tool

Before running your application, export the Kokkos Tools environment variable to point to the `kp_memory_events.so` tool:
```
export KOKKOS_TOOLS_LIBS={PATH_TO_TOOL_DIRECTORY}/kp_memory_events.so 
```
Replace `{PATH_TO_TOOL_DIRECTORY}` with the actual path where the tool is located.


### 4. Run your Application

Execute your application normally. The MemoryEvents tool will automatically collect data during execution. For example:

```
./application COMMANDS
```

### 5. Output Files

The MemoryEvents tool will generate the following files:

- `HOSTNAME-PROCESSID.mem_events:` Lists memory events.
- `HOSTNAME-PROCESSID-MEMSPACE.memspace_usage:` Provides a utilization timeline for each active memory space.

### 6. Example on with SLURM

Here’s an example of how to run the profiling with a SLURM system using `sbatch`:
```
sbatch -n 2 --wrap="export KOKKOS_TOOLS_LIBS=$HOME/kokkos-tools/kp_memory_events.so; \
mpirun -n 2 LandauDamping 128 128 128 10000 10 FFT 0.01 LeapFrog --overallocate 2.0 --info 10"
```

In this example:

- `sbatch -n 2` specifies 2 nodes.
- The Kokkos tool is exported and applied to the `LandauDamping` application.

This guide provides the basic steps for integrating Kokkos profiling tools into your IPPL-based projects. You can adjust the commands as needed depending on your specific application and environment.

## Example 

Consider the following code:

```
#include <Kokkos_Core.hpp>

  typedef Kokkos::View<int*,Kokkos::CudaSpace> a_type;
  typedef Kokkos::View<int*,Kokkos::CudaUVMSpace> b_type;
  typedef Kokkos::View<int*,Kokkos::CudaHostPinnedSpace> c_type;

int main() {
  Kokkos::initialize();
  {
    int N = 10000000;
    for(int i =0; i<2; i++) { 
      a_type a("A",N);
      {
        b_type b("B",N);
        c_type c("C",N);
        for(int j =0; j<N; j++) {
          b(j)=2*j;
          c(j)=3*j;
        }
      }
    }
  }
  Kokkos::finalize();

}
```

This will produce the following output:

**HOSTNAME-PROCESSID.mem_events**

```
# Memory Events
# Time     Ptr                  Size        MemSpace      Op         Name
0.311749      0x2048a0080            128   CudaHostPinned Allocate   InternalScratchUnified
0.311913     0x2305ca0080           2048             Cuda Allocate   InternalScratchFlags
0.312108     0x2305da0080          16384             Cuda Allocate   InternalScratchSpace
0.312667     0x23060a0080       40000000             Cuda Allocate   A
0.317260     0x23086e0080       40000000          CudaUVM Allocate   B
0.335289      0x2049a0080       40000000   CudaHostPinned Allocate   C
0.368485      0x2049a0080      -40000000   CudaHostPinned DeAllocate C
0.377285     0x23086e0080      -40000000          CudaUVM DeAllocate B
0.379795     0x23060a0080      -40000000             Cuda DeAllocate A
0.380185     0x23060a0080       40000000             Cuda Allocate   A
0.384785     0x23086e0080       40000000          CudaUVM Allocate   B
0.400073      0x2049a0080       40000000   CudaHostPinned Allocate   C
0.433218      0x2049a0080      -40000000   CudaHostPinned DeAllocate C
0.441988     0x23086e0080      -40000000          CudaUVM DeAllocate B
0.444391     0x23060a0080      -40000000             Cuda DeAllocate A
```
**HOSTNAME-PROCESSID-Cuda.memspace_usage**

```
# Space Cuda
# Time(s)  Size(MB)   HighWater(MB)   HighWater-Process(MB)
0.311913 0.0 0.0 81.8
0.312108 0.0 0.0 81.8
0.312667 38.2 38.2 81.8
0.379795 0.0 38.2 158.1
0.380185 38.2 38.2 158.1
0.444391 0.0 38.2 158.1
``` 
**HOSTNAME-PROCESSID-CudaUVM.memspace_usage**

```
# Space CudaUVM
# Time(s)  Size(MB)   HighWater(MB)   HighWater-Process(MB)
0.317260 38.1 38.1 81.8
0.377285 0.0 38.1 158.1
0.384785 38.1 38.1 158.1
0.441988 0.0 38.1 158.1
```

**HOSTNAME-PROCESSID-CudaHostPinned.memspace_usage**

```
# Space CudaHostPinned
# Time(s)  Size(MB)   HighWater(MB)   HighWater-Process(MB)
0.311749 0.0 0.0 81.8
0.335289 38.1 38.1 120.0
0.368485 0.0 38.1 158.1
0.400073 38.1 38.1 158.1
0.433218 0.0 38.1 158.1
```

## mpiP.py

import os
import sys
import re

def sum_send_bytes_from_file(file_path):
    total_bytes_sent = 0.0
    pattern = re.compile(
        r'^\s*Send\s+\d+\s+(\d+)\s+\d+\s+[\d.eE+-]+\s+[\d.eE+-]+\s+[\d.eE+-]+\s+([\d.eE+-]+)\s*$'
    )
    with open(file_path, 'r') as f:
        for line in f:
            match = pattern.match(line)
            if match:
                rank, sum_bytes = match.groups()
                if rank.isdigit():
                    total_bytes_sent += float(sum_bytes)
    return total_bytes_sent

if __name__ == '__main__':
    folder_path = sys.argv[1]
    if os.path.exists(folder_path):
        # Look for any file in the folder ending with .mpiP
        for filename in os.listdir(folder_path):
            if filename.endswith('.mpiP'):
                file_path = os.path.join(folder_path, filename)
                try:
                    total_bytes = sum_send_bytes_from_file(file_path)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                break  # Only one histogram file per run assumed
        print(f"Total bytes = ", total_bytes)
    else:
        print(f"Path given does not exist!")


