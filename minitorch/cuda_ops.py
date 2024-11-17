# type: ignore
# Currently pyright doesn't support numba.cuda

from typing import Callable, Optional, TypeVar, Any

import numba
from numba import cuda
from numba.cuda import jit as _jit
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

FakeCUDAKernel = Any

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

Fn = TypeVar("Fn")


def device_jit(fn: Fn, **kwargs: Any) -> Fn:
    """Compile function specifically for GPU on device."""
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn: Fn, **kwargs: Any) -> FakeCUDAKernel:
    """Compile function specifically for GPU."""
    return _jit(**kwargs)(fn)  # type: ignore


to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        cufn: Callable[[float], float] = device_jit(fn)
        f = tensor_map(cufn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            # Instantiate and run the cuda kernel.
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """See `tensor_ops.py`"""
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_zip(cufn)

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """See `tensor_ops.py`"""
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_reduce(cufn)

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            threadsperblock = 1024
            blockspergrid = out_a.size
            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Matrix multiply specifically for CUDA."""
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        # One block per batch, extra rows, extra col
        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implement


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        in_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        # TODO: Implement for Task 3.3.
        # raise NotImplementedError("Need to implement for Task 3.3")

        # Check if the thread is within the bounds or not
        if i < out_size:
            # Convert an `ordinal` to an index in the `shape`
            to_index(i, out_shape, out_index)
            # Broadcast indices from out_shape to in_shape
            broadcast_index(out_index, out_shape, in_shape, in_index)
            # Converts a multidimensional tensor `index` into a single-dimensional
            # position in storage based on out_strides and in_strides
            out_pos = index_to_position(out_index, out_strides)
            in_pos = index_to_position(in_index, in_strides)
            # Apply fn to input value and store result in the output array
            out[out_pos] = fn(in_storage[in_pos])

    return cuda.jit()(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
    ----
        fn: function mappings two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        # TODO: Implement for Task 3.3.
        # raise NotImplementedError("Need to implement for Task 3.3")
        if i < out_size:
            # Convert an `ordinal` to an index in the `shape`
            to_index(i, out_shape, out_index)
            out_pos = index_to_position(out_index, out_strides)
            broadcast_index(out_index, out_shape, a_shape, a_index)
            a_pos = index_to_position(a_index, a_strides)
            broadcast_index(out_index, out_shape, b_shape, b_index)
            b_pos = index_to_position(b_index, b_strides)
            out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])

    return cuda.jit()(_zip)  # type: ignore


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    r"""Apply a practice sum kernel to prepare for reduce.

    Given an array of length $n$ and out of size $n // \text{blockDIM}$
    it should sum up each blockDim values into an out cell.

    $[a_1, a_2, ..., a_{100}]$

    |

    $[a_1 +...+ a_{31}, a_{32} + ... + a_{64}, ... ,]$

    Note: Each block must do the sum using shared memory!

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        size (int):  length of a.

    """
    BLOCK_DIM = 32

    # Shared memory allocation for the current block
    cache = cuda.shared.array(BLOCK_DIM, numba.float64)
    # Calculate the global index of the current thread 
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    # Get the thread's local position within the block
    pos = cuda.threadIdx.x

    # TODO: Implement for Task 3.3.
    # raise NotImplementedError("Need to implement for Task 3.3")

    # Load data into shared memory from storage `a`` if within the bounds, 
    # or paddling with the value of 0.0 to the reduction if out of the bounds
    if i < size:
        cache[pos] = a[i] 
        # Synchronize the threads to make sure that everything is loaded into cache
        cuda.syncthreads()
        # Perform the reduction within each block using a 
        # doubling stride pattern with [1, 2, 4, 8, 16]
        idx = 0
        stride = 2 ** idx
        # Each iteration halves the number of active threads
        while stride < BLOCK_DIM:
            modulus_double_stride_pos = pos % (2 * stride)
            if modulus_double_stride_pos == 0:
                cache[pos] += cache[pos + stride]
                # Synchronizes all threads within the block, ensuring that each 
                # step of the reduction completes before moving to the next.
                cuda.syncthreads()
            # Stride doubles after each iteration for [2^0=1, 2^1=2, 2^2=4, 2^3=8, 2^4=16]
            idx += 1
            stride = 2 ** idx
        # Store the block result from the very first thread within each block after reduction completes
        if pos == 0:    # the first thread in each block
            # Each block writes a single result to out, where out contains the
            # partial sums from each block after kernel execution
            out[cuda.blockIdx.x] = cache[0]
    else:
        cache[pos] = 0.0

jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    """Apply a practice sum function to prepare for reduce."""
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """CUDA higher-order tensor reduce function.

    Args:
    ----
        fn: reduction function maps two floats to float.

    Returns:
    -------
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
    ) -> None:
        BLOCK_DIM = 1024
        cache = cuda.shared.array(BLOCK_DIM, numba.float64)
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        out_pos = cuda.blockIdx.x
        pos = cuda.threadIdx.x

        # TODO: Implement for Task 3.3.
        # raise NotImplementedError("Need to implement for Task 3.3")

        # Initialize shared memory with the reduction initial value
        cache[pos] = reduce_value
        if out_pos < out_size:
            # Map out_pos to the appropriate multi-dimensional index
            to_index(out_pos, out_shape, out_index)
            o = index_to_position(out_index, out_strides)

            # Adjust out_index for the reduction dimension
            out_index[reduce_dim] = pos + out_index[reduce_dim] * BLOCK_DIM

            # Check whether within bounds for the reduction dimension
            if out_index[reduce_dim] < a_shape[reduce_dim]:
                # Calculate the input position index
                a_pos = index_to_position(out_index, a_strides)
                # Load the relevant value from a_storage into shared memory
                cache[pos] = a_storage[a_pos]
            
                # Synchronize threads to ensure all threads have loaded their values into cache
                cuda.syncthreads()
            
                # Perform parallel reduction in shared memory using a binary reduction pattern [2^0=1, 2^1=2, 2^2=4, 2^3=8, ]
                idx = 0 
                stride = 2 ** idx
                while stride < BLOCK_DIM:
                    if pos % (stride * 2) == 0:
                        cache[pos] = fn(cache[pos], cache[pos + stride])
                        # Synchronize threads to ensure each reduction step is complete
                        cuda.syncthreads()
                    idx += 1
                    stride = 2 ** idx
                # Only the first thread writes the reduced result to the output array
                if pos == 0:
                    out[o] = cache[0]

    return jit(_reduce)  # type: ignore


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    r"""Apply a practice square MM kernel to prepare for matmul.

    Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].

    Size is always < 32.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Compute

    ```
     for i:
         for j:
              for k:
                  out[i, j] += a[i, k] * b[k, j]
    ```

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        b (Storage): storage for `b` tensor.
        size (int): size of the square

    """
    BLOCK_DIM = 32
    # TODO: Implement for Task 3.3.
    # raise NotImplementedError("Need to implement for Task 3.3")

    # Get the indices of the threads
    x_pos = cuda.threadIdx.x
    y_pos = cuda.threadIdx.y
    # Define shared memory for tiles of `a` and `b`
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    # Load data into shared memory if within bounds
    if x_pos < size and y_pos < size:
        a_shared[x_pos, y_pos] = a[size * x_pos + y_pos]
        b_shared[x_pos, y_pos] = b[size * x_pos + y_pos]
    # Explicitly added cuda.syncthreads() after loading shared memory 
    # and at the end of the reduction step to avoid race conditions
    cuda.syncthreads()  # Synchronize to ensure all threads finish loading
        
    # Perform the matrix multiplication
    if x_pos < size and y_pos < size:
        accumulator = 0.0
        # Iterate over shared memory to compute
        for i in range(size):
            accumulator += a_shared[x_pos, i] * b_shared[i, y_pos]
        cuda.syncthreads()  # Synchronize between reduction steps
        
        # Write the final result to global memory
        out[size * x_pos + y_pos] = accumulator

jit_mm_practice = jit(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    """Performs matrix multiplication of two tensors using CUDA."""
    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """CUDA tensor matrix multiply function.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as ::

    ```python
    assert a_shape[-1] == b_shape[-2]
    ```
    Returns:
        None : Fills in `out`
    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    # Batch dimension - fixed
    batch = cuda.blockIdx.z

    BLOCK_DIM = 32
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # The final position c[i, j]
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # The local position in the block.
    pi = cuda.threadIdx.x
    pj = cuda.threadIdx.y

    # Code Plan:
    # 1) Move across shared dimension by block dim.
    #    a) Copy into shared memory for a matrix.
    #    b) Copy into shared memory for b matrix
    #    c) Compute the dot produce for position c[i, j]
    # TODO: Implement for Task 3.4.
    # raise NotImplementedError("Need to implement for Task 3.4")

    # Initialize accumulator
    accumulator = 0.0

    # Number of chunks to cover the K dimension; for example if a_shape[2] = 128,
    # it will only have 4 chunks with [0, 32, 64, 96] without the boundary of 128 (ceiling function)
    chunks = (BLOCK_DIM + a_shape[2] - 1) // BLOCK_DIM

    # Loop over all shared memory blocks along the shared dimension; shared_dim_start
    # is the starting point for the current block in the shared dimension;
    # It iterates from 0 to a_shape[2] (the size of the shared dimension) in steps of BLOCK_DIM
    for idx in range(chunks):
        shared_dim = idx * BLOCK_DIM
        # Load elements from `a` into shared memory
        shared_dim_index_a = shared_dim + pj
        if i < a_shape[1] and shared_dim_index_a < a_shape[2]: 
            a_index = a_batch_stride * batch + a_strides[1] * i + a_strides[2] * shared_dim_index_a
            a_shared[pi, pj] = a_storage[a_index]
        else:
            a_shared[pi, pj] = 0.0  # Pad with 0 if out of bounds
        
        # Load elements from `b` into shared memory
        shared_dim_index_b = shared_dim + pi
        if shared_dim_index_b < b_shape[1] and j < b_shape[2]: 
            b_index = b_batch_stride * batch + b_strides[1] * shared_dim_index_b + b_strides[2] * j
            b_shared[pi, pj] = b_storage[b_index]
        else:
            a_shared[pi, pj] = 0.0  # Pad with 0 if out of bounds

        # Synchronize threads within the block
        cuda.syncthreads()

        # Perform matrix multiplication for the current block within the shared dimensions
        for shared_dim_index_local in range(BLOCK_DIM):
            # Calculate the global index for the current block in the shared dimension by adding
            # the starting point and shared_dim_index_local, which is the local offset within the block
            shared_dim_index_global = shared_dim + shared_dim_index_local
            # Check if the global index is within bounds of the shared dimension
            if shared_dim_index_global < a_shape[2]:
                # Accumulate the product of corresponding elements from shared memory
                accumulator += a_shared[pi, shared_dim_index_local] * b_shared[shared_dim_index_local, pj]
        
        # Synchronize threads before the next iteration
        cuda.syncthreads()
                
    # Write the result to global memory
    if i < out_shape[1] and j < out_shape[2]:
        ordinal_pos = out_strides[0] * batch + out_strides[1] * i + out_strides[2] * j
        out[ordinal_pos] = accumulator

tensor_matrix_multiply = jit(_tensor_matrix_multiply)
