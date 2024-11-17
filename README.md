# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py




# Diagnostics Output (Parallel Analytics Script) from Task 3_1 (tensor_map, tensor_zip, and tensor_reduce):
```
(.venv) (.venv) (base) zhufangyu@dhcp-vl2051-1069 mod3-fangyuzhu1101 % python project/parallel_check.py
MAP
OMP: Info #276: omp_set_nested routine deprecated, please use omp_set_max_active_levels instead.
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map, 
/Users/zhufangyu/workspace/mod3-fangyuzhu1101/minitorch/fast_ops.py (163)  
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /Users/zhufangyu/workspace/mod3-fangyuzhu1101/minitorch/fast_ops.py (163) 
-----------------------------------------------------------------------------------------------|loop #ID
    def _map(                                                                                  | 
        out: Storage,                                                                          | 
        out_shape: Shape,                                                                      | 
        out_strides: Strides,                                                                  | 
        in_storage: Storage,                                                                   | 
        in_shape: Shape,                                                                       | 
        in_strides: Strides,                                                                   | 
    ) -> None:                                                                                 | 
        # TODO: Implement for Task 3.1.                                                        | 
        # raise NotImplementedError("Need to implement for Task 3.1")                          | 
                                                                                               | 
        # Check both stride alignment and equal shapes to avoid explicit indexing              | 
        if (                                                                                   | 
            len(out_strides) == len(in_strides)                                                | 
            and (out_strides == in_strides).all()----------------------------------------------| #0
            and (out_shape == in_shape).all()--------------------------------------------------| #1
        ):                                                                                     | 
            # Using prange for parallel loop; The main loop iterates over the output           | 
            # tensor indices in parallel, utilizing prange to allow for parallel execution     | 
            for i in prange(len(out)):---------------------------------------------------------| #2
                out[i] = fn(in_storage[i])                                                     | 
        # else if not stride-aligned, index explicitly; run loop in parallel                   | 
        else:                                                                                  | 
            # Loop through all elements in the ordinal array (1d)                              | 
            for i in prange(len(out)):---------------------------------------------------------| #3
                # Initialize all indices using numpy buffers                                   | 
                out_index: Index = np.empty(MAX_DIMS, np.int32)                                | 
                in_index: Index = np.empty(MAX_DIMS, np.int32)                                 | 
                # Convert an `ordinal` to an index in the `shape`                              | 
                to_index(i, out_shape, out_index)                                              | 
                # Broadcast indices from out_shape to in_shape                                 | 
                broadcast_index(out_index, out_shape, in_shape, in_index)                      | 
                # Converts a multidimensional tensor `index` into a single-dimensional         | 
                # position in storage based on out_strides and in_strides                      | 
                out_pos = index_to_position(out_index, out_strides)                            | 
                in_pos = index_to_position(in_index, in_strides)                               | 
                # Apply fn to input value and store result in the output array                 | 
                out[out_pos] = fn(in_storage[in_pos])                                          | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 4 parallel for-
loop(s) (originating from loops labelled: #0, #1, #2, #3).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/zhufangyu/workspace/mod3-fangyuzhu1101/minitorch/fast_ops.py (189) is 
hoisted out of the parallel loop labelled #3 (it will be performed before the 
loop is executed and reused inside the loop):
   Allocation:: out_index: Index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/zhufangyu/workspace/mod3-fangyuzhu1101/minitorch/fast_ops.py (190) is 
hoisted out of the parallel loop labelled #3 (it will be performed before the 
loop is executed and reused inside the loop):
   Allocation:: in_index: Index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip, 
/Users/zhufangyu/workspace/mod3-fangyuzhu1101/minitorch/fast_ops.py (228)  
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/zhufangyu/workspace/mod3-fangyuzhu1101/minitorch/fast_ops.py (228) 
-----------------------------------------------------------------------------------------------|loop #ID
    def _zip(                                                                                  | 
        out: Storage,                                                                          | 
        out_shape: Shape,                                                                      | 
        out_strides: Strides,                                                                  | 
        a_storage: Storage,                                                                    | 
        a_shape: Shape,                                                                        | 
        a_strides: Strides,                                                                    | 
        b_storage: Storage,                                                                    | 
        b_shape: Shape,                                                                        | 
        b_strides: Strides,                                                                    | 
    ) -> None:                                                                                 | 
        # TODO: Implement for Task 3.1.                                                        | 
        # raise NotImplementedError("Need to implement for Task 3.1")                          | 
                                                                                               | 
        # Check both stride alignment and equal shapes to avoid explicit indexing              | 
        if (                                                                                   | 
            len(out_strides) == len(a_strides)                                                 | 
            and len(out_strides) == len(b_strides)                                             | 
            and (out_strides == a_strides).all()-----------------------------------------------| #4
            and (out_strides == b_strides).all()-----------------------------------------------| #5
            and (out_shape == a_shape).all()---------------------------------------------------| #6
            and (out_shape == b_shape).all()---------------------------------------------------| #7
        ):                                                                                     | 
            # Using prange for parallel loop; The main loop iterates over the output           | 
            # tensor indices in parallel, utilizing prange to allow for parallel execution     | 
            for i in prange(len(out)):---------------------------------------------------------| #8
                out[i] = fn(a_storage[i], b_storage[i])                                        | 
        # else if not stride-aligned, index explicitly; run loop in parallel                   | 
        else:                                                                                  | 
            # Loop through all elements in the ordinal array (1d)                              | 
            for i in prange(len(out)):---------------------------------------------------------| #9
                # Initialize all indices using numpy buffers                                   | 
                out_index: Index = np.empty(MAX_DIMS, np.int32)                                | 
                a_index: Index = np.empty(MAX_DIMS, np.int32)                                  | 
                b_index: Index = np.empty(MAX_DIMS, np.int32)                                  | 
                # Convert an `ordinal` to an index in the `shape`                              | 
                to_index(i, out_shape, out_index)                                              | 
                # Converts a multidimensional tensor `index` into a                            | 
                # single-dimensional position in storage based on out_strides                  | 
                out_pos = index_to_position(out_index, out_strides)                            | 
                # Broadcast indices from out_shape to a_shape                                  | 
                broadcast_index(out_index, out_shape, a_shape, a_index)                        | 
                # Converts a multidimensional tensor `index` into a                            | 
                # single-dimensional position in storage based on a_strides                    | 
                a_pos = index_to_position(a_index, a_strides)                                  | 
                # Broadcast indices from out_shape to b_shape                                  | 
                broadcast_index(out_index, out_shape, b_shape, b_index)                        | 
                # Converts a multidimensional tensor `index` into a                            | 
                # single-dimensional position in storage based on b_strides                    | 
                b_pos = index_to_position(b_index, b_strides)                                  | 
                # Apply fn to input value and store result in the output array                 | 
                out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])                          | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 6 parallel for-
loop(s) (originating from loops labelled: #4, #5, #6, #7, #8, #9).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/zhufangyu/workspace/mod3-fangyuzhu1101/minitorch/fast_ops.py (260) is 
hoisted out of the parallel loop labelled #9 (it will be performed before the 
loop is executed and reused inside the loop):
   Allocation:: out_index: Index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/zhufangyu/workspace/mod3-fangyuzhu1101/minitorch/fast_ops.py (261) is 
hoisted out of the parallel loop labelled #9 (it will be performed before the 
loop is executed and reused inside the loop):
   Allocation:: a_index: Index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/zhufangyu/workspace/mod3-fangyuzhu1101/minitorch/fast_ops.py (262) is 
hoisted out of the parallel loop labelled #9 (it will be performed before the 
loop is executed and reused inside the loop):
   Allocation:: b_index: Index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce, 
/Users/zhufangyu/workspace/mod3-fangyuzhu1101/minitorch/fast_ops.py (305)  
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/zhufangyu/workspace/mod3-fangyuzhu1101/minitorch/fast_ops.py (305) 
------------------------------------------------------------------------------------------------|loop #ID
    def _reduce(                                                                                | 
        out: Storage,                                                                           | 
        out_shape: Shape,                                                                       | 
        out_strides: Strides,                                                                   | 
        a_storage: Storage,                                                                     | 
        a_shape: Shape,                                                                         | 
        a_strides: Strides,                                                                     | 
        reduce_dim: int,                                                                        | 
    ) -> None:                                                                                  | 
        # TODO: Implement for Task 3.1.                                                         | 
        # raise NotImplementedError("Need to implement for Task 3.1")                           | 
                                                                                                | 
        # Initialize the index for the output tensor                                            | 
        # Loop through all elements in the ordinal array (1d)                                   | 
        for i in prange(len(out)):--------------------------------------------------------------| #10
            # Initialize all indices using numpy buffers                                        | 
            out_index: Index = np.empty(MAX_DIMS, dtype=np.int32)                               | 
            reduce_size = a_shape[reduce_dim]                                                   | 
            # Convert an `ordinal` to an index in the `shape`                                   | 
            to_index(i, out_shape, out_index)                                                   | 
            # Converts a multidimensional tensor `index` into a single-dimensional              | 
            # position in storage based on out_strides and a_strides                            | 
            out_pos = index_to_position(out_index, out_strides)                                 | 
            a_pos = index_to_position(out_index, a_strides)                                     | 
            accumulator = out[out_pos]                                                          | 
            a_step = a_strides[reduce_dim]                                                      | 
            # Inner-loop should not call any external functions or write non-local variables    | 
            for _ in range(reduce_size):                                                        | 
                # call fn normally inside the inner loop                                        | 
                accumulator = fn(accumulator, a_storage[a_pos])                                 | 
                a_pos += a_step                                                                 | 
            out[out_pos] = accumulator                                                          | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #10).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/zhufangyu/workspace/mod3-fangyuzhu1101/minitorch/fast_ops.py (321) is 
hoisted out of the parallel loop labelled #10 (it will be performed before the 
loop is executed and reused inside the loop):
   Allocation:: out_index: Index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
```


# Diagnostics Output (Parallel Analytics Script) from Task 3_2 (_tensor_matrix_multiply):
```
MATRIX MULTIPLY
 
================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply, 
/Users/zhufangyu/workspace/mod3-fangyuzhu1101/minitorch/fast_ops.py (341)  
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /Users/zhufangyu/workspace/mod3-fangyuzhu1101/minitorch/fast_ops.py (341) 
--------------------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                                      | 
    out: Storage,                                                                                 | 
    out_shape: Shape,                                                                             | 
    out_strides: Strides,                                                                         | 
    a_storage: Storage,                                                                           | 
    a_shape: Shape,                                                                               | 
    a_strides: Strides,                                                                           | 
    b_storage: Storage,                                                                           | 
    b_shape: Shape,                                                                               | 
    b_strides: Strides,                                                                           | 
) -> None:                                                                                        | 
    """NUMBA tensor matrix multiply function.                                                     | 
                                                                                                  | 
    Should work for any tensor shapes that broadcast as long as                                   | 
                                                                                                  | 
    ```                                                                                           | 
    assert a_shape[-1] == b_shape[-2]                                                             | 
    ```                                                                                           | 
                                                                                                  | 
    Optimizations:                                                                                | 
                                                                                                  | 
    * Outer loop in parallel                                                                      | 
    * No index buffers or function calls                                                          | 
    * Inner loop should have no global writes, 1 multiply.                                        | 
                                                                                                  | 
                                                                                                  | 
    Args:                                                                                         | 
    ----                                                                                          | 
        out (Storage): storage for `out` tensor                                                   | 
        out_shape (Shape): shape for `out` tensor                                                 | 
        out_strides (Strides): strides for `out` tensor                                           | 
        a_storage (Storage): storage for `a` tensor                                               | 
        a_shape (Shape): shape for `a` tensor                                                     | 
        a_strides (Strides): strides for `a` tensor                                               | 
        b_storage (Storage): storage for `b` tensor                                               | 
        b_shape (Shape): shape for `b` tensor                                                     | 
        b_strides (Strides): strides for `b` tensor                                               | 
                                                                                                  | 
    Returns:                                                                                      | 
    -------                                                                                       | 
        None : Fills in `out`                                                                     | 
                                                                                                  | 
    """                                                                                           | 
    assert a_shape[-1] == b_shape[-2]                                                             | 
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                                        | 
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                                        | 
    # TODO: Implement for Task 3.2.                                                               | 
    # raise NotImplementedError("Need to implement for Task 3.2")                                 | 
                                                                                                  | 
    batch_size = out_shape[0]                                                                     | 
    row_size = out_shape[1]                                                                       | 
    col_size = out_shape[2]                                                                       | 
    # row of matrix A and col of matrix B                                                         | 
    inner_dimension_size = a_shape[2]                                                             | 
                                                                                                  | 
    # Outer loop in parallel                                                                      | 
    for batch in prange(batch_size):--------------------------------------------------------------| #11
        a_batch_offset = batch * a_batch_stride                                                   | 
        b_batch_offset = batch * b_batch_stride                                                   | 
        # All inner loops should have no global writes, 1 multiply.                               | 
        for row in range(row_size):                                                               | 
            # Get the positions of the starting elements in the storage arrays                    | 
            row_offset_a = row * a_strides[1] + a_batch_offset                                    | 
            for col in range(col_size):                                                           | 
                # Get the positions of the starting elements in the storage arrays                | 
                col_offset_b = col * b_strides[2] + b_batch_offset                                | 
                # initialize the accumulator of products of elements in                           | 
                # the row of matrix A and the column of matrix B                                  | 
                accumulator = 0.0                                                                 | 
                for i in range(inner_dimension_size):                                             | 
                    a_pos = i * a_strides[2] + row_offset_a                                       | 
                    b_pos = i * b_strides[1] + col_offset_b                                       | 
                    accumulator += a_storage[a_pos] * b_storage[b_pos]                            | 
                                                                                                  | 
                # Calculate output position (i,j,k) of the current element in the output array    | 
                out_pos = batch * out_strides[0] + row * out_strides[1] + col * out_strides[2]    | 
                out[out_pos] = accumulator                                                        | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #11).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
```

# Training Log Results / Scripts for Task 3_5 for SMALL Model:
## Dataset: Simple
### CPU
```bash
!python project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET simple --RATE 0.05
```
Total time:  xxxx Time per epoch:  xxxx
```

```

### GPU
```bash
!python project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET simple --RATE 0.05
```
Total time:  xxxx Time per epoch:  xxxx
```

```



## Dataset: Split
### CPU
```bash
!python project/run_fast_tensor.py --BACKEND cpu --HIDDEN 150 --DATASET split --RATE 0.05
```
Total time:  xxxx Time per epoch:  xxxx
```

```

### GPU
```bash
!python project/run_fast_tensor.py --BACKEND gpu --HIDDEN 150 --DATASET split --RATE 0.05
```
Total time:  xxxx Time per epoch:  xxxx
```

```



## Dataset: XOR
### CPU
```bash
!python project/run_fast_tensor.py --BACKEND cpu --HIDDEN 200 --DATASET xor --RATE 0.05
```
Total time:  xxxx Time per epoch:  xxxx
```

```

### GPU
```bash
!python project/run_fast_tensor.py --BACKEND gpu --HIDDEN 200 --DATASET xor --RATE 0.05
```

Total time:  xxxx Time per epoch:  xxxx
```

```




# Training Log Results / Scripts for Task 3_5 for BIG Model:
## Dataset: XXXX
### CPU
```bash
!python project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET simple --RATE 0.05
```
Total time:  xxxx Time per epoch:  xxxx
```

```

### GPU
```bash
!python project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET simple --RATE 0.05
```
Total time:  xxxx Time per epoch:  xxxx
```

```