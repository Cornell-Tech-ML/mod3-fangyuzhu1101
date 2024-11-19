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



# Below is my timing summary output to show my GPU run faster than CPU when size is bigger:
```
Running size 64
/usr/local/lib/python3.10/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 8 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
{'fast': np.float64(0.0034505526224772134), 'gpu': np.float64(0.006682793299357097)}
Running size 128
/usr/local/lib/python3.10/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 32 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
{'fast': np.float64(0.015791893005371094), 'gpu': np.float64(0.01562841733296712)}
Running size 256
{'fast': np.float64(0.09758281707763672), 'gpu': np.float64(0.06906938552856445)}
Running size 512
{'fast': np.float64(0.985863447189331), 'gpu': np.float64(0.21808258692423502)}
Running size 1024
{'fast': np.float64(7.924315452575684), 'gpu': np.float64(1.0107994079589844)}

Timing summary
Size: 64
    fast: 0.00345
    gpu: 0.00668
Size: 128
    fast: 0.01579
    gpu: 0.01563
Size: 256
    fast: 0.09758
    gpu: 0.06907
Size: 512
    fast: 0.98586
    gpu: 0.21808
Size: 1024
    fast: 7.92432
    gpu: 1.01080
```



# Training Log Results / Scripts for Task 3_5 for SMALL Model:
## Dataset: Simple
### CPU
```bash
!python project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET simple --RATE 0.05
```
Number of Points: 50 <br>
Size of Hidden Layer: 100 <br>
Number of Epochs: 500 <br>
Learning Rate: 0.05 <br>
```
Epoch  0  loss  4.5355393188731234 correct 46 time per epoch 29.635552406311035
Epoch  10  loss  1.37573350400431 correct 50 time per epoch 0.13977766036987305
Epoch  20  loss  1.0721839287772474 correct 50 time per epoch 0.13875961303710938
Epoch  30  loss  0.796063965459155 correct 50 time per epoch 0.14733052253723145
Epoch  40  loss  0.6151716748445172 correct 50 time per epoch 0.13936305046081543
Epoch  50  loss  0.12823901136866822 correct 50 time per epoch 0.24874401092529297
Epoch  60  loss  0.14369873834093572 correct 50 time per epoch 0.1385972499847412
Epoch  70  loss  0.47682716606924413 correct 50 time per epoch 0.14442205429077148
Epoch  80  loss  0.3720152270930525 correct 50 time per epoch 0.13903594017028809
Epoch  90  loss  0.06065483002576047 correct 50 time per epoch 0.14135122299194336
Epoch  100  loss  0.19369682050599338 correct 50 time per epoch 0.1389462947845459
Epoch  110  loss  0.3664544481824471 correct 50 time per epoch 0.13926410675048828
Epoch  120  loss  0.43378356823316233 correct 50 time per epoch 0.1406412124633789
Epoch  130  loss  0.14745728148969448 correct 50 time per epoch 0.234788179397583
Epoch  140  loss  0.38035067886934215 correct 50 time per epoch 0.14019536972045898
Epoch  150  loss  0.2149744509010663 correct 50 time per epoch 0.13859128952026367
Epoch  160  loss  0.15169367719939025 correct 50 time per epoch 0.14002275466918945
Epoch  170  loss  0.015446123394160469 correct 50 time per epoch 0.15187692642211914
Epoch  180  loss  0.08787871233707062 correct 50 time per epoch 0.13937711715698242
Epoch  190  loss  0.20696625425502474 correct 50 time per epoch 0.14031553268432617
Epoch  200  loss  0.18296700920198017 correct 50 time per epoch 0.13939857482910156
Epoch  210  loss  0.1265707111823651 correct 50 time per epoch 0.13811016082763672
Epoch  220  loss  0.12543877122187305 correct 50 time per epoch 0.3266918659210205
Epoch  230  loss  0.012902889179787654 correct 50 time per epoch 0.15084052085876465
Epoch  240  loss  0.05571982563829585 correct 50 time per epoch 0.13856172561645508
Epoch  250  loss  0.06755697969102782 correct 50 time per epoch 0.13771533966064453
Epoch  260  loss  0.013065037513833684 correct 50 time per epoch 0.15232610702514648
Epoch  270  loss  0.028935609433419475 correct 50 time per epoch 0.13860654830932617
Epoch  280  loss  0.03220772405005083 correct 50 time per epoch 0.13825511932373047
Epoch  290  loss  0.09257471684521282 correct 50 time per epoch 0.1385955810546875
Epoch  300  loss  0.05489546953275265 correct 50 time per epoch 0.19448232650756836
Epoch  310  loss  0.07994557631416221 correct 50 time per epoch 0.13819527626037598
Epoch  320  loss  0.011184502994513023 correct 50 time per epoch 0.14258956909179688
Epoch  330  loss  0.0677175732236138 correct 50 time per epoch 0.13904190063476562
Epoch  340  loss  0.04585402607973744 correct 50 time per epoch 0.13884425163269043
Epoch  350  loss  0.04494181802454293 correct 50 time per epoch 0.14274048805236816
Epoch  360  loss  0.07066666319671548 correct 50 time per epoch 0.14765024185180664
Epoch  370  loss  0.0262447509937281 correct 50 time per epoch 0.14013075828552246
Epoch  380  loss  0.02040134867055935 correct 50 time per epoch 0.2958567142486572
Epoch  390  loss  0.12177938890170943 correct 50 time per epoch 0.14121770858764648
Epoch  400  loss  0.02236470057144798 correct 50 time per epoch 0.14000940322875977
Epoch  410  loss  0.0006425914164911817 correct 50 time per epoch 0.13969135284423828
Epoch  420  loss  0.02152276592400106 correct 50 time per epoch 0.15120577812194824
Epoch  430  loss  0.042775872445652716 correct 50 time per epoch 0.13901376724243164
Epoch  440  loss  0.01031898939124597 correct 50 time per epoch 0.13920950889587402
Epoch  450  loss  0.08157899084598533 correct 50 time per epoch 0.1497194766998291
Epoch  460  loss  0.02154700654950008 correct 50 time per epoch 0.24375152587890625
Epoch  470  loss  0.011259028972474034 correct 50 time per epoch 0.14055299758911133
Epoch  480  loss  0.0016911329999263873 correct 50 time per epoch 0.13914155960083008
Epoch  490  loss  0.00684314756337077 correct 50 time per epoch 0.1378183364868164
```

### GPU
```bash
!python project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET simple --RATE 0.05
```
Number of Points: 50 <br>
Size of Hidden Layer: 100 <br>
Number of Epochs: 500 <br>
Learning Rate: 0.05 <br>
```
Epoch  0  loss  5.751353024733923 correct 37 time per epoch 5.1931328773498535
Epoch  10  loss  1.8468575076392475 correct 50 time per epoch 1.902599573135376
Epoch  20  loss  1.5833079010727147 correct 49 time per epoch 2.2753849029541016
Epoch  30  loss  1.4687144123141915 correct 50 time per epoch 1.9555490016937256
Epoch  40  loss  0.5798972414609586 correct 50 time per epoch 1.881248950958252
Epoch  50  loss  0.4365202744739489 correct 50 time per epoch 2.68739914894104
Epoch  60  loss  0.22216937967121275 correct 49 time per epoch 1.8965890407562256
Epoch  70  loss  0.17895463386531837 correct 50 time per epoch 1.9588263034820557
Epoch  80  loss  0.44473423437946863 correct 50 time per epoch 2.439940929412842
Epoch  90  loss  0.12309057810259737 correct 50 time per epoch 1.8913819789886475
Epoch  100  loss  0.23926389806061565 correct 50 time per epoch 1.8709590435028076
Epoch  110  loss  0.30252617290308165 correct 50 time per epoch 1.8926129341125488
Epoch  120  loss  0.3522470331556375 correct 50 time per epoch 1.95005202293396
Epoch  130  loss  0.1733951502125227 correct 50 time per epoch 2.0649595260620117
Epoch  140  loss  0.19053569958103847 correct 50 time per epoch 1.8947646617889404
Epoch  150  loss  0.07118162554538704 correct 50 time per epoch 1.8737051486968994
Epoch  160  loss  0.14494599420181342 correct 50 time per epoch 2.466050386428833
Epoch  170  loss  0.03538919429721538 correct 50 time per epoch 1.9583404064178467
Epoch  180  loss  0.07731283469118179 correct 50 time per epoch 1.9093005657196045
Epoch  190  loss  0.06105247151785305 correct 50 time per epoch 2.334167003631592
Epoch  200  loss  0.259971354301202 correct 50 time per epoch 1.9051940441131592
Epoch  210  loss  0.10170118533564669 correct 50 time per epoch 1.8879594802856445
Epoch  220  loss  0.35562768197335853 correct 50 time per epoch 1.9462308883666992
Epoch  230  loss  0.02030670836867822 correct 50 time per epoch 1.8994829654693604
Epoch  240  loss  0.02423896323926567 correct 50 time per epoch 2.1499485969543457
Epoch  250  loss  0.005146342334379608 correct 50 time per epoch 1.8913569450378418
Epoch  260  loss  0.17598338343854555 correct 50 time per epoch 1.9442272186279297
Epoch  270  loss  0.07177258727868148 correct 50 time per epoch 2.672820806503296
Epoch  280  loss  0.11025941045773027 correct 50 time per epoch 1.9066097736358643
Epoch  290  loss  0.13773751334656367 correct 50 time per epoch 1.9147062301635742
Epoch  300  loss  0.029632654580307144 correct 50 time per epoch 2.7198824882507324
Epoch  310  loss  0.022885055063148474 correct 50 time per epoch 1.9847280979156494
Epoch  320  loss  0.013065375765486394 correct 50 time per epoch 1.8938345909118652
Epoch  330  loss  0.17511637599877047 correct 50 time per epoch 2.2096478939056396
Epoch  340  loss  0.07863747088998842 correct 50 time per epoch 1.880237102508545
Epoch  350  loss  0.04969631849403272 correct 50 time per epoch 1.9656217098236084
Epoch  360  loss  0.07681210571081282 correct 50 time per epoch 1.9173588752746582
Epoch  370  loss  0.12534686044505594 correct 50 time per epoch 1.8898406028747559
Epoch  380  loss  0.023680284225598203 correct 50 time per epoch 2.665383815765381
Epoch  390  loss  0.08455045203217139 correct 50 time per epoch 1.9632463455200195
Epoch  400  loss  0.012911620181819617 correct 50 time per epoch 1.9579427242279053
Epoch  410  loss  0.0015466782583703274 correct 50 time per epoch 2.1695797443389893
Epoch  420  loss  0.011221679504789831 correct 50 time per epoch 1.883885383605957
Epoch  430  loss  0.1331149270132045 correct 50 time per epoch 1.9118492603302002
Epoch  440  loss  0.002493484887674113 correct 50 time per epoch 1.9556608200073242
Epoch  450  loss  0.059002564803680664 correct 50 time per epoch 1.8949339389801025
Epoch  460  loss  0.06896397908910479 correct 50 time per epoch 2.2837588787078857
Epoch  470  loss  0.06438369446740927 correct 50 time per epoch 1.9015684127807617
Epoch  480  loss  0.0026273529169106914 correct 50 time per epoch 1.947411060333252
Epoch  490  loss  0.0003444003845738285 correct 50 time per epoch 2.6712636947631836
```



## Dataset: Split
### CPU
```bash
!python project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET split --RATE 0.05
```
Number of Points: 50 <br>
Size of Hidden Layer: 100 <br>
Number of Epochs: 500 <br>
Learning Rate: 0.05 <br>
```
Epoch  0  loss  8.841741913270358 correct 37 time per epoch 29.337392330169678
Epoch  10  loss  7.502702108606075 correct 41 time per epoch 0.13925528526306152
Epoch  20  loss  3.3453344642289657 correct 45 time per epoch 0.15218377113342285
Epoch  30  loss  2.3635596812173376 correct 44 time per epoch 0.1389918327331543
Epoch  40  loss  1.6345379498373669 correct 49 time per epoch 0.13911700248718262
Epoch  50  loss  2.887987600443232 correct 48 time per epoch 0.301480770111084
Epoch  60  loss  1.6614733587761708 correct 49 time per epoch 0.13973760604858398
Epoch  70  loss  0.8247376946332452 correct 49 time per epoch 0.13986420631408691
Epoch  80  loss  0.5950006942939867 correct 50 time per epoch 0.14726495742797852
Epoch  90  loss  0.9547228783561761 correct 50 time per epoch 0.13901829719543457
Epoch  100  loss  1.8542390444325643 correct 50 time per epoch 0.1405472755432129
Epoch  110  loss  0.5705704941615127 correct 50 time per epoch 0.13936996459960938
Epoch  120  loss  0.9906074340548539 correct 50 time per epoch 0.14113521575927734
Epoch  130  loss  1.1260300352100259 correct 50 time per epoch 0.2556438446044922
Epoch  140  loss  0.2727463934285847 correct 50 time per epoch 0.140214204788208
Epoch  150  loss  0.5678460165543068 correct 50 time per epoch 0.14081692695617676
Epoch  160  loss  0.1702332641293024 correct 50 time per epoch 0.13713812828063965
Epoch  170  loss  0.40313644837881646 correct 50 time per epoch 0.14160466194152832
Epoch  180  loss  0.6481180706993881 correct 50 time per epoch 0.13934636116027832
Epoch  190  loss  0.1872243069774321 correct 50 time per epoch 0.14200162887573242
Epoch  200  loss  0.8109188554138113 correct 50 time per epoch 0.14224982261657715
Epoch  210  loss  0.1832627022186339 correct 50 time per epoch 0.27408909797668457
Epoch  220  loss  0.46626813927360417 correct 50 time per epoch 0.13966870307922363
Epoch  230  loss  0.10453416944172672 correct 50 time per epoch 0.1419050693511963
Epoch  240  loss  0.10211631589289283 correct 50 time per epoch 0.1562027931213379
Epoch  250  loss  0.5033589638537991 correct 50 time per epoch 0.13994646072387695
Epoch  260  loss  0.3721580534850496 correct 50 time per epoch 0.13902044296264648
Epoch  270  loss  0.5389937700052322 correct 50 time per epoch 0.16551470756530762
Epoch  280  loss  0.12727237277263956 correct 50 time per epoch 0.14142775535583496
Epoch  290  loss  0.4857241512309145 correct 50 time per epoch 0.25151872634887695
Epoch  300  loss  0.27164769221073654 correct 50 time per epoch 0.13886427879333496
Epoch  310  loss  0.4716041661381346 correct 50 time per epoch 0.13906002044677734
Epoch  320  loss  0.027026634768552472 correct 50 time per epoch 0.13998866081237793
Epoch  330  loss  0.08443151280524344 correct 50 time per epoch 0.1376032829284668
Epoch  340  loss  0.05756851854450872 correct 50 time per epoch 0.14018988609313965
Epoch  350  loss  0.3671504556625222 correct 50 time per epoch 0.13950657844543457
Epoch  360  loss  0.12559649222826147 correct 50 time per epoch 0.1418755054473877
Epoch  370  loss  0.27027181358316515 correct 50 time per epoch 0.15355753898620605
Epoch  380  loss  0.2650149675025698 correct 50 time per epoch 0.2549617290496826
Epoch  390  loss  0.444043199981697 correct 50 time per epoch 0.13901782035827637
Epoch  400  loss  0.04176097958647931 correct 50 time per epoch 0.14959502220153809
Epoch  410  loss  0.08729451609997299 correct 50 time per epoch 0.1399223804473877
Epoch  420  loss  0.07374869693242646 correct 50 time per epoch 0.14131975173950195
Epoch  430  loss  0.0897583348797974 correct 50 time per epoch 0.1444246768951416
Epoch  440  loss  0.08686612675173626 correct 50 time per epoch 0.1400601863861084
Epoch  450  loss  0.04462539002167261 correct 50 time per epoch 0.1392064094543457
Epoch  460  loss  0.04115077016142798 correct 50 time per epoch 0.26902151107788086
Epoch  470  loss  0.012329430054743278 correct 50 time per epoch 0.13974452018737793
Epoch  480  loss  0.05014619080998955 correct 50 time per epoch 0.13775372505187988
Epoch  490  loss  0.02586378834353629 correct 50 time per epoch 0.13785338401794434
```

### GPU
```bash
!python project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET split --RATE 0.05
```
Number of Points: 50 <br>
Size of Hidden Layer: 100 <br>
Number of Epochs: 500 <br>
Learning Rate: 0.05 <br>
```
Epoch  0  loss  3.7924611930465986 correct 35 time per epoch 5.3063554763793945
Epoch  10  loss  9.220994788690673 correct 36 time per epoch 1.9216480255126953
Epoch  20  loss  3.5653545407514176 correct 41 time per epoch 1.9046099185943604
Epoch  30  loss  4.217630839990408 correct 41 time per epoch 2.4145727157592773
Epoch  40  loss  3.3542521291463734 correct 43 time per epoch 1.8928143978118896
Epoch  50  loss  4.816191196962796 correct 45 time per epoch 1.9004030227661133
Epoch  60  loss  2.937406514660018 correct 41 time per epoch 2.038313388824463
Epoch  70  loss  1.322277494733767 correct 50 time per epoch 1.9662418365478516
Epoch  80  loss  1.957907605687796 correct 49 time per epoch 1.973874807357788
Epoch  90  loss  1.2119476471484532 correct 50 time per epoch 1.9382495880126953
Epoch  100  loss  2.2757904773935236 correct 48 time per epoch 1.9163131713867188
Epoch  110  loss  0.6009710671169647 correct 47 time per epoch 2.125844717025757
Epoch  120  loss  3.225518901615747 correct 45 time per epoch 2.0072073936462402
Epoch  130  loss  2.014256885260985 correct 49 time per epoch 1.9017164707183838
Epoch  140  loss  1.1161495281969067 correct 46 time per epoch 2.47948956489563
Epoch  150  loss  0.2545288288306168 correct 50 time per epoch 1.8966522216796875
Epoch  160  loss  0.5949964161492048 correct 44 time per epoch 1.9078280925750732
Epoch  170  loss  0.31889342407544025 correct 48 time per epoch 2.715015411376953
Epoch  180  loss  0.4506359259320774 correct 50 time per epoch 1.9238338470458984
Epoch  190  loss  0.301999814461715 correct 48 time per epoch 1.9052133560180664
Epoch  200  loss  1.5802327375102865 correct 47 time per epoch 2.2435901165008545
Epoch  210  loss  0.11443374433257862 correct 46 time per epoch 1.910311222076416
Epoch  220  loss  0.6412919554794994 correct 48 time per epoch 1.989691972732544
Epoch  230  loss  0.21238122229419706 correct 50 time per epoch 1.8909530639648438
Epoch  240  loss  0.35226572355879904 correct 50 time per epoch 1.9121601581573486
Epoch  250  loss  0.1276128686001799 correct 49 time per epoch 2.0223276615142822
Epoch  260  loss  0.2612455759807306 correct 50 time per epoch 1.9722354412078857
Epoch  270  loss  2.9169655452360987 correct 49 time per epoch 1.8969709873199463
Epoch  280  loss  1.6075955569529385 correct 50 time per epoch 2.459754228591919
Epoch  290  loss  1.7129492618890283 correct 48 time per epoch 1.898406982421875
Epoch  300  loss  0.5772260633714869 correct 47 time per epoch 1.918031930923462
Epoch  310  loss  0.9124369715980599 correct 48 time per epoch 2.7306933403015137
Epoch  320  loss  0.527275110536915 correct 49 time per epoch 1.885777473449707
Epoch  330  loss  1.0859361182459666 correct 43 time per epoch 1.9090287685394287
Epoch  340  loss  1.1299709848880777 correct 50 time per epoch 2.2965002059936523
Epoch  350  loss  2.3485161949843705 correct 49 time per epoch 1.969367504119873
Epoch  360  loss  0.05915019524081376 correct 50 time per epoch 1.9070909023284912
Epoch  370  loss  0.41291404109174135 correct 50 time per epoch 1.9445109367370605
Epoch  380  loss  1.6160489062735797 correct 49 time per epoch 1.9227724075317383
Epoch  390  loss  1.4487116544845753 correct 50 time per epoch 1.9575855731964111
Epoch  400  loss  0.1401619577965308 correct 50 time per epoch 1.9709241390228271
Epoch  410  loss  0.4024506438932068 correct 48 time per epoch 1.9078123569488525
Epoch  420  loss  0.12066135141842055 correct 50 time per epoch 2.208515167236328
Epoch  430  loss  0.13966296952250548 correct 47 time per epoch 1.9411704540252686
Epoch  440  loss  0.32521988530877055 correct 50 time per epoch 1.9881787300109863
Epoch  450  loss  0.7004677955225781 correct 49 time per epoch 2.4656291007995605
Epoch  460  loss  0.1578022920864075 correct 50 time per epoch 1.9330980777740479
Epoch  470  loss  0.073365138586502 correct 50 time per epoch 1.930927038192749
Epoch  480  loss  0.3661102292793289 correct 50 time per epoch 2.7838618755340576
Epoch  490  loss  0.025197850294483785 correct 50 time per epoch 1.8903377056121826
```



## Dataset: XOR
### CPU
```bash
!python project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET xor --RATE 0.05
```
Number of Points: 50 <br>
Size of Hidden Layer: 100 <br>
Number of Epochs: 500 <br>
Learning Rate: 0.05 <br>
```
Epoch  0  loss  6.620686684961389 correct 33 time per epoch 29.01722478866577
Epoch  10  loss  3.9364534998191996 correct 37 time per epoch 0.13757729530334473
Epoch  20  loss  4.038920123421797 correct 43 time per epoch 0.1392359733581543
Epoch  30  loss  5.885718774624159 correct 38 time per epoch 0.14774632453918457
Epoch  40  loss  3.6230396466268577 correct 45 time per epoch 0.2467949390411377
Epoch  50  loss  2.9569803106526535 correct 46 time per epoch 0.1391277313232422
Epoch  60  loss  1.6407887198512547 correct 47 time per epoch 0.1404416561126709
Epoch  70  loss  2.118878697179449 correct 47 time per epoch 0.13968253135681152
Epoch  80  loss  3.183421633410651 correct 47 time per epoch 0.14117884635925293
Epoch  90  loss  2.820171734058356 correct 41 time per epoch 0.14048171043395996
Epoch  100  loss  3.189231360190968 correct 47 time per epoch 0.14095425605773926
Epoch  110  loss  2.7413856612317127 correct 47 time per epoch 0.13928723335266113
Epoch  120  loss  0.734569141790183 correct 41 time per epoch 0.2242887020111084
Epoch  130  loss  3.1623968982055715 correct 48 time per epoch 0.13999199867248535
Epoch  140  loss  1.5812152668313109 correct 48 time per epoch 0.14118742942810059
Epoch  150  loss  3.766794751167706 correct 49 time per epoch 0.1421492099761963
Epoch  160  loss  1.7590977451700187 correct 45 time per epoch 0.14403104782104492
Epoch  170  loss  0.9101806066565754 correct 48 time per epoch 0.14072179794311523
Epoch  180  loss  1.06921970265429 correct 48 time per epoch 0.13862323760986328
Epoch  190  loss  1.6396782620176367 correct 48 time per epoch 0.14663362503051758
Epoch  200  loss  1.9822195007563874 correct 48 time per epoch 0.15130066871643066
Epoch  210  loss  1.0244228130964774 correct 48 time per epoch 0.23320221900939941
Epoch  220  loss  2.0616119541318363 correct 50 time per epoch 0.13843917846679688
Epoch  230  loss  0.4547762600483041 correct 50 time per epoch 0.1432511806488037
Epoch  240  loss  0.37448128242022577 correct 48 time per epoch 0.1413881778717041
Epoch  250  loss  1.1724157882337052 correct 48 time per epoch 0.15286874771118164
Epoch  260  loss  1.7331242785068701 correct 48 time per epoch 0.14248180389404297
Epoch  270  loss  0.3477988307942892 correct 48 time per epoch 0.14307785034179688
Epoch  280  loss  0.6787849662217013 correct 48 time per epoch 0.142838716506958
Epoch  290  loss  1.064051503727734 correct 50 time per epoch 0.27017712593078613
Epoch  300  loss  0.4100101657887074 correct 50 time per epoch 0.1405010223388672
Epoch  310  loss  1.2029839551607215 correct 50 time per epoch 0.14108490943908691
Epoch  320  loss  1.9414509397586341 correct 48 time per epoch 0.14056921005249023
Epoch  330  loss  1.4229980983499768 correct 50 time per epoch 0.1407608985900879
Epoch  340  loss  0.25583243715760934 correct 48 time per epoch 0.1479177474975586
Epoch  350  loss  0.8660965115634096 correct 48 time per epoch 0.1378002166748047
Epoch  360  loss  0.30018563728602476 correct 48 time per epoch 0.13745713233947754
Epoch  370  loss  0.2466953443257228 correct 49 time per epoch 0.31356143951416016
Epoch  380  loss  1.625601865540228 correct 48 time per epoch 0.14574050903320312
Epoch  390  loss  1.5539094893289407 correct 48 time per epoch 0.13998150825500488
Epoch  400  loss  0.5677830864833617 correct 49 time per epoch 0.1395714282989502
Epoch  410  loss  0.9684906976633822 correct 50 time per epoch 0.13889217376708984
Epoch  420  loss  0.3266228700026391 correct 50 time per epoch 0.13998913764953613
Epoch  430  loss  1.8039882965588205 correct 50 time per epoch 0.13851094245910645
Epoch  440  loss  1.999053003413062 correct 50 time per epoch 0.1507585048675537
Epoch  450  loss  1.2290424849255195 correct 50 time per epoch 0.2706782817840576
Epoch  460  loss  0.7994658791827036 correct 48 time per epoch 0.13910603523254395
Epoch  470  loss  2.113551813817592 correct 48 time per epoch 0.14992642402648926
Epoch  480  loss  0.9439852089756632 correct 50 time per epoch 0.13941287994384766
Epoch  490  loss  0.49334782028886476 correct 49 time per epoch 0.13959717750549316
```

### GPU
```bash
!python project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET xor --RATE 0.05
```
Number of Points: 50 <br>
Size of Hidden Layer: 100 <br>
Number of Epochs: 500 <br>
Learning Rate: 0.05 <br>
```
Epoch  0  loss  7.397245949654798 correct 27 time per epoch 4.361684799194336
Epoch  10  loss  5.399030254391471 correct 43 time per epoch 1.8997726440429688
Epoch  20  loss  5.052002785300129 correct 41 time per epoch 2.7014832496643066
Epoch  30  loss  4.773276620544582 correct 41 time per epoch 1.9812016487121582
Epoch  40  loss  4.339882209801492 correct 43 time per epoch 1.9507811069488525
Epoch  50  loss  3.5207814901996795 correct 44 time per epoch 2.717831611633301
Epoch  60  loss  3.3673392586740665 correct 45 time per epoch 1.9236981868743896
Epoch  70  loss  1.8287475576641383 correct 44 time per epoch 1.9255526065826416
Epoch  80  loss  4.985761688637895 correct 45 time per epoch 2.558478832244873
Epoch  90  loss  2.6670883987707352 correct 44 time per epoch 1.9026556015014648
Epoch  100  loss  4.774097446150259 correct 45 time per epoch 1.919525146484375
Epoch  110  loss  3.2696719778920382 correct 46 time per epoch 2.184699535369873
Epoch  120  loss  2.6293452239030852 correct 49 time per epoch 2.000990629196167
Epoch  130  loss  2.02906902483804 correct 48 time per epoch 1.9197354316711426
Epoch  140  loss  1.7948838202557749 correct 48 time per epoch 2.0055243968963623
Epoch  150  loss  1.7666166206015652 correct 48 time per epoch 1.920973539352417
Epoch  160  loss  1.3748357970363483 correct 48 time per epoch 1.927476167678833
Epoch  170  loss  2.012167718927966 correct 46 time per epoch 2.0233232975006104
Epoch  180  loss  1.8999268693769067 correct 48 time per epoch 1.911919355392456
Epoch  190  loss  1.9062614340925892 correct 46 time per epoch 2.26053786277771
Epoch  200  loss  1.4940702711154634 correct 48 time per epoch 1.9545440673828125
Epoch  210  loss  1.2325148566179907 correct 49 time per epoch 1.9341838359832764
Epoch  220  loss  0.42357086476415073 correct 50 time per epoch 2.607538938522339
Epoch  230  loss  2.070504557100569 correct 48 time per epoch 1.8924343585968018
Epoch  240  loss  0.3430802910245228 correct 49 time per epoch 1.9106497764587402
Epoch  250  loss  0.465569159656921 correct 49 time per epoch 2.714266538619995
Epoch  260  loss  1.7325724847671233 correct 50 time per epoch 1.9903700351715088
Epoch  270  loss  1.1437074871145128 correct 50 time per epoch 1.9230842590332031
Epoch  280  loss  0.741418780533251 correct 50 time per epoch 2.5917179584503174
Epoch  290  loss  0.804823786040134 correct 49 time per epoch 1.9345104694366455
Epoch  300  loss  0.4920413244251899 correct 48 time per epoch 1.945329189300537
Epoch  310  loss  0.8060433032773556 correct 48 time per epoch 2.507370710372925
Epoch  320  loss  0.9028790662731624 correct 48 time per epoch 1.9116671085357666
Epoch  330  loss  0.6644034696431432 correct 50 time per epoch 1.9195184707641602
Epoch  340  loss  2.088273521020078 correct 50 time per epoch 2.1980645656585693
Epoch  350  loss  0.4957430581759439 correct 49 time per epoch 1.9848802089691162
Epoch  360  loss  0.08996831057553002 correct 49 time per epoch 1.9033317565917969
Epoch  370  loss  1.0202566497485357 correct 49 time per epoch 2.012047052383423
Epoch  380  loss  0.9181452374861928 correct 50 time per epoch 1.8850407600402832
Epoch  390  loss  0.2646685474088007 correct 49 time per epoch 1.9740478992462158
Epoch  400  loss  0.8688384610814861 correct 49 time per epoch 1.9814252853393555
Epoch  410  loss  1.0445510184870215 correct 49 time per epoch 1.925135850906372
Epoch  420  loss  2.5535168520016294 correct 45 time per epoch 2.1093854904174805
Epoch  430  loss  0.13260308277959745 correct 50 time per epoch 1.9276180267333984
Epoch  440  loss  1.2078962860559277 correct 48 time per epoch 1.979560375213623
Epoch  450  loss  0.8582849296916533 correct 49 time per epoch 2.518927574157715
Epoch  460  loss  2.406730635986636 correct 48 time per epoch 1.911318063735962
Epoch  470  loss  0.2029219413154943 correct 49 time per epoch 1.894622564315796
Epoch  480  loss  0.6011686350005646 correct 49 time per epoch 2.777606964111328
Epoch  490  loss  0.2127468176454351 correct 50 time per epoch 1.92478609085083
```




# Training Log Results / Scripts for Task 3_5 for BIG Model:
## Dataset: XOR
### CPU
```bash
!python project/run_fast_tensor.py --BACKEND cpu --HIDDEN 200 --DATASET xor --RATE 0.05
```
Number of Points: 50 <br>
Size of Hidden Layer: 200 <br>
Number of Epochs: 500 <br>
Learning Rate: 0.05 <br>
```
Epoch  0  loss  11.046790236633907 correct 31 time per epoch 29.33056902885437
Epoch  10  loss  5.137631119958359 correct 45 time per epoch 0.2962794303894043
Epoch  20  loss  1.5115268201555176 correct 42 time per epoch 0.28212523460388184
Epoch  30  loss  1.6835400800150186 correct 45 time per epoch 0.2846078872680664
Epoch  40  loss  2.890332679259843 correct 45 time per epoch 0.2945516109466553
Epoch  50  loss  2.581235307838582 correct 47 time per epoch 0.28324222564697266
Epoch  60  loss  2.3658229515236746 correct 47 time per epoch 0.28623080253601074
Epoch  70  loss  0.46602315541181644 correct 46 time per epoch 0.2885313034057617
Epoch  80  loss  2.5986143487987396 correct 46 time per epoch 0.2848536968231201
Epoch  90  loss  0.8906051225227625 correct 47 time per epoch 0.28232598304748535
Epoch  100  loss  3.2377641872505043 correct 47 time per epoch 0.3031001091003418
Epoch  110  loss  2.5423853265941663 correct 49 time per epoch 0.28447604179382324
Epoch  120  loss  1.002198352144503 correct 48 time per epoch 0.2811245918273926
Epoch  130  loss  1.9816633529620435 correct 49 time per epoch 0.2937805652618408
Epoch  140  loss  0.5976751881103312 correct 50 time per epoch 0.29532527923583984
Epoch  150  loss  0.7123362178310474 correct 49 time per epoch 0.28647661209106445
Epoch  160  loss  2.005914205117951 correct 49 time per epoch 0.28016090393066406
Epoch  170  loss  0.9524175969010535 correct 50 time per epoch 0.2955660820007324
Epoch  180  loss  1.414784669053581 correct 49 time per epoch 0.29291200637817383
Epoch  190  loss  0.29403443638668647 correct 49 time per epoch 0.2857334613800049
Epoch  200  loss  0.7363304209205883 correct 49 time per epoch 0.3196084499359131
Epoch  210  loss  2.1197486689418383 correct 48 time per epoch 0.28421616554260254
Epoch  220  loss  1.1778155014555727 correct 50 time per epoch 0.3399965763092041
Epoch  230  loss  0.1973354377315161 correct 50 time per epoch 0.304706335067749
Epoch  240  loss  0.40673445616436205 correct 50 time per epoch 0.28145813941955566
Epoch  250  loss  0.6438390288440644 correct 50 time per epoch 0.28087878227233887
Epoch  260  loss  0.8319790171286416 correct 49 time per epoch 0.4701516628265381
Epoch  270  loss  1.0995284225599242 correct 49 time per epoch 0.2963733673095703
Epoch  280  loss  1.0871526200274273 correct 50 time per epoch 0.28341007232666016
Epoch  290  loss  0.44485764645540926 correct 50 time per epoch 0.2912611961364746
Epoch  300  loss  0.26350521186244225 correct 49 time per epoch 0.564849853515625
Epoch  310  loss  0.08952002460858789 correct 49 time per epoch 0.2827625274658203
Epoch  320  loss  0.5684178717412395 correct 50 time per epoch 0.2946512699127197
Epoch  330  loss  0.5682152271812964 correct 50 time per epoch 0.2822732925415039
Epoch  340  loss  0.5719682226544385 correct 50 time per epoch 0.5874745845794678
Epoch  350  loss  0.40257127954244176 correct 49 time per epoch 0.2987630367279053
Epoch  360  loss  1.4271511985910306 correct 49 time per epoch 0.28512001037597656
Epoch  370  loss  0.1054049838219111 correct 50 time per epoch 0.2839813232421875
Epoch  380  loss  0.11811303412757215 correct 50 time per epoch 0.5836887359619141
Epoch  390  loss  0.11997125632890208 correct 50 time per epoch 0.28145813941955566
Epoch  400  loss  1.1324695204926087 correct 49 time per epoch 0.2823460102081299
Epoch  410  loss  0.1866643743803394 correct 50 time per epoch 0.28018689155578613
Epoch  420  loss  0.46626263019495384 correct 49 time per epoch 0.5189547538757324
Epoch  430  loss  0.7385354408460985 correct 50 time per epoch 0.2790710926055908
Epoch  440  loss  0.4587357797718838 correct 50 time per epoch 0.28610920906066895
Epoch  450  loss  0.3399960689246787 correct 50 time per epoch 0.29938244819641113
Epoch  460  loss  0.5413601862354342 correct 50 time per epoch 0.5626306533813477
Epoch  470  loss  0.21741903985671648 correct 50 time per epoch 0.28055500984191895
Epoch  480  loss  0.22561552351721778 correct 50 time per epoch 0.2798182964324951
Epoch  490  loss  0.34785229951013447 correct 50 time per epoch 0.2807753086090088
```

### GPU
```bash
!python project/run_fast_tensor.py --BACKEND gpu --HIDDEN 200 --DATASET xor --RATE 0.05
```
Number of Points: 50 <br>
Size of Hidden Layer: 200 <br>
Number of Epochs: 500 <br>
Learning Rate: 0.05 <br>
```
Epoch  0  loss  7.5582730347609886 correct 25 time per epoch 4.331260919570923
Epoch  10  loss  2.4499099205395454 correct 47 time per epoch 2.009685516357422
Epoch  20  loss  1.8573320042751973 correct 48 time per epoch 2.5933711528778076
Epoch  30  loss  1.2692166872939008 correct 48 time per epoch 2.0150556564331055
Epoch  40  loss  2.1814474407971884 correct 49 time per epoch 1.9494102001190186
Epoch  50  loss  0.6419468835357629 correct 49 time per epoch 2.060699462890625
Epoch  60  loss  3.138751571065228 correct 46 time per epoch 1.9735021591186523
Epoch  70  loss  0.5571726757556184 correct 49 time per epoch 1.957369089126587
Epoch  80  loss  0.4945743035057465 correct 49 time per epoch 1.9957695007324219
Epoch  90  loss  1.1141852234419867 correct 49 time per epoch 2.396120071411133
Epoch  100  loss  1.5131659725546762 correct 47 time per epoch 1.9425077438354492
Epoch  110  loss  0.6781577210547419 correct 48 time per epoch 1.9745492935180664
Epoch  120  loss  0.8993694113621022 correct 49 time per epoch 2.5468363761901855
Epoch  130  loss  0.3613452604009711 correct 49 time per epoch 1.9479098320007324
Epoch  140  loss  0.4324325642816964 correct 49 time per epoch 1.9503529071807861
Epoch  150  loss  1.0792365124763539 correct 49 time per epoch 1.9427666664123535
Epoch  160  loss  0.7147966680089949 correct 49 time per epoch 2.024935483932495
Epoch  170  loss  0.07449628953984042 correct 49 time per epoch 2.050154447555542
Epoch  180  loss  1.259684656057137 correct 49 time per epoch 1.954024076461792
Epoch  190  loss  0.8845224137394783 correct 48 time per epoch 2.6610562801361084
Epoch  200  loss  1.0603689237669638 correct 49 time per epoch 2.0329110622406006
Epoch  210  loss  1.127138955931907 correct 50 time per epoch 1.9546945095062256
Epoch  220  loss  0.8457615859598882 correct 49 time per epoch 2.1038661003112793
Epoch  230  loss  1.2557834021212828 correct 49 time per epoch 1.9481356143951416
Epoch  240  loss  0.05729871156577625 correct 50 time per epoch 1.941288709640503
Epoch  250  loss  0.29911772209206733 correct 50 time per epoch 1.949523687362671
Epoch  260  loss  0.6559762454907972 correct 50 time per epoch 2.5857644081115723
Epoch  270  loss  0.16889534461746625 correct 49 time per epoch 1.9901232719421387
Epoch  280  loss  1.0716947015929408 correct 50 time per epoch 1.9737775325775146
Epoch  290  loss  0.5568002913751056 correct 49 time per epoch 2.360858201980591
Epoch  300  loss  0.2470489264835658 correct 50 time per epoch 1.9715168476104736
Epoch  310  loss  0.7341582145961889 correct 50 time per epoch 2.051875591278076
Epoch  320  loss  0.4761425170791252 correct 49 time per epoch 1.964310646057129
Epoch  330  loss  0.06705885970150369 correct 50 time per epoch 2.4478909969329834
Epoch  340  loss  0.5149972109521497 correct 49 time per epoch 1.9763462543487549
Epoch  350  loss  1.0321668776962212 correct 50 time per epoch 2.0267245769500732
Epoch  360  loss  0.22552308582489827 correct 49 time per epoch 2.5648880004882812
Epoch  370  loss  0.07474101106115587 correct 50 time per epoch 1.950404405593872
Epoch  380  loss  0.33123829705598984 correct 50 time per epoch 1.9388315677642822
Epoch  390  loss  0.24489459252112533 correct 49 time per epoch 2.063502073287964
Epoch  400  loss  0.3052060428705405 correct 50 time per epoch 2.0172693729400635
Epoch  410  loss  0.12502576482296412 correct 50 time per epoch 1.9588160514831543
Epoch  420  loss  0.20498816208243464 correct 50 time per epoch 1.9725089073181152
Epoch  430  loss  0.03036123826704442 correct 50 time per epoch 2.461699962615967
Epoch  440  loss  0.009551603869916526 correct 50 time per epoch 2.043578624725342
Epoch  450  loss  0.05238268239105418 correct 49 time per epoch 1.9580016136169434
Epoch  460  loss  0.4445171295026314 correct 50 time per epoch 2.496366500854492
Epoch  470  loss  0.19881790923885842 correct 50 time per epoch 1.9358994960784912
Epoch  480  loss  0.21277062361202914 correct 50 time per epoch 2.005868434906006
Epoch  490  loss  0.24769145026421505 correct 50 time per epoch 1.9506199359893799
```
