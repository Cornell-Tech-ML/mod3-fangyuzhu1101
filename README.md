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
/usr/local/lib/python3.10/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 2 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.10/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
Running size 64
/usr/local/lib/python3.10/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 8 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
{'fast': np.float64(0.003516117731730143), 'gpu': np.float64(0.00702659289042155)}
Running size 128
/usr/local/lib/python3.10/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 32 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
{'fast': np.float64(0.016291379928588867), 'gpu': np.float64(0.015493392944335938)}
Running size 256
{'fast': np.float64(0.10075187683105469), 'gpu': np.float64(0.055118163426717125)}
Running size 512
{'fast': np.float64(1.0496012369791667), 'gpu': np.float64(0.2842566172281901)}
Running size 1024
{'fast': np.float64(8.341062307357788), 'gpu': np.float64(1.0091346899668376)}

Timing summary
Size: 64
    fast: 0.00352
    gpu: 0.00703
Size: 128
    fast: 0.01629
    gpu: 0.01549
Size: 256
    fast: 0.10075
    gpu: 0.05512
Size: 512
    fast: 1.04960
    gpu: 0.28426
Size: 1024
    fast: 8.34106
    gpu: 1.00913
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
Epoch  0  loss  6.055453497407641  correct  49  time per epoch 29.641653776168823
Epoch  10  loss  1.5325939674289142  correct  48  time per epoch 0.15250444412231445
Epoch  20  loss  1.0542389550304883  correct  48  time per epoch 0.13964152336120605
Epoch  30  loss  1.6063340620384357  correct  50  time per epoch 0.33839917182922363
Epoch  40  loss  0.5106178794457054  correct  50  time per epoch 0.1522974967956543
Epoch  50  loss  0.7538813239231393  correct  50  time per epoch 0.14200234413146973
Epoch  60  loss  0.6197642912195378  correct  50  time per epoch 0.14045095443725586
Epoch  70  loss  0.8444165319036342  correct  50  time per epoch 0.13904094696044922
Epoch  80  loss  0.3148761247160681  correct  50  time per epoch 0.14426183700561523
Epoch  90  loss  0.5697950774751478  correct  50  time per epoch 0.1399545669555664
Epoch  100  loss  0.4805030552507293  correct  50  time per epoch 0.14220857620239258
Epoch  110  loss  0.19252043942939973  correct  50  time per epoch 0.28334832191467285
Epoch  120  loss  0.029478836106127046  correct  50  time per epoch 0.14001727104187012
Epoch  130  loss  0.3457644914908328  correct  50  time per epoch 0.14234280586242676
Epoch  140  loss  0.4917489909350409  correct  50  time per epoch 0.14195013046264648
Epoch  150  loss  0.15686647309711427  correct  50  time per epoch 0.14156436920166016
Epoch  160  loss  0.1380244568298112  correct  50  time per epoch 0.1422443389892578
Epoch  170  loss  0.2081622605365725  correct  50  time per epoch 0.1489579677581787
Epoch  180  loss  0.39358081429089137  correct  50  time per epoch 0.14189672470092773
Epoch  190  loss  0.1330971033907231  correct  50  time per epoch 0.32007360458374023
Epoch  200  loss  0.041784998767833155  correct  50  time per epoch 0.13933014869689941
Epoch  210  loss  0.0007343722713154064  correct  50  time per epoch 0.14021968841552734
Epoch  220  loss  0.12228062102092654  correct  50  time per epoch 0.15865230560302734
Epoch  230  loss  0.18399519514080803  correct  50  time per epoch 0.14336872100830078
Epoch  240  loss  0.13469011307472034  correct  50  time per epoch 0.14084553718566895
Epoch  250  loss  0.2752352400447486  correct  50  time per epoch 0.14910626411437988
Epoch  260  loss  0.19867769125467655  correct  50  time per epoch 0.1414189338684082
Epoch  270  loss  0.02573994666935293  correct  50  time per epoch 0.2491898536682129
Epoch  280  loss  0.039727284067585  correct  50  time per epoch 0.3018627166748047
Epoch  290  loss  0.021602604809820965  correct  50  time per epoch 0.1408684253692627
Epoch  300  loss  0.08271984835069957  correct  50  time per epoch 0.14020705223083496
Epoch  310  loss  0.05155212024393192  correct  50  time per epoch 0.1425034999847412
Epoch  320  loss  0.051334836622053  correct  50  time per epoch 0.14212775230407715
Epoch  330  loss  0.014839300535507105  correct  50  time per epoch 0.1421947479248047
Epoch  340  loss  0.12265687298861395  correct  50  time per epoch 0.13978004455566406
Epoch  350  loss  0.0280065140347287  correct  50  time per epoch 0.2308332920074463
Epoch  360  loss  0.1226707340741552  correct  50  time per epoch 0.1410965919494629
Epoch  370  loss  0.01807702319332543  correct  50  time per epoch 0.13890361785888672
Epoch  380  loss  0.06603454368635868  correct  50  time per epoch 0.14190340042114258
Epoch  390  loss  0.03729644656689176  correct  50  time per epoch 0.14562678337097168
Epoch  400  loss  0.07022774272019236  correct  50  time per epoch 0.1417827606201172
Epoch  410  loss  0.019890902807366686  correct  50  time per epoch 0.14052700996398926
Epoch  420  loss  0.10184616849812977  correct  50  time per epoch 0.1470940113067627
Epoch  430  loss  0.03156199349114189  correct  50  time per epoch 0.26030993461608887
Epoch  440  loss  0.04150444709530793  correct  50  time per epoch 0.1407759189605713
Epoch  450  loss  0.014910605060474976  correct  50  time per epoch 0.15066838264465332
Epoch  460  loss  0.14002141749636313  correct  50  time per epoch 0.14026951789855957
Epoch  470  loss  0.08777351335644414  correct  50  time per epoch 0.14022159576416016
Epoch  480  loss  0.004853236467213432  correct  50  time per epoch 0.13997554779052734
Epoch  490  loss  0.038965467653502284  correct  50  time per epoch 0.14061212539672852
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
Epoch  0  loss  6.296293638091872  correct  38  time per epoch 5.29325270652771
Epoch  10  loss  3.196950379900904  correct  48  time per epoch 1.921257495880127
Epoch  20  loss  0.9985483281661072  correct  49  time per epoch 2.684199094772339
Epoch  30  loss  1.2340904227433684  correct  49  time per epoch 1.974893569946289
Epoch  40  loss  2.0983168300156647  correct  49  time per epoch 1.9148104190826416
Epoch  50  loss  0.4719580385926751  correct  49  time per epoch 2.7618560791015625
Epoch  60  loss  1.9384247486217439  correct  49  time per epoch 1.9049360752105713
Epoch  70  loss  0.46743056437018904  correct  49  time per epoch 1.8913803100585938
Epoch  80  loss  0.860718286606348  correct  49  time per epoch 2.5183467864990234
Epoch  90  loss  0.3893300146640979  correct  49  time per epoch 1.9147472381591797
Epoch  100  loss  0.13108204414691205  correct  50  time per epoch 1.896594762802124
Epoch  110  loss  0.6617019218212514  correct  50  time per epoch 2.6682510375976562
Epoch  120  loss  0.2506581134202516  correct  50  time per epoch 1.944800615310669
Epoch  130  loss  0.19822673360672907  correct  50  time per epoch 2.0071308612823486
Epoch  140  loss  0.12240608875189947  correct  50  time per epoch 2.4426679611206055
Epoch  150  loss  0.1586346741137517  correct  49  time per epoch 1.9200727939605713
Epoch  160  loss  0.5506615941367643  correct  50  time per epoch 1.8947792053222656
Epoch  170  loss  0.28868650002001706  correct  50  time per epoch 2.244100570678711
Epoch  180  loss  0.04304982778782786  correct  50  time per epoch 2.0046606063842773
Epoch  190  loss  0.002074513859731797  correct  50  time per epoch 2.1576828956604004
Epoch  200  loss  0.6096682163569964  correct  50  time per epoch 1.921687364578247
Epoch  210  loss  0.0975299608988788  correct  50  time per epoch 1.915780782699585
Epoch  220  loss  0.047857215822245044  correct  50  time per epoch 2.043987989425659
Epoch  230  loss  0.01639517239311085  correct  50  time per epoch 1.9084157943725586
Epoch  240  loss  0.2297581567939046  correct  50  time per epoch 1.9412260055541992
Epoch  250  loss  0.05615925310819191  correct  50  time per epoch 2.1933305263519287
Epoch  260  loss  0.21430440189218924  correct  50  time per epoch 2.004666566848755
Epoch  270  loss  0.016725427259673135  correct  50  time per epoch 1.9248576164245605
Epoch  280  loss  0.3531864660780067  correct  50  time per epoch 2.441735029220581
Epoch  290  loss  0.09285611336702959  correct  50  time per epoch 1.925328016281128
Epoch  300  loss  0.10498590053151269  correct  50  time per epoch 1.9042394161224365
Epoch  310  loss  0.26563405688498104  correct  50  time per epoch 2.451421022415161
Epoch  320  loss  0.7647167786610666  correct  50  time per epoch 1.9097833633422852
Epoch  330  loss  0.18403782505824365  correct  50  time per epoch 1.9146599769592285
Epoch  340  loss  0.4902990599307692  correct  50  time per epoch 2.5497119426727295
Epoch  350  loss  0.03681607969089161  correct  50  time per epoch 1.9885554313659668
Epoch  360  loss  0.286132188788933  correct  50  time per epoch 1.9931938648223877
Epoch  370  loss  0.2843692781651235  correct  50  time per epoch 2.647642135620117
Epoch  380  loss  0.267430499697402  correct  50  time per epoch 1.9319722652435303
Epoch  390  loss  0.00816178409099148  correct  50  time per epoch 1.9114818572998047
Epoch  400  loss  0.7018795744130781  correct  50  time per epoch 2.7842087745666504
Epoch  410  loss  0.4168364889458648  correct  50  time per epoch 1.920851230621338
Epoch  420  loss  0.1082909399618562  correct  50  time per epoch 1.9566054344177246
Epoch  430  loss  0.021768146879661145  correct  50  time per epoch 2.542206048965454
Epoch  440  loss  0.13097710634961435  correct  50  time per epoch 1.9899461269378662
Epoch  450  loss  0.3397164899464814  correct  50  time per epoch 1.9493942260742188
Epoch  460  loss  0.07517917904771151  correct  50  time per epoch 2.6559345722198486
Epoch  470  loss  0.007546557251917934  correct  50  time per epoch 1.9315454959869385
Epoch  480  loss  0.31015573219116577  correct  50  time per epoch 1.936788558959961
Epoch  490  loss  0.013480602841298818  correct  50  time per epoch 2.7739882469177246
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
Epoch  0  loss  9.825954806354392  correct  31  time per epoch 29.579895973205566
Epoch  10  loss  6.31980103990808  correct  42  time per epoch 0.2945723533630371
Epoch  20  loss  7.380934157032588  correct  37  time per epoch 0.14773869514465332
Epoch  30  loss  4.195871685669423  correct  46  time per epoch 0.14298486709594727
Epoch  40  loss  3.7863430205753477  correct  47  time per epoch 0.13978958129882812
Epoch  50  loss  5.151507301595434  correct  39  time per epoch 0.15570974349975586
Epoch  60  loss  2.875683540681366  correct  49  time per epoch 0.14364337921142578
Epoch  70  loss  2.4406874414217063  correct  50  time per epoch 0.14255547523498535
Epoch  80  loss  1.0108647051891393  correct  47  time per epoch 0.14612197875976562
Epoch  90  loss  1.0158623326225116  correct  49  time per epoch 0.17914128303527832
Epoch  100  loss  1.791702866785803  correct  49  time per epoch 0.1418626308441162
Epoch  110  loss  1.6224359827191779  correct  49  time per epoch 0.15391302108764648
Epoch  120  loss  1.4987937116018335  correct  50  time per epoch 0.13979887962341309
Epoch  130  loss  1.2104351280381171  correct  50  time per epoch 0.14137530326843262
Epoch  140  loss  1.1580656037195816  correct  49  time per epoch 0.14843058586120605
Epoch  150  loss  1.7884871750885656  correct  50  time per epoch 0.1410682201385498
Epoch  160  loss  1.2893915906976323  correct  50  time per epoch 0.14194989204406738
Epoch  170  loss  0.32711542224843115  correct  50  time per epoch 0.14931082725524902
Epoch  180  loss  0.8593281872553129  correct  50  time per epoch 0.2599296569824219
Epoch  190  loss  1.6467065857167862  correct  50  time per epoch 0.14099931716918945
Epoch  200  loss  1.1928241634539754  correct  50  time per epoch 0.15182042121887207
Epoch  210  loss  1.0974995471901634  correct  50  time per epoch 0.14052343368530273
Epoch  220  loss  0.999221371737279  correct  50  time per epoch 0.14198923110961914
Epoch  230  loss  1.4224915521154016  correct  50  time per epoch 0.15605974197387695
Epoch  240  loss  0.7837239723093711  correct  50  time per epoch 0.14257001876831055
Epoch  250  loss  0.7907786242692152  correct  50  time per epoch 0.14264631271362305
Epoch  260  loss  0.6816158692124433  correct  50  time per epoch 0.2632579803466797
Epoch  270  loss  0.33532401735942935  correct  50  time per epoch 0.14060688018798828
Epoch  280  loss  0.5668671724554836  correct  50  time per epoch 0.14037799835205078
Epoch  290  loss  0.7653732704942944  correct  50  time per epoch 0.1452045440673828
Epoch  300  loss  0.9768512941490917  correct  50  time per epoch 0.1448974609375
Epoch  310  loss  0.18460121104789343  correct  50  time per epoch 0.14162015914916992
Epoch  320  loss  0.2169916515209936  correct  50  time per epoch 0.14098095893859863
Epoch  330  loss  1.0758307582060596  correct  50  time per epoch 0.14469170570373535
Epoch  340  loss  0.5295604693757543  correct  50  time per epoch 0.2960054874420166
Epoch  350  loss  0.3985661561727871  correct  50  time per epoch 0.1653904914855957
Epoch  360  loss  0.755133677869855  correct  50  time per epoch 0.1383974552154541
Epoch  370  loss  0.3624417304935474  correct  50  time per epoch 0.15178322792053223
Epoch  380  loss  0.2527697719389962  correct  50  time per epoch 0.1487412452697754
Epoch  390  loss  0.4074423330564112  correct  50  time per epoch 0.1392383575439453
Epoch  400  loss  0.07650210796449657  correct  50  time per epoch 0.14081668853759766
Epoch  410  loss  0.5373006818042021  correct  50  time per epoch 0.1431725025177002
Epoch  420  loss  0.10208009364477084  correct  50  time per epoch 0.19531941413879395
Epoch  430  loss  0.14849558849838085  correct  50  time per epoch 0.14036059379577637
Epoch  440  loss  0.36290653443228926  correct  50  time per epoch 0.13937830924987793
Epoch  450  loss  0.44303570029381584  correct  50  time per epoch 0.14053773880004883
Epoch  460  loss  0.35391496586579496  correct  50  time per epoch 0.14159488677978516
Epoch  470  loss  0.1703339621635172  correct  50  time per epoch 0.14392733573913574
Epoch  480  loss  0.1765887313116027  correct  50  time per epoch 0.1434180736541748
Epoch  490  loss  0.07102328928360557  correct  50  time per epoch 0.1398909091949463
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
Epoch  0  loss  7.296362864445331  correct  30  time per epoch 5.2170634269714355
Epoch  10  loss  9.442917389036225  correct  20  time per epoch 1.919832706451416
Epoch  20  loss  4.831018762636433  correct  41  time per epoch 1.912672519683838
Epoch  30  loss  6.9992713195968195  correct  41  time per epoch 2.422185182571411
Epoch  40  loss  3.40360531536953  correct  42  time per epoch 1.9039554595947266
Epoch  50  loss  2.920138627989596  correct  41  time per epoch 1.9183971881866455
Epoch  60  loss  3.7615534163739954  correct  47  time per epoch 2.5172274112701416
Epoch  70  loss  4.072402597258538  correct  46  time per epoch 1.9049897193908691
Epoch  80  loss  3.3142576932077574  correct  49  time per epoch 1.995239019393921
Epoch  90  loss  1.560238196835904  correct  45  time per epoch 2.7363221645355225
Epoch  100  loss  2.743770506058837  correct  49  time per epoch 1.8939094543457031
Epoch  110  loss  3.153137690283031  correct  48  time per epoch 1.9097073078155518
Epoch  120  loss  2.4857469793497136  correct  47  time per epoch 2.6201319694519043
Epoch  130  loss  1.6296013196671089  correct  50  time per epoch 1.9760479927062988
Epoch  140  loss  1.7363682819620032  correct  48  time per epoch 1.9402484893798828
Epoch  150  loss  1.867960977577727  correct  50  time per epoch 2.7892258167266846
Epoch  160  loss  0.8751097948210038  correct  48  time per epoch 1.9029383659362793
Epoch  170  loss  1.668804705814108  correct  49  time per epoch 1.9488935470581055
Epoch  180  loss  1.3894995597596604  correct  48  time per epoch 2.560696601867676
Epoch  190  loss  3.3908007661368553  correct  47  time per epoch 1.9119443893432617
Epoch  200  loss  0.906544954098758  correct  50  time per epoch 1.9496288299560547
Epoch  210  loss  1.1402533080176203  correct  47  time per epoch 2.341169834136963
Epoch  220  loss  0.404076538413039  correct  49  time per epoch 1.974822759628296
Epoch  230  loss  1.534304984417813  correct  50  time per epoch 1.9134187698364258
Epoch  240  loss  1.021465184120353  correct  49  time per epoch 2.596318483352661
Epoch  250  loss  0.5132492853903572  correct  50  time per epoch 1.9126386642456055
Epoch  260  loss  1.2076343928980495  correct  49  time per epoch 1.959977149963379
Epoch  270  loss  0.5466240994786459  correct  47  time per epoch 2.2823405265808105
Epoch  280  loss  0.8677445701195399  correct  50  time per epoch 1.9281930923461914
Epoch  290  loss  1.0016855358286383  correct  50  time per epoch 1.8849663734436035
Epoch  300  loss  0.6342440789726351  correct  50  time per epoch 2.0249946117401123
Epoch  310  loss  1.0272909797463374  correct  49  time per epoch 1.9716129302978516
Epoch  320  loss  0.8099216113778864  correct  50  time per epoch 1.8991189002990723
Epoch  330  loss  0.13902345779866276  correct  50  time per epoch 1.8937509059906006
Epoch  340  loss  0.4561303964917934  correct  50  time per epoch 1.907392978668213
Epoch  350  loss  0.4280014204757084  correct  50  time per epoch 2.00071382522583
Epoch  360  loss  1.4051438565484475  correct  49  time per epoch 2.191462278366089
Epoch  370  loss  0.2843512776806362  correct  50  time per epoch 1.929800271987915
Epoch  380  loss  0.42035811356820924  correct  50  time per epoch 1.9587671756744385
Epoch  390  loss  0.5905690384142647  correct  50  time per epoch 1.9023160934448242
Epoch  400  loss  0.16224269343922879  correct  50  time per epoch 1.971376895904541
Epoch  410  loss  0.6582117545350287  correct  50  time per epoch 1.9722559452056885
Epoch  420  loss  0.9570837634957052  correct  50  time per epoch 1.898684024810791
Epoch  430  loss  0.8866613199513435  correct  49  time per epoch 1.9099493026733398
Epoch  440  loss  0.47064668341336924  correct  49  time per epoch 2.2926228046417236
Epoch  450  loss  0.19742029119630986  correct  50  time per epoch 1.905956506729126
Epoch  460  loss  0.2280460440440791  correct  49  time per epoch 1.9088265895843506
Epoch  470  loss  0.9671426549973349  correct  50  time per epoch 2.174873113632202
Epoch  480  loss  0.9091651163920705  correct  50  time per epoch 1.9071681499481201
Epoch  490  loss  0.5923070985287102  correct  50  time per epoch 1.9817304611206055
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
Epoch  0  loss  6.582564315112794  correct  22  time per epoch 29.38428568840027
Epoch  10  loss  5.751275197530983  correct  40  time per epoch 0.28057861328125
Epoch  20  loss  4.841675619175796  correct  41  time per epoch 0.14139223098754883
Epoch  30  loss  3.922240557251153  correct  44  time per epoch 0.13801956176757812
Epoch  40  loss  3.797166930263507  correct  45  time per epoch 0.15673398971557617
Epoch  50  loss  2.603383245025957  correct  45  time per epoch 0.1395401954650879
Epoch  60  loss  3.2653771743268853  correct  43  time per epoch 0.13906574249267578
Epoch  70  loss  3.315617826638091  correct  46  time per epoch 0.15513062477111816
Epoch  80  loss  2.1068522528095404  correct  46  time per epoch 0.13966107368469238
Epoch  90  loss  1.8753974386077552  correct  47  time per epoch 0.2034916877746582
Epoch  100  loss  1.9320509846364284  correct  45  time per epoch 0.15331745147705078
Epoch  110  loss  1.2282568090664918  correct  50  time per epoch 0.13875937461853027
Epoch  120  loss  1.4819156175442711  correct  49  time per epoch 0.1423342227935791
Epoch  130  loss  0.3934207898884474  correct  49  time per epoch 0.14629554748535156
Epoch  140  loss  1.2709253199981672  correct  50  time per epoch 0.13840818405151367
Epoch  150  loss  1.6573307515706395  correct  50  time per epoch 0.14027023315429688
Epoch  160  loss  0.7319649858559145  correct  49  time per epoch 0.13829374313354492
Epoch  170  loss  1.1011501152170449  correct  50  time per epoch 0.24184536933898926
Epoch  180  loss  0.8261118341042583  correct  49  time per epoch 0.14328217506408691
Epoch  190  loss  1.9559073402915776  correct  48  time per epoch 0.13892793655395508
Epoch  200  loss  0.8997467026839671  correct  50  time per epoch 0.14941096305847168
Epoch  210  loss  0.4913655117176599  correct  50  time per epoch 0.14108061790466309
Epoch  220  loss  1.4628796019924266  correct  49  time per epoch 0.13940215110778809
Epoch  230  loss  0.6084176315217755  correct  50  time per epoch 0.14292526245117188
Epoch  240  loss  0.9132311172671912  correct  49  time per epoch 0.14053130149841309
Epoch  250  loss  1.015067709570886  correct  50  time per epoch 0.14255046844482422
Epoch  260  loss  0.17029670174987063  correct  49  time per epoch 0.2025151252746582
Epoch  270  loss  0.45287689992618924  correct  50  time per epoch 0.14143919944763184
Epoch  280  loss  1.1926877559356401  correct  50  time per epoch 0.14026570320129395
Epoch  290  loss  1.702148252132338  correct  48  time per epoch 0.14115500450134277
Epoch  300  loss  1.2865049101280817  correct  49  time per epoch 0.14175701141357422
Epoch  310  loss  0.10947450983913538  correct  49  time per epoch 0.14096617698669434
Epoch  320  loss  0.5483200335311678  correct  50  time per epoch 0.13859057426452637
Epoch  330  loss  0.3970351477317502  correct  50  time per epoch 0.14030957221984863
Epoch  340  loss  0.44222033944244404  correct  50  time per epoch 0.28086328506469727
Epoch  350  loss  0.3882487801262849  correct  50  time per epoch 0.1406841278076172
Epoch  360  loss  0.24014209684935112  correct  50  time per epoch 0.14122653007507324
Epoch  370  loss  1.4171434300554373  correct  48  time per epoch 0.14654159545898438
Epoch  380  loss  1.319038851832158  correct  48  time per epoch 0.13990330696105957
Epoch  390  loss  0.8162107521855294  correct  50  time per epoch 0.14203119277954102
Epoch  400  loss  0.6289506522295171  correct  50  time per epoch 0.23543190956115723
Epoch  410  loss  0.16530586832751004  correct  50  time per epoch 0.2761037349700928
Epoch  420  loss  0.05514003214037218  correct  50  time per epoch 0.1371314525604248
Epoch  430  loss  0.30974266696086783  correct  50  time per epoch 0.1410994529724121
Epoch  440  loss  0.37258822973368977  correct  50  time per epoch 0.14080452919006348
Epoch  450  loss  0.398737137063372  correct  50  time per epoch 0.1438124179840088
Epoch  460  loss  0.32139927790764106  correct  50  time per epoch 0.14453506469726562
Epoch  470  loss  0.8022331588117132  correct  50  time per epoch 0.13949155807495117
Epoch  480  loss  0.1597431191963008  correct  49  time per epoch 0.14176225662231445
Epoch  490  loss  0.15521271528293817  correct  50  time per epoch 0.1404423713684082
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
Epoch  0  loss  8.423223645696899  correct  17  time per epoch 4.299160480499268
Epoch  10  loss  5.766353981755997  correct  41  time per epoch 2.6319823265075684
Epoch  20  loss  4.760755781812202  correct  44  time per epoch 1.901956558227539
Epoch  30  loss  2.851564755409071  correct  36  time per epoch 1.990736484527588
Epoch  40  loss  3.6789828136077984  correct  46  time per epoch 2.440455675125122
Epoch  50  loss  2.120717962515098  correct  45  time per epoch 1.9239559173583984
Epoch  60  loss  3.7082005342927107  correct  45  time per epoch 1.89664888381958
Epoch  70  loss  1.9140757100658292  correct  45  time per epoch 1.9268078804016113
Epoch  80  loss  3.200712668606391  correct  45  time per epoch 2.0026190280914307
Epoch  90  loss  2.8378441322680494  correct  45  time per epoch 1.9396634101867676
Epoch  100  loss  1.8858248612036357  correct  47  time per epoch 1.8966891765594482
Epoch  110  loss  3.258727981649688  correct  48  time per epoch 1.9005420207977295
Epoch  120  loss  1.846667154790834  correct  46  time per epoch 2.5137932300567627
Epoch  130  loss  1.082014137069436  correct  46  time per epoch 1.9579746723175049
Epoch  140  loss  1.2291222314986006  correct  48  time per epoch 1.8972339630126953
Epoch  150  loss  3.0494077892999187  correct  50  time per epoch 2.515479326248169
Epoch  160  loss  1.8081545482456116  correct  49  time per epoch 1.927865982055664
Epoch  170  loss  1.4618841150650295  correct  48  time per epoch 1.8874082565307617
Epoch  180  loss  1.9153875756888337  correct  49  time per epoch 2.5517220497131348
Epoch  190  loss  2.781060040329341  correct  48  time per epoch 1.8896160125732422
Epoch  200  loss  0.6098708548528339  correct  49  time per epoch 1.9013841152191162
Epoch  210  loss  0.5289968047381858  correct  50  time per epoch 1.9000039100646973
Epoch  220  loss  1.2618080122444852  correct  49  time per epoch 1.972057580947876
Epoch  230  loss  0.976200556682011  correct  49  time per epoch 2.1364986896514893
Epoch  240  loss  0.8435783960607105  correct  48  time per epoch 1.8995249271392822
Epoch  250  loss  0.5637442122299875  correct  49  time per epoch 1.9171912670135498
Epoch  260  loss  0.6720296178648586  correct  48  time per epoch 2.633310556411743
Epoch  270  loss  1.680031010756547  correct  48  time per epoch 1.8785064220428467
Epoch  280  loss  0.28477772974887294  correct  49  time per epoch 1.8868110179901123
Epoch  290  loss  1.2185836896845545  correct  50  time per epoch 2.0615460872650146
Epoch  300  loss  0.8204667662424279  correct  50  time per epoch 1.9085452556610107
Epoch  310  loss  1.2736262184548233  correct  50  time per epoch 1.9979569911956787
Epoch  320  loss  1.0502246307745564  correct  49  time per epoch 1.8779041767120361
Epoch  330  loss  0.5873998361608075  correct  48  time per epoch 1.8911323547363281
Epoch  340  loss  0.5699426792013771  correct  50  time per epoch 2.626896619796753
Epoch  350  loss  2.309134912998514  correct  49  time per epoch 1.9346349239349365
Epoch  360  loss  0.6700245552361833  correct  50  time per epoch 1.9639966487884521
Epoch  370  loss  2.3296827506123963  correct  50  time per epoch 2.2658588886260986
Epoch  380  loss  0.7654349963220412  correct  50  time per epoch 1.8782188892364502
Epoch  390  loss  1.0868586012564687  correct  50  time per epoch 1.892486810684204
Epoch  400  loss  1.0653772213682857  correct  50  time per epoch 2.145101547241211
Epoch  410  loss  1.1530646339897372  correct  48  time per epoch 1.8832440376281738
Epoch  420  loss  0.28157336742426764  correct  50  time per epoch 1.9184787273406982
Epoch  430  loss  1.834168859971879  correct  50  time per epoch 1.9211952686309814
Epoch  440  loss  0.35506669943161534  correct  50  time per epoch 1.9825844764709473
Epoch  450  loss  1.7757532467871426  correct  49  time per epoch 2.564380168914795
Epoch  460  loss  0.1255596617935536  correct  50  time per epoch 1.8827378749847412
Epoch  470  loss  1.1460285507400287  correct  48  time per epoch 1.90350341796875
Epoch  480  loss  0.8167653608421153  correct  50  time per epoch 2.255444049835205
Epoch  490  loss  0.15780286050326867  correct  50  time per epoch 1.962766170501709
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
Epoch  0  loss  10.552810699733755  correct  29  time per epoch 30.428833484649658
Epoch  10  loss  4.39979273567821  correct  45  time per epoch 0.28137683868408203
Epoch  20  loss  6.222284424110658  correct  38  time per epoch 0.2838103771209717
Epoch  30  loss  1.3602966857970868  correct  45  time per epoch 0.2936115264892578
Epoch  40  loss  4.404937588941485  correct  41  time per epoch 0.5097393989562988
Epoch  50  loss  2.0935785044543644  correct  46  time per epoch 0.2859015464782715
Epoch  60  loss  2.713558016779565  correct  47  time per epoch 0.2957572937011719
Epoch  70  loss  2.243980852616881  correct  46  time per epoch 0.3105733394622803
Epoch  80  loss  2.981556672119811  correct  45  time per epoch 0.5962865352630615
Epoch  90  loss  3.098591626953965  correct  42  time per epoch 0.28256988525390625
Epoch  100  loss  3.563016283671855  correct  48  time per epoch 0.2960813045501709
Epoch  110  loss  3.5594755933154496  correct  46  time per epoch 0.2931368350982666
Epoch  120  loss  3.360322227072642  correct  46  time per epoch 0.5846505165100098
Epoch  130  loss  1.8709994864481705  correct  49  time per epoch 0.2897651195526123
Epoch  140  loss  2.7270548558954735  correct  45  time per epoch 0.29209041595458984
Epoch  150  loss  0.6111139464018301  correct  48  time per epoch 0.28166866302490234
Epoch  160  loss  3.1426052719766884  correct  48  time per epoch 0.6003763675689697
Epoch  170  loss  3.518642944108538  correct  47  time per epoch 0.2947070598602295
Epoch  180  loss  1.5877813919956232  correct  45  time per epoch 0.2817838191986084
Epoch  190  loss  1.6126850289559953  correct  49  time per epoch 0.2866184711456299
Epoch  200  loss  0.8284981781485823  correct  49  time per epoch 0.6332621574401855
Epoch  210  loss  1.9256248387536876  correct  49  time per epoch 0.2842977046966553
Epoch  220  loss  2.3261569642385127  correct  47  time per epoch 0.287243127822876
Epoch  230  loss  1.214465931655418  correct  49  time per epoch 0.28569841384887695
Epoch  240  loss  2.0479188212711796  correct  47  time per epoch 0.5087502002716064
Epoch  250  loss  1.6742350748799166  correct  49  time per epoch 0.28215670585632324
Epoch  260  loss  3.093026168544612  correct  49  time per epoch 0.29148411750793457
Epoch  270  loss  1.3154071027675918  correct  48  time per epoch 0.30651068687438965
Epoch  280  loss  1.1887306373005722  correct  45  time per epoch 0.33915185928344727
Epoch  290  loss  0.8298497827930726  correct  49  time per epoch 0.29044651985168457
Epoch  300  loss  1.8013237388677017  correct  49  time per epoch 0.30201101303100586
Epoch  310  loss  1.5632507541974325  correct  49  time per epoch 0.2841799259185791
Epoch  320  loss  0.5016407042809233  correct  49  time per epoch 0.282745361328125
Epoch  330  loss  0.8845688093479337  correct  49  time per epoch 0.2829904556274414
Epoch  340  loss  1.6546645138869251  correct  47  time per epoch 0.29515600204467773
Epoch  350  loss  3.453897562762501  correct  44  time per epoch 0.28810572624206543
Epoch  360  loss  1.2479795992467657  correct  49  time per epoch 0.28362417221069336
Epoch  370  loss  1.2503954263111072  correct  49  time per epoch 0.2925682067871094
Epoch  380  loss  1.1704576553667736  correct  49  time per epoch 0.28266263008117676
Epoch  390  loss  1.5036608956419164  correct  50  time per epoch 0.2849771976470947
Epoch  400  loss  1.045680118599766  correct  49  time per epoch 0.28623127937316895
Epoch  410  loss  1.314809181830843  correct  50  time per epoch 0.3015775680541992
Epoch  420  loss  2.5440649627385  correct  49  time per epoch 0.2852456569671631
Epoch  430  loss  0.7035551477916738  correct  50  time per epoch 0.3101038932800293
Epoch  440  loss  1.4597181046011471  correct  50  time per epoch 0.28812217712402344
Epoch  450  loss  1.4296089547428592  correct  50  time per epoch 0.2829439640045166
Epoch  460  loss  0.1787217332141405  correct  50  time per epoch 0.2831716537475586
Epoch  470  loss  1.1330155001234765  correct  50  time per epoch 0.2816905975341797
Epoch  480  loss  1.0157262551726822  correct  50  time per epoch 0.2817997932434082
Epoch  490  loss  0.34438641122444913  correct  50  time per epoch 0.2868154048919678
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
Epoch  0  loss  18.33445429860606  correct  27  time per epoch 5.164613485336304
Epoch  10  loss  3.4441247466654406  correct  43  time per epoch 1.9572865962982178
Epoch  20  loss  3.1765152715430327  correct  31  time per epoch 1.962019920349121
Epoch  30  loss  2.4954697299081277  correct  47  time per epoch 2.0607433319091797
Epoch  40  loss  0.9496218542982041  correct  47  time per epoch 2.4064157009124756
Epoch  50  loss  2.3989433728795633  correct  47  time per epoch 1.9619405269622803
Epoch  60  loss  3.413719856976062  correct  46  time per epoch 1.9897680282592773
Epoch  70  loss  2.381565655305448  correct  48  time per epoch 2.691392183303833
Epoch  80  loss  2.660354968774216  correct  47  time per epoch 2.043222665786743
Epoch  90  loss  2.4287082535247224  correct  48  time per epoch 1.977492332458496
Epoch  100  loss  1.0121077795845521  correct  48  time per epoch 2.16451358795166
Epoch  110  loss  0.8385567485279248  correct  50  time per epoch 1.9736196994781494
Epoch  120  loss  2.491340434989916  correct  48  time per epoch 1.9560377597808838
Epoch  130  loss  1.7676071278998091  correct  49  time per epoch 2.0707015991210938
Epoch  140  loss  0.21214811666757225  correct  49  time per epoch 2.822059154510498
Epoch  150  loss  1.2074033140027483  correct  49  time per epoch 2.0021212100982666
Epoch  160  loss  0.8225433914000941  correct  47  time per epoch 1.9756524562835693
Epoch  170  loss  0.5422393902668969  correct  50  time per epoch 2.1615867614746094
Epoch  180  loss  0.2476701502925304  correct  49  time per epoch 2.0442757606506348
Epoch  190  loss  0.5158953781809927  correct  49  time per epoch 1.963273525238037
Epoch  200  loss  1.0595364727665468  correct  49  time per epoch 2.007188558578491
Epoch  210  loss  0.7596062241410294  correct  50  time per epoch 2.4792065620422363
Epoch  220  loss  0.26157958367188533  correct  50  time per epoch 2.0266504287719727
Epoch  230  loss  0.926446239574852  correct  49  time per epoch 1.9686014652252197
Epoch  240  loss  0.24371627741346719  correct  50  time per epoch 2.7017288208007812
Epoch  250  loss  0.9996633362630173  correct  49  time per epoch 2.004945755004883
Epoch  260  loss  0.1822214403251717  correct  50  time per epoch 2.0577316284179688
Epoch  270  loss  0.23402621149557884  correct  50  time per epoch 1.9943809509277344
Epoch  280  loss  0.4968192129063987  correct  50  time per epoch 2.2846388816833496
Epoch  290  loss  0.4257667575282168  correct  50  time per epoch 1.9893722534179688
Epoch  300  loss  0.5523081816477287  correct  49  time per epoch 2.0195658206939697
Epoch  310  loss  0.7213124626726911  correct  50  time per epoch 2.808286666870117
Epoch  320  loss  0.027469700989154657  correct  50  time per epoch 1.9713332653045654
Epoch  330  loss  0.027799713234249762  correct  49  time per epoch 1.9874699115753174
Epoch  340  loss  0.2283277585297141  correct  50  time per epoch 2.3340649604797363
Epoch  350  loss  0.49345017299338073  correct  49  time per epoch 2.0463967323303223
Epoch  360  loss  0.039879521121818584  correct  49  time per epoch 2.0337471961975098
Epoch  370  loss  0.3522202110869229  correct  49  time per epoch 1.9598186016082764
Epoch  380  loss  0.7693803623031036  correct  50  time per epoch 2.1389172077178955
Epoch  390  loss  0.3719415569809946  correct  49  time per epoch 1.9631803035736084
Epoch  400  loss  0.43316672990903665  correct  50  time per epoch 2.0207927227020264
Epoch  410  loss  0.02685382262694054  correct  50  time per epoch 2.6072309017181396
Epoch  420  loss  0.15749529240455948  correct  49  time per epoch 1.976989984512329
Epoch  430  loss  3.3252515809514227  correct  49  time per epoch 1.9988839626312256
Epoch  440  loss  0.9343970903991833  correct  49  time per epoch 2.794128656387329
Epoch  450  loss  0.32642695675164324  correct  50  time per epoch 1.992530107498169
Epoch  460  loss  0.0344128328638602  correct  50  time per epoch 1.9990363121032715
Epoch  470  loss  0.1486290321245423  correct  49  time per epoch 2.020720958709717
Epoch  480  loss  0.1446915928443576  correct  50  time per epoch 1.9579944610595703
Epoch  490  loss  0.32916343654617974  correct  50  time per epoch 2.619441270828247
```
