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
!python project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET Simple --RATE 0.05
```
Number of Points: 50 <br>
Size of Hidden Layer: 100 <br>
Number of Epochs: 500 <br>
Learning Rate: 0.05 <br>
Epoch 490/500. Time per epoch: 0.112s. Time left: 1.12s.
```
Epoch: 0/500, loss: 0, correct: 0
Epoch: 0/500, loss: 4.6276702323438, correct: 44
Epoch: 10/500, loss: 1.7982561155206835, correct: 49
Epoch: 20/500, loss: 0.7147380182748001, correct: 49
Epoch: 30/500, loss: 0.8246045791451226, correct: 49
Epoch: 40/500, loss: 0.4965707800014286, correct: 50
Epoch: 50/500, loss: 1.4369150030382798, correct: 50
Epoch: 60/500, loss: 0.5905338295903549, correct: 50
Epoch: 70/500, loss: 0.14430874006467503, correct: 50
Epoch: 80/500, loss: 0.29922619722855603, correct: 50
Epoch: 90/500, loss: 0.4347454454170408, correct: 50
Epoch: 100/500, loss: 0.0737695271958698, correct: 50
Epoch: 110/500, loss: 0.719898884949906, correct: 50
Epoch: 120/500, loss: 0.26461187549399656, correct: 50
Epoch: 130/500, loss: 0.07511754774016872, correct: 50
Epoch: 140/500, loss: 0.08369833365164363, correct: 50
Epoch: 150/500, loss: 0.4607291634625539, correct: 50
Epoch: 160/500, loss: 0.03914782878847417, correct: 50
Epoch: 170/500, loss: 0.43517339609261285, correct: 50
Epoch: 180/500, loss: 0.3108479001085624, correct: 50
Epoch: 190/500, loss: 0.06287667211692803, correct: 50
Epoch: 200/500, loss: 0.5905893631744293, correct: 50
Epoch: 210/500, loss: 0.35778923619085806, correct: 50
Epoch: 220/500, loss: 0.061549748121738856, correct: 50
Epoch: 230/500, loss: 0.0976863442475848, correct: 50
Epoch: 240/500, loss: 0.057953063737469064, correct: 50
Epoch: 250/500, loss: 0.11954083555473911, correct: 50
Epoch: 260/500, loss: 0.3627433319965475, correct: 50
Epoch: 270/500, loss: 0.0039648145505755674, correct: 50
Epoch: 280/500, loss: 0.05880347131367268, correct: 50
Epoch: 290/500, loss: 0.014477352076745918, correct: 50
Epoch: 300/500, loss: 0.01782242040635138, correct: 50
Epoch: 310/500, loss: 0.032804896434433975, correct: 50
Epoch: 320/500, loss: 0.12142826954734597, correct: 50
Epoch: 330/500, loss: 0.23988683423621282, correct: 50
Epoch: 340/500, loss: 0.3743869510235657, correct: 50
Epoch: 350/500, loss: 0.007013128317318239, correct: 50
Epoch: 360/500, loss: 0.18015942398872026, correct: 50
Epoch: 370/500, loss: 0.0018663887384009145, correct: 50
Epoch: 380/500, loss: 0.23991966399851136, correct: 50
Epoch: 390/500, loss: 0.34456641957992457, correct: 50
Epoch: 400/500, loss: 0.12276978706935687, correct: 50
Epoch: 410/500, loss: 0.010000449298664608, correct: 50
Epoch: 420/500, loss: 0.0018503649896064748, correct: 50
Epoch: 430/500, loss: 0.09584941442474963, correct: 50
Epoch: 440/500, loss: 0.15537220863968257, correct: 50
Epoch: 450/500, loss: 0.3085293661857787, correct: 50
Epoch: 460/500, loss: 0.14021641010758676, correct: 50
Epoch: 470/500, loss: 0.00011115119720785864, correct: 50
Epoch: 480/500, loss: 0.016479194752659736, correct: 50
Epoch: 490/500, loss: 0.01568572568008696, correct: 50
```

### GPU
```bash
!python project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET simple --RATE 0.05
```
Number of Points: 50 <br>
Size of Hidden Layer: 100 <br>
Number of Epochs: 500 <br>
Learning Rate: 0.05 <br>
Epoch 490/500. Time per epoch: 0.290s. Time left: 2.90s.
```
Epoch: 0/500, loss: 0, correct: 0
Epoch: 0/500, loss: 5.2747962549955405, correct: 45
Epoch: 10/500, loss: 1.1084247612454807, correct: 50
Epoch: 20/500, loss: 0.9124844089972648, correct: 50
Epoch: 30/500, loss: 1.1077607573744557, correct: 50
Epoch: 40/500, loss: 0.7958253793865593, correct: 50
Epoch: 50/500, loss: 0.4663311267712166, correct: 50
Epoch: 60/500, loss: 0.6218980719438741, correct: 50
Epoch: 70/500, loss: 0.11549471726911753, correct: 50
Epoch: 80/500, loss: 0.3623212772236521, correct: 50
Epoch: 90/500, loss: 0.1799883028280257, correct: 50
Epoch: 100/500, loss: 0.33035422990104, correct: 50
Epoch: 110/500, loss: 0.2193086545932469, correct: 50
Epoch: 120/500, loss: 0.03688749594622003, correct: 50
Epoch: 130/500, loss: 0.01993971354201848, correct: 50
Epoch: 140/500, loss: 0.1637146281362142, correct: 50
Epoch: 150/500, loss: 0.07935554855511621, correct: 50
Epoch: 160/500, loss: 0.10153498712183277, correct: 50
Epoch: 170/500, loss: 0.08344507007653806, correct: 50
Epoch: 180/500, loss: 0.021927144807524467, correct: 50
Epoch: 190/500, loss: 0.06088419409337123, correct: 50
Epoch: 200/500, loss: 0.02970357965416834, correct: 50
Epoch: 210/500, loss: 0.032723783357968945, correct: 50
Epoch: 220/500, loss: 0.04400002473117838, correct: 50
Epoch: 230/500, loss: 0.06885581393173641, correct: 50
Epoch: 240/500, loss: 0.01144354353149842, correct: 50
Epoch: 250/500, loss: 0.08827344745145653, correct: 50
Epoch: 260/500, loss: 0.07377613767906234, correct: 50
Epoch: 270/500, loss: 0.07129970710331017, correct: 50
Epoch: 280/500, loss: 0.07550800630786886, correct: 50
Epoch: 290/500, loss: 0.08151858086222336, correct: 50
Epoch: 300/500, loss: 0.001661691556728868, correct: 50
Epoch: 310/500, loss: 0.012758857034962793, correct: 50
Epoch: 320/500, loss: 0.06383599606890449, correct: 50
Epoch: 330/500, loss: 0.04196354904428731, correct: 50
Epoch: 340/500, loss: 0.10044864173667281, correct: 50
Epoch: 350/500, loss: 0.01036165577780735, correct: 50
Epoch: 360/500, loss: 0.02600189704657817, correct: 50
Epoch: 370/500, loss: 0.051517403594135446, correct: 50
Epoch: 380/500, loss: 0.026527715631740327, correct: 50
Epoch: 390/500, loss: 0.022188485363317436, correct: 50
Epoch: 400/500, loss: 0.012312122477060686, correct: 50
Epoch: 410/500, loss: 0.03530727207209809, correct: 50
Epoch: 420/500, loss: 0.019875389808557246, correct: 50
Epoch: 430/500, loss: 0.0009727437231678814, correct: 50
Epoch: 440/500, loss: 0.04104525854254347, correct: 50
Epoch: 450/500, loss: 0.001995974933608585, correct: 50
Epoch: 460/500, loss: 0.02202921549596331, correct: 50
Epoch: 470/500, loss: 0.014720985578636344, correct: 50
Epoch: 480/500, loss: 0.05052451418975928, correct: 50
Epoch: 490/500, loss: 0.060755392132336015, correct: 50
```



## Dataset: Split
### CPU
```bash
!python project/run_fast_tensor.py --BACKEND cpu --HIDDEN 150 --DATASET split --RATE 0.05
```
Number of Points: 50 <br>
Size of Hidden Layer: 150 <br>
Number of Epochs: 500 <br>
Learning Rate: 0.05 <br>
Epoch 490/500. Time per epoch: 0.147s. Time left: 1.47s.
```
Epoch: 0/500, loss: 0, correct: 0
Epoch: 0/500, loss: 14.858255178198212, correct: 26
Epoch: 10/500, loss: 12.746184630775302, correct: 30
Epoch: 20/500, loss: 5.936055646999712, correct: 43
Epoch: 30/500, loss: 3.1187965219107436, correct: 39
Epoch: 40/500, loss: 2.7379953882841144, correct: 46
Epoch: 50/500, loss: 1.459591344730414, correct: 45
Epoch: 60/500, loss: 2.7827694152207423, correct: 49
Epoch: 70/500, loss: 1.2853631563726418, correct: 49
Epoch: 80/500, loss: 1.2864794887271198, correct: 50
Epoch: 90/500, loss: 1.5210723127180215, correct: 49
Epoch: 100/500, loss: 1.841914971011223, correct: 49
Epoch: 110/500, loss: 0.9870919834924026, correct: 50
Epoch: 120/500, loss: 1.8507989691093214, correct: 50
Epoch: 130/500, loss: 0.8933511808956179, correct: 49
Epoch: 140/500, loss: 1.8679029080847724, correct: 49
Epoch: 150/500, loss: 0.5811012169209421, correct: 50
Epoch: 160/500, loss: 0.777377529340898, correct: 49
Epoch: 170/500, loss: 0.6173332956111323, correct: 50
Epoch: 180/500, loss: 0.36821735964273317, correct: 49
Epoch: 190/500, loss: 0.8736074911768809, correct: 50
Epoch: 200/500, loss: 0.14257544528119032, correct: 49
Epoch: 210/500, loss: 0.2975802012101334, correct: 50
Epoch: 220/500, loss: 0.4938057361298685, correct: 49
Epoch: 230/500, loss: 0.2586351558656067, correct: 48
Epoch: 240/500, loss: 0.3147175473054268, correct: 50
Epoch: 250/500, loss: 0.40162969607178844, correct: 50
Epoch: 260/500, loss: 0.11401334845022999, correct: 49
Epoch: 270/500, loss: 0.4916933252604225, correct: 49
Epoch: 280/500, loss: 0.5487689443520753, correct: 50
Epoch: 290/500, loss: 0.43106910759929745, correct: 50
Epoch: 300/500, loss: 0.18432352521061712, correct: 49
Epoch: 310/500, loss: 0.06468911528831438, correct: 49
Epoch: 320/500, loss: 0.24412865633535713, correct: 50
Epoch: 330/500, loss: 0.07207934123936373, correct: 50
Epoch: 340/500, loss: 0.08841571942643962, correct: 49
Epoch: 350/500, loss: 0.2156138303513979, correct: 50
Epoch: 360/500, loss: 0.19869778959200377, correct: 49
Epoch: 370/500, loss: 1.0223620904518746, correct: 50
Epoch: 380/500, loss: 0.8980127524367444, correct: 49
Epoch: 390/500, loss: 0.19640076022828404, correct: 49
Epoch: 400/500, loss: 0.25698731310513717, correct: 49
Epoch: 410/500, loss: 0.15861153502094752, correct: 49
Epoch: 420/500, loss: 0.7913847450554925, correct: 49
Epoch: 430/500, loss: 0.5275729712088097, correct: 50
Epoch: 440/500, loss: 1.1933230559325643, correct: 49
Epoch: 450/500, loss: 0.2490300407990459, correct: 49
Epoch: 460/500, loss: 0.19748368202885944, correct: 49
Epoch: 470/500, loss: 0.17115501083976253, correct: 49
Epoch: 480/500, loss: 0.25756705848299044, correct: 49
Epoch: 490/500, loss: 0.036421543899996335, correct: 49
```

### GPU
```bash
!python project/run_fast_tensor.py --BACKEND gpu --HIDDEN 150 --DATASET split --RATE 0.05
```
Number of Points: 50 <br>
Size of Hidden Layer: 150 <br>
Number of Epochs: 500 <br>
Learning Rate: 0.05 <br>
Epoch 490/500. Time per epoch: 0.395s. Time left: 3.95s.
```
Epoch: 0/500, loss: 0, correct: 0
Epoch: 0/500, loss: 6.734791630571907, correct: 31
Epoch: 10/500, loss: 3.653355172275024, correct: 38
Epoch: 20/500, loss: 5.5487460539633355, correct: 41
Epoch: 30/500, loss: 7.6574257507016, correct: 40
Epoch: 40/500, loss: 2.16038983544741, correct: 46
Epoch: 50/500, loss: 2.058291577465003, correct: 49
Epoch: 60/500, loss: 1.1125659968386672, correct: 45
Epoch: 70/500, loss: 2.64800529764191, correct: 48
Epoch: 80/500, loss: 2.18497075246375, correct: 49
Epoch: 90/500, loss: 1.3209724564619032, correct: 49
Epoch: 100/500, loss: 2.3189117428757164, correct: 48
Epoch: 110/500, loss: 1.3036912790781214, correct: 49
Epoch: 120/500, loss: 1.7305307049309806, correct: 46
Epoch: 130/500, loss: 0.5284364724610195, correct: 46
Epoch: 140/500, loss: 0.6569172498321681, correct: 49
Epoch: 150/500, loss: 1.5250400462785934, correct: 49
Epoch: 160/500, loss: 0.5499632501625453, correct: 45
Epoch: 170/500, loss: 1.6495958223951768, correct: 49
Epoch: 180/500, loss: 1.9025418247483814, correct: 50
Epoch: 190/500, loss: 1.2290858095042356, correct: 50
Epoch: 200/500, loss: 1.3313411995552977, correct: 48
Epoch: 210/500, loss: 1.0840153302175968, correct: 49
Epoch: 220/500, loss: 1.1868188458141535, correct: 48
Epoch: 230/500, loss: 0.7676541424441857, correct: 49
Epoch: 240/500, loss: 2.0888144766035115, correct: 48
Epoch: 250/500, loss: 0.7946745143331977, correct: 50
Epoch: 260/500, loss: 0.7770936894515288, correct: 49
Epoch: 270/500, loss: 1.2525489931329297, correct: 49
Epoch: 280/500, loss: 1.0740306890427942, correct: 49
Epoch: 290/500, loss: 0.8269253217304274, correct: 50
Epoch: 300/500, loss: 0.3367427595999616, correct: 48
Epoch: 310/500, loss: 2.3977314341746796, correct: 50
Epoch: 320/500, loss: 1.1164747069730714, correct: 48
Epoch: 330/500, loss: 1.7067202336091407, correct: 47
Epoch: 340/500, loss: 0.3068583762509663, correct: 50
Epoch: 350/500, loss: 1.4424468399949055, correct: 50
Epoch: 360/500, loss: 2.0680987849399104, correct: 50
Epoch: 370/500, loss: 2.0236662329489796, correct: 45
Epoch: 380/500, loss: 0.29178977795843986, correct: 48
Epoch: 390/500, loss: 0.9474237071923058, correct: 50
Epoch: 400/500, loss: 0.7738573765809104, correct: 49
Epoch: 410/500, loss: 0.23426687527319934, correct: 50
Epoch: 420/500, loss: 1.2622268255166553, correct: 49
Epoch: 430/500, loss: 1.1169899500771263, correct: 48
Epoch: 440/500, loss: 0.19565245304555304, correct: 50
Epoch: 450/500, loss: 1.077186283472917, correct: 49
Epoch: 460/500, loss: 0.36913334961790506, correct: 48
Epoch: 470/500, loss: 0.053220643110630225, correct: 50
Epoch: 480/500, loss: 0.30010593715003303, correct: 49
Epoch: 490/500, loss: 0.07281201666532089, correct: 49
```



## Dataset: XOR
### CPU
```bash
!python project/run_fast_tensor.py --BACKEND cpu --HIDDEN 150 --DATASET xor --RATE 0.05
```
Number of Points: 50 <br>
Size of Hidden Layer: 150 <br>
Number of Epochs: 500 <br>
Learning Rate: 0.05 <br>
Epoch 490/500. Time per epoch: 0.141s. Time left: 1.41s.
```
Epoch: 0/500, loss: 0, correct: 0
Epoch: 0/500, loss: 9.627507950007024, correct: 28
Epoch: 10/500, loss: 3.501394847433767, correct: 44
Epoch: 20/500, loss: 2.7214249547526106, correct: 42
Epoch: 30/500, loss: 6.350434765425493, correct: 45
Epoch: 40/500, loss: 2.4022406969484376, correct: 45
Epoch: 50/500, loss: 4.4644127119371655, correct: 42
Epoch: 60/500, loss: 1.2862319848138999, correct: 46
Epoch: 70/500, loss: 1.0312013221111878, correct: 46
Epoch: 80/500, loss: 2.452473849805261, correct: 46
Epoch: 90/500, loss: 2.7993060441641795, correct: 49
Epoch: 100/500, loss: 2.6359267131124087, correct: 46
Epoch: 110/500, loss: 0.9842662248390842, correct: 48
Epoch: 120/500, loss: 0.5335849881755123, correct: 47
Epoch: 130/500, loss: 1.4558286932897555, correct: 48
Epoch: 140/500, loss: 0.5438859671694672, correct: 50
Epoch: 150/500, loss: 1.0592029375028893, correct: 50
Epoch: 160/500, loss: 0.7664176359578069, correct: 49
Epoch: 170/500, loss: 1.230936898837485, correct: 50
Epoch: 180/500, loss: 0.7670786880185266, correct: 50
Epoch: 190/500, loss: 1.320549496383746, correct: 50
Epoch: 200/500, loss: 0.7434712984999127, correct: 49
Epoch: 210/500, loss: 1.4571464559187433, correct: 50
Epoch: 220/500, loss: 1.1869068751345135, correct: 50
Epoch: 230/500, loss: 0.22259477044509385, correct: 50
Epoch: 240/500, loss: 0.7891739888927606, correct: 50
Epoch: 250/500, loss: 0.674011078377361, correct: 50
Epoch: 260/500, loss: 0.29328110405415464, correct: 50
Epoch: 270/500, loss: 1.1118001433452296, correct: 50
Epoch: 280/500, loss: 0.5946934166244686, correct: 50
Epoch: 290/500, loss: 0.15727511441466893, correct: 50
Epoch: 300/500, loss: 1.7458261239246111, correct: 49
Epoch: 310/500, loss: 0.34369130765970235, correct: 50
Epoch: 320/500, loss: 0.30775250880490973, correct: 49
Epoch: 330/500, loss: 0.1782075144407547, correct: 50
Epoch: 340/500, loss: 0.6725764896466818, correct: 49
Epoch: 350/500, loss: 0.967189148893154, correct: 49
Epoch: 360/500, loss: 0.167130714922302, correct: 49
Epoch: 370/500, loss: 0.38760261330960005, correct: 49
Epoch: 380/500, loss: 0.5093104192810299, correct: 50
Epoch: 390/500, loss: 1.0854501253722124, correct: 50
Epoch: 400/500, loss: 1.214033110790703, correct: 49
Epoch: 410/500, loss: 0.0756691963412403, correct: 50
Epoch: 420/500, loss: 1.0030330822536344, correct: 50
Epoch: 430/500, loss: 1.0578852607269271, correct: 50
Epoch: 440/500, loss: 0.07159455335796121, correct: 50
Epoch: 450/500, loss: 0.20685582980052805, correct: 50
Epoch: 460/500, loss: 0.1855398794645271, correct: 50
Epoch: 470/500, loss: 0.05049155758080727, correct: 50
Epoch: 480/500, loss: 0.07550733955674366, correct: 50
Epoch: 490/500, loss: 0.6723804455028775, correct: 49
```

### GPU
```bash
!python project/run_fast_tensor.py --BACKEND gpu --HIDDEN 150 --DATASET xor --RATE 0.05
```
Number of Points: 50 <br>
Size of Hidden Layer: 150 <br>
Number of Epochs: 500 <br>
Learning Rate: 0.05 <br>
Epoch 490/500. Time per epoch: ?s. Time left: ?s.
```

```




# Training Log Results / Scripts for Task 3_5 for BIG Model:
## Dataset: XOR
### CPU
```bash
!python project/run_fast_tensor.py --BACKEND cpu --PTS 150 --HIDDEN 200 --DATASET xor --RATE 0.05
```
Number of Points: 150 <br>
Size of Hidden Layer: 200 <br>
Number of Epochs: 500 <br>
Learning Rate: 0.05 <br>
Epoch 490/500. Time per epoch: 0.476s. Time left: 4.76s.
```
Epoch: 0/500, loss: 0, correct: 0
Epoch: 0/500, loss: 12.592414317556536, correct: 69
Epoch: 10/500, loss: 4.009563997889028, correct: 129
Epoch: 20/500, loss: 3.912327260211337, correct: 116
Epoch: 30/500, loss: 0.9560699291448949, correct: 137
Epoch: 40/500, loss: 0.9605332904691323, correct: 140
Epoch: 50/500, loss: 0.43732166977778153, correct: 143
Epoch: 60/500, loss: 0.5882669678487282, correct: 136
Epoch: 70/500, loss: 1.0094721781071467, correct: 145
Epoch: 80/500, loss: 0.28842121310795105, correct: 139
Epoch: 90/500, loss: 0.38741485944762316, correct: 139
Epoch: 100/500, loss: 1.127758940950819, correct: 143
Epoch: 110/500, loss: 0.6096663476712423, correct: 144
Epoch: 120/500, loss: 0.7197875685640909, correct: 144
Epoch: 130/500, loss: 0.5792076303876016, correct: 142
Epoch: 140/500, loss: 0.9950459876964395, correct: 148
Epoch: 150/500, loss: 1.2744890133224474, correct: 146
Epoch: 160/500, loss: 0.33406581011697456, correct: 147
Epoch: 170/500, loss: 0.7845292557421676, correct: 145
Epoch: 180/500, loss: 1.4813693988658958, correct: 148
Epoch: 190/500, loss: 1.523465379942074, correct: 146
Epoch: 200/500, loss: 0.8434399538295596, correct: 149
Epoch: 210/500, loss: 1.055883997246258, correct: 147
Epoch: 220/500, loss: 0.39991139405210685, correct: 149
Epoch: 230/500, loss: 0.3170994403040883, correct: 144
Epoch: 240/500, loss: 2.125455977065032, correct: 145
Epoch: 250/500, loss: 1.7083608828128365, correct: 139
Epoch: 260/500, loss: 2.826361538076645, correct: 135
Epoch: 270/500, loss: -5.573246540333458e-06, correct: 148
Epoch: 280/500, loss: 0.8207831772270222, correct: 148
Epoch: 290/500, loss: 0.9043768828537269, correct: 148
Epoch: 300/500, loss: 0.46751280660203043, correct: 148
Epoch: 310/500, loss: 0.9891832087433334, correct: 146
Epoch: 320/500, loss: 0.17763284997247195, correct: 148
Epoch: 330/500, loss: 0.5124874829499231, correct: 148
Epoch: 340/500, loss: 1.381400862084856, correct: 145
Epoch: 350/500, loss: 0.027577701258826044, correct: 147
Epoch: 360/500, loss: 0.6906758515354867, correct: 148
Epoch: 370/500, loss: 1.9571060004748293, correct: 147
Epoch: 380/500, loss: 0.39487752841708496, correct: 145
Epoch: 390/500, loss: 0.21813644076825184, correct: 150
Epoch: 400/500, loss: 3.754779197322264, correct: 145
Epoch: 410/500, loss: 0.6760665962260459, correct: 150
Epoch: 420/500, loss: 0.07308579942263083, correct: 146
Epoch: 430/500, loss: 0.5993267384070827, correct: 149
Epoch: 440/500, loss: 0.9215244850353175, correct: 147
Epoch: 450/500, loss: 0.6907161035635481, correct: 145
Epoch: 460/500, loss: 0.15187944933103192, correct: 148
Epoch: 470/500, loss: 0.12516060531132198, correct: 148
Epoch: 480/500, loss: 0.01778306351441681, correct: 147
Epoch: 490/500, loss: 0.22800506023847006, correct: 148
```

### GPU
```bash
!python project/run_fast_tensor.py --BACKEND gpu --PTS 150 --HIDDEN 200 --DATASET xor --RATE 0.05
```
Number of Points: 150 <br>
Size of Hidden Layer: 200 <br>
Number of Epochs: 500 <br>
Learning Rate: 0.05 <br>
Total time:  ???? Time per epoch:  ???
```
Epoch: 0/500, loss: 0, correct: 0
Epoch: 0/500, loss: 7.420626333858612, correct: 96
Epoch: 10/500, loss: 0.969074474601098, correct: 142
Epoch: 20/500, loss: 2.7286959359199594, correct: 139
Epoch: 30/500, loss: 1.4063755583214455, correct: 144
Epoch: 40/500, loss: 0.7687777797070903, correct: 149
Epoch: 50/500, loss: 0.2622249116739929, correct: 149
Epoch: 60/500, loss: 2.5032149235589376, correct: 143
Epoch: 70/500, loss: 1.5805275185331182, correct: 149
Epoch: 80/500, loss: 0.37365659214894503, correct: 150
Epoch: 90/500, loss: 0.041543362293844296, correct: 148
Epoch: 100/500, loss: 0.13598476170598006, correct: 147
Epoch: 110/500, loss: 0.8440884787157044, correct: 149
Epoch: 120/500, loss: 0.3862949855416755, correct: 150
Epoch: 130/500, loss: 0.36454347786457253, correct: 150
Epoch: 140/500, loss: 1.2675182707854704, correct: 150
Epoch: 150/500, loss: 0.09914769851886598, correct: 149
Epoch: 160/500, loss: 0.6059082572999411, correct: 149
Epoch: 170/500, loss: 0.30644990702995895, correct: 149
Epoch: 180/500, loss: 0.8015226776531563, correct: 150
Epoch: 190/500, loss: 0.44073599817429726, correct: 150
Epoch: 200/500, loss: 0.590824126048523, correct: 150
Epoch: 210/500, loss: 0.01702333697511014, correct: 150
Epoch: 220/500, loss: 0.04154172911457159, correct: 150
Epoch: 230/500, loss: 0.4688638469087397, correct: 150
Epoch: 240/500, loss: 0.1686051205914939, correct: 150
Epoch: 250/500, loss: 0.43610312846433286, correct: 150
Epoch: 260/500, loss: 0.11974496414255847, correct: 150
Epoch: 270/500, loss: 0.1280347514417979, correct: 150
Epoch: 280/500, loss: 0.005924522299325911, correct: 150
Epoch: 290/500, loss: 0.24947197178711666, correct: 150
Epoch: 300/500, loss: 1.1107494492924717, correct: 150
Epoch: 310/500, loss: 0.2315859788534948, correct: 150
Epoch: 320/500, loss: 0.024009079718941565, correct: 150
Epoch: 330/500, loss: 0.10111979583251238, correct: 150
Epoch: 340/500, loss: 0.04595567218364933, correct: 150
Epoch: 350/500, loss: 0.11022910916243267, correct: 150
Epoch: 360/500, loss: 0.5469416636783089, correct: 150
Epoch: 370/500, loss: 0.10256114765485931, correct: 150
Epoch: 380/500, loss: 0.2749259417529192, correct: 150
Epoch: 390/500, loss: 0.282396349628022, correct: 150
Epoch: 400/500, loss: 0.1294282392870291, correct: 150
Epoch: 410/500, loss: 0.43947208782635294, correct: 150
Epoch: 420/500, loss: 0.004676064318448044, correct: 150
Epoch: 430/500, loss: 0.003148595149319381, correct: 150
Epoch: 440/500, loss: 0.017076366032435713, correct: 150
Epoch: 450/500, loss: 0.05951376662019023, correct: 150
Epoch: 460/500, loss: 0.17974201929832243, correct: 150
Epoch: 470/500, loss: 0.07918425253753092, correct: 150
Epoch: 480/500, loss: 0.16736554374258802, correct: 150
Epoch: 490/500, loss: 0.326900068552855, correct: 150
```
