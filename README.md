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
Epoch 490/500. Time per epoch: 0.094s. Time left: 0.94s.
```
Epoch: 0/500, loss: 0, correct: 0
Epoch: 0/500, loss: 6.1309437914545555, correct: 29
Epoch: 10/500, loss: 2.547439863429067, correct: 49
Epoch: 20/500, loss: 0.8760080775603306, correct: 49
Epoch: 30/500, loss: 0.9234162129740129, correct: 50
Epoch: 40/500, loss: 1.1754571313761333, correct: 49
Epoch: 50/500, loss: 0.6987835536960919, correct: 50
Epoch: 60/500, loss: 1.3164156749365064, correct: 49
Epoch: 70/500, loss: 1.0580619524218136, correct: 49
Epoch: 80/500, loss: 1.7460524146452037, correct: 50
Epoch: 90/500, loss: 1.0297581737022505, correct: 49
Epoch: 100/500, loss: 0.2798259780752261, correct: 50
Epoch: 110/500, loss: 1.4680060892550602, correct: 50
Epoch: 120/500, loss: 0.5794364657935419, correct: 49
Epoch: 130/500, loss: 0.6204435174368399, correct: 49
Epoch: 140/500, loss: 0.007101947022703552, correct: 49
Epoch: 150/500, loss: 1.3677295476890787, correct: 50
Epoch: 160/500, loss: 0.12600435055248835, correct: 49
Epoch: 170/500, loss: 0.6227606021129468, correct: 50
Epoch: 180/500, loss: 0.15753044430395452, correct: 50
Epoch: 190/500, loss: 0.6432236635223691, correct: 49
Epoch: 200/500, loss: 1.285544722253647, correct: 49
Epoch: 210/500, loss: 0.9695726637098321, correct: 50
Epoch: 220/500, loss: 0.12675396172914058, correct: 49
Epoch: 230/500, loss: 0.399604916767162, correct: 50
Epoch: 240/500, loss: 0.9865768637173497, correct: 50
Epoch: 250/500, loss: 0.44866467934782445, correct: 50
Epoch: 260/500, loss: 0.029580189996901003, correct: 50
Epoch: 270/500, loss: 0.1257279428619415, correct: 49
Epoch: 280/500, loss: 0.3465851091489496, correct: 50
Epoch: 290/500, loss: 0.0072852229990475435, correct: 50
Epoch: 300/500, loss: 0.06870044415489598, correct: 50
Epoch: 310/500, loss: 0.9196844297284107, correct: 50
Epoch: 320/500, loss: 0.001606637957091112, correct: 50
Epoch: 330/500, loss: 0.5146552635021808, correct: 49
Epoch: 340/500, loss: 0.184782067039076, correct: 50
Epoch: 350/500, loss: 0.8927091446644712, correct: 50
Epoch: 360/500, loss: 0.6209224068745461, correct: 50
Epoch: 370/500, loss: 0.8264062061107913, correct: 50
Epoch: 380/500, loss: 0.7514697282847416, correct: 49
Epoch: 390/500, loss: 0.04838876136205005, correct: 50
Epoch: 400/500, loss: 0.7208410563458337, correct: 49
Epoch: 410/500, loss: 0.05288232011799464, correct: 50
Epoch: 420/500, loss: 0.14200656275343843, correct: 50
Epoch: 430/500, loss: 0.3921467100524038, correct: 50
Epoch: 440/500, loss: 0.06412712900993893, correct: 50
Epoch: 450/500, loss: 0.0018730998951490337, correct: 50
Epoch: 460/500, loss: 0.6854926996441637, correct: 50
Epoch: 470/500, loss: 0.6558724188954957, correct: 50
Epoch: 480/500, loss: 0.0007956783496497057, correct: 50
Epoch: 490/500, loss: 0.0680371012686525, correct: 50
```

### GPU
```bash
!python project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET simple --RATE 0.05
```
Number of Points: 50 <br>
Size of Hidden Layer: 100 <br>
Number of Epochs: 500 <br>
Learning Rate: 0.05 <br>
xxxx
```

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
Epoch 490/500. Time per epoch: 0.096s. Time left: 0.96s.
```
Epoch: 0/500, loss: 0, correct: 0
Epoch: 0/500, loss: 4.985911469118869, correct: 29
Epoch: 10/500, loss: 4.172681462804957, correct: 42
Epoch: 20/500, loss: 4.250809151716022, correct: 42
Epoch: 30/500, loss: 4.071187540807513, correct: 42
Epoch: 40/500, loss: 3.9473639091103183, correct: 45
Epoch: 50/500, loss: 4.143323434472034, correct: 45
Epoch: 60/500, loss: 2.7308443813217314, correct: 46
Epoch: 70/500, loss: 2.098019751192104, correct: 46
Epoch: 80/500, loss: 2.777006183370127, correct: 47
Epoch: 90/500, loss: 0.6687411101818445, correct: 47
Epoch: 100/500, loss: 2.556635982639034, correct: 47
Epoch: 110/500, loss: 1.9872734513048105, correct: 48
Epoch: 120/500, loss: 1.0115971105779966, correct: 49
Epoch: 130/500, loss: 0.5793761117540237, correct: 49
Epoch: 140/500, loss: 2.1924465659248566, correct: 48
Epoch: 150/500, loss: 1.666633549209258, correct: 48
Epoch: 160/500, loss: 1.3033564466536143, correct: 48
Epoch: 170/500, loss: 0.7797539710563236, correct: 49
Epoch: 180/500, loss: 1.779598498559662, correct: 49
Epoch: 190/500, loss: 0.9691063913291847, correct: 49
Epoch: 200/500, loss: 2.6138866563059318, correct: 49
Epoch: 210/500, loss: 0.858222006686173, correct: 48
Epoch: 220/500, loss: 3.719855892509756, correct: 46
Epoch: 230/500, loss: 0.9472977084536949, correct: 49
Epoch: 240/500, loss: 1.4055700138652962, correct: 50
Epoch: 250/500, loss: 0.2821526578914398, correct: 49
Epoch: 260/500, loss: 0.900139222112419, correct: 49
Epoch: 270/500, loss: 1.954081097154671, correct: 49
Epoch: 280/500, loss: 1.38240254707894, correct: 48
Epoch: 290/500, loss: 0.9323257603568934, correct: 49
Epoch: 300/500, loss: 0.07552932073450433, correct: 49
Epoch: 310/500, loss: 0.11269314967098812, correct: 49
Epoch: 320/500, loss: 1.3930899635101528, correct: 49
Epoch: 330/500, loss: 1.5627955595293508, correct: 49
Epoch: 340/500, loss: 0.24570332603849757, correct: 48
Epoch: 350/500, loss: 1.7077349765702314, correct: 48
Epoch: 360/500, loss: 3.0326916421897034, correct: 48
Epoch: 370/500, loss: 0.6957031346012548, correct: 49
Epoch: 380/500, loss: 1.7836290346608903, correct: 49
Epoch: 390/500, loss: 0.29958305293796605, correct: 49
Epoch: 400/500, loss: 1.1944536768977403, correct: 49
Epoch: 410/500, loss: 1.6736758242394552, correct: 49
Epoch: 420/500, loss: 0.14657182269151295, correct: 49
Epoch: 430/500, loss: 0.1276398836051795, correct: 48
Epoch: 440/500, loss: 0.729279903305404, correct: 49
Epoch: 450/500, loss: 0.06932927830704114, correct: 49
Epoch: 460/500, loss: 1.4248584064860512, correct: 50
Epoch: 470/500, loss: 1.2805106321610744, correct: 48
Epoch: 480/500, loss: 0.9769447493775355, correct: 49
Epoch: 490/500, loss: 0.019770209300418125, correct: 50
```

### GPU
```bash
!python project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET split --RATE 0.05
```
Number of Points: 50 <br>
Size of Hidden Layer: 100 <br>
Number of Epochs: 500 <br>
Learning Rate: 0.05 <br>
xxxxx
```

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
Epoch 490/500. Time per epoch: 0.111s. Time left: 1.11s.
```
Epoch: 0/500, loss: 0, correct: 0
Epoch: 0/500, loss: 5.4448613088074875, correct: 36
Epoch: 10/500, loss: 2.6294694399107974, correct: 36
Epoch: 20/500, loss: 3.0513634362971924, correct: 46
Epoch: 30/500, loss: 3.0206486066790728, correct: 46
Epoch: 40/500, loss: 1.9746770574184458, correct: 47
Epoch: 50/500, loss: 2.6716380705262313, correct: 45
Epoch: 60/500, loss: 2.827190304899672, correct: 49
Epoch: 70/500, loss: 2.963956322408267, correct: 46
Epoch: 80/500, loss: 5.0828317314490326, correct: 46
Epoch: 90/500, loss: 1.0482965219177707, correct: 49
Epoch: 100/500, loss: 3.9253724678985322, correct: 46
Epoch: 110/500, loss: 1.0958182551816096, correct: 48
Epoch: 120/500, loss: 1.224988503604569, correct: 48
Epoch: 130/500, loss: 0.8047966912553797, correct: 49
Epoch: 140/500, loss: 2.6845590447648835, correct: 46
Epoch: 150/500, loss: 1.871403156974904, correct: 49
Epoch: 160/500, loss: 1.679994318260821, correct: 49
Epoch: 170/500, loss: 2.475279765550174, correct: 49
Epoch: 180/500, loss: 2.977243726352885, correct: 46
Epoch: 190/500, loss: 1.5892126668711737, correct: 49
Epoch: 200/500, loss: 1.4577418796044046, correct: 48
Epoch: 210/500, loss: 1.1941033981633704, correct: 49
Epoch: 220/500, loss: 2.117125816522166, correct: 47
Epoch: 230/500, loss: 1.3413184827228826, correct: 48
Epoch: 240/500, loss: 1.7823933306268003, correct: 48
Epoch: 250/500, loss: 0.757526272944018, correct: 49
Epoch: 260/500, loss: 0.4268175883520157, correct: 48
Epoch: 270/500, loss: 0.48237978820910915, correct: 48
Epoch: 280/500, loss: 0.733270020186804, correct: 48
Epoch: 290/500, loss: 1.1551890888770968, correct: 49
Epoch: 300/500, loss: 0.3150502475158421, correct: 47
Epoch: 310/500, loss: 1.4000656805042113, correct: 47
Epoch: 320/500, loss: 2.8470770497214315, correct: 49
Epoch: 330/500, loss: 0.4647527636241856, correct: 47
Epoch: 340/500, loss: 0.6097627639688151, correct: 49
Epoch: 350/500, loss: 0.6363944048609286, correct: 49
Epoch: 360/500, loss: 0.24500653570164346, correct: 49
Epoch: 370/500, loss: 1.6218599514743146, correct: 49
Epoch: 380/500, loss: 0.6546604324555297, correct: 49
Epoch: 390/500, loss: 1.3862721240833469, correct: 49
Epoch: 400/500, loss: 0.5720805395987072, correct: 49
Epoch: 410/500, loss: 2.177069471966575, correct: 49
Epoch: 420/500, loss: 0.6979814087239045, correct: 49
Epoch: 430/500, loss: 0.2954250430593788, correct: 49
Epoch: 440/500, loss: 1.9652982392431837, correct: 49
Epoch: 450/500, loss: 0.4763872378598757, correct: 49
Epoch: 460/500, loss: 0.4161133290355786, correct: 48
Epoch: 470/500, loss: 1.3737850972451902, correct: 49
Epoch: 480/500, loss: 0.32869372509462147, correct: 49
Epoch: 490/500, loss: 0.8598539699480887, correct: 50
```

### GPU
```bash
!python project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET xor --RATE 0.05
```
Number of Points: 50 <br>
Size of Hidden Layer: 100 <br>
Number of Epochs: 500 <br>
Learning Rate: 0.05 <br>
xxxxxxx
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
xxxxx
```

```
