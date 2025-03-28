# %% [markdown]
#
# Create a parallel grid with a non-overlapping domain decomposition.
#
# First run the script as usual. Then run python script with the command
# ```
# mpirun -np 4 python parallelization_exercise.py
# ```

import dune.alugrid
from dune.common import comm
from dune import fem
import dune
import dune.common
import dune.grid
print(comm.rank)


# %%
##### TASK
domain = dune.grid.cartesianDomain([0, 0], [1, 1], [10, 10])
view = dune.grid.yaspGrid(domain)
view = dune.alugrid.aluConformGrid(domain, overlap=0)
print(comm.rank)
# %% [markdown]
#
# Compute the overall number of interior cells and the overall number of cells
# and make sure to only output it
# on one core.
num_interior_cells = 0
num_cells = 0
for element in view.elements:
    num_cells += 1
    if element.partitionType == dune.grid.PartitionType.Interior:
        num_interior_cells += 1
# print(f"process {comm.rank} has {num_interior_cells} interior cells and {num_cells} cells in total")
min_cells = comm.min(num_cells)
max_cells = comm.max(num_cells)
num_inter = comm.sum(num_interior_cells)
num_cells = comm.sum(num_cells)
if comm.rank == 0:
    print(f"the grid has {num_inter} interior cells and {num_cells} cells in total")
    print(f"the minimal number of cells per process is {min_cells} and the maximal number of cells per process is {max_cells}")
# %%
##### TASK

# %% [markdown]
# Compute the minimal and maximal number of elements held by a process.


# %%
##### TASK

# %% [markdown]
# Write a grid function that outputs the rank that each element belongs to.
# Output this grid function using `writeVTK`.

@fem.function.gridFunction(view, codim=0, order=0)
def rank(element, x):
    return comm.rank

fv = fem.space.finiteVolume(view)
save = fv.interpolate(rank)
# save.plot()
write_solution = view.writeVTK(
            "rank",
            pointdata=[save],
            celldata=[save],
        )


# %%
##### TASK

# %% [markdown]
# Repeat the above with `aluGridConform`.


# %%
##### TASK

