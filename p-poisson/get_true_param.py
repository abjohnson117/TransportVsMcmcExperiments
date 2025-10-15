import numpy as np
import os
import meshio
from pathlib import Path

xdmf_path = Path("mtrue_grid.xdmf")

# Direct single-snapshot read
m = meshio.read(str(xdmf_path))

coords = np.asarray(m.points)

print("Converting data to numpy array...")
if m.point_data:
    name, vals = next(iter(m.point_data.items()))
    vals = np.asarray(vals).squeeze()
else:
    ctype = m.cells[0].type
    name, vals = next(iter(m.cell_data_dict[ctype].items()))
    vals = np.asarray(vals).squeeze()

# Now handle the 2D coordinates
xs = np.unique(coords[:, 0])
ys = np.unique(coords[:, 1])

print("Sorting in correct shape...")
# sort nodes in x-y order
idx = np.lexsort((coords[:, 1], coords[:, 0]))
vals_sorted = vals[idx]

print("Reshaping into right order...")
# reshape into 2D array
field = vals_sorted.reshape(len(xs), len(ys), order="C")

print("Saving file...")
# Save to file
output_dir = "training_dataset"
os.makedirs(output_dir, exist_ok=True)
np.save(os.path.join(output_dir, "true_param_grid.npy"), field)
