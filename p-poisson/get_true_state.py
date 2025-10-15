import numpy as np
import os
import meshio
from pathlib import Path

xdmf_path = Path("utrue.xdmf")
txt = xdmf_path.read_text()

# 1) If the file starts with <Grid> (no wrapper), wrap it so meshio can parse it
if txt.lstrip().startswith("<Grid"):
    wrapped = f"""<?xml version="1.0" ?>
<Xdmf Version="3.0">
  <Domain>
{txt}
  </Domain>
</Xdmf>
"""
    xdmf_fixed = xdmf_path.with_name("utrue_fixed.xdmf")
    xdmf_fixed.write_text(wrapped)
    xdf = str(xdmf_fixed)
else:
    xdf = str(xdmf_path)

# 2) Try time-series reader first (many fenics writers create a collection)
try:
    with meshio.xdmf.TimeSeriesReader(xdf) as rdr:
        points, cells = rdr.read_points_cells()
        k = rdr.num_steps - 1  # last step (or set k = 0 if you know there is only one)
        t, point_data, cell_data = rdr.read_data(k)
        coords = np.asarray(points)

        if point_data:                      # nodal field (most CG spaces)
            name = next(iter(point_data))
            vals = np.asarray(point_data[name]).squeeze()
        else:                               # cell-wise field (DG)
            ctype = cells[0].type
            # pick the first available data array
            name = next(iter(cell_data[ctype]))
            vals = np.asarray(cell_data[ctype][name]).squeeze()

except Exception:
    # Fallback for single-snapshot XDMF
    m = meshio.read(xdf)
    coords = np.asarray(m.points)
    if m.point_data:
        name, vals = next(iter(m.point_data.items()))
        vals = np.asarray(vals).squeeze()
    else:
        ctype = m.cells[0].type
        name, vals = next(iter(m.cell_data_dict[ctype].items()))
        vals = np.asarray(vals).squeeze()


xs = np.unique(coords[:, 0]); ys = np.unique(coords[:, 1]); zs = np.unique(coords[:, 2])

# sort nodes in x-y-z order
idx = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0]))
vals_sorted = vals[idx]

field_C = vals_sorted.reshape(len(xs), len(ys), len(zs), order="C") # This is the correct order
field_F = vals_sorted.reshape(len(xs), len(ys), len(zs), order="F") # This is the wrong order

output_dir = "training_dataset"

np.save(
        os.path.join(output_dir, "true_state.npy"),
        field_C,
    )