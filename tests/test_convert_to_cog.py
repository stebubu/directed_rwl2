import os
import ast
import importlib.util
import pathlib
import numpy as np
import rasterio
from rasterio.transform import from_origin

ROOT = pathlib.Path(__file__).resolve().parents[1]
# Load only the convert_to_cog function without importing other heavy dependencies
with open(ROOT / "home1.py") as f:
    module_ast = ast.parse(f.read())
func_node = next(node for node in module_ast.body if isinstance(node, ast.FunctionDef) and node.name == "convert_to_cog")
module = ast.Module(body=[func_node], type_ignores=[])
namespace = {}
spec = importlib.util.find_spec("rasterio")
if spec is None:
    import rasterio  # ensure dependency is present
else:
    import rasterio
from rasterio.enums import Resampling
namespace["rasterio"] = rasterio
namespace["Resampling"] = Resampling
exec(compile(module, "convert_to_cog", "exec"), namespace)
convert_to_cog = namespace["convert_to_cog"]


def test_convert_to_cog(tmp_path):
    # create temporary GeoTIFF
    data = np.random.rand(10, 10).astype('float32')
    geotiff_path = tmp_path / "test.tif"
    with rasterio.open(
        geotiff_path,
        'w',
        driver='GTiff',
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        transform=from_origin(0, 10, 1, 1),
    ) as dst:
        dst.write(data, 1)

    # run conversion
    cog_path = convert_to_cog(str(geotiff_path))

    assert os.path.exists(cog_path)
