
import os
from pathlib import Path

import h5py
import pytest

from quasar_utils.io.hdf5.linelist import hdf5_to_linelist, linelist_to_hdf5
from quasar_utils.linelist import LineList
from quasar_utils.setup import Info

DATA_CSV = Path(__file__).parent / "linelist.csv"


def test_linelist_hdf5_roundtrip(tmp_path):
    info = Info()

    # load from provided csv
    ll = LineList.read_csv(path=DATA_CSV, info=info)

    h5path = tmp_path / "test.h5"
    with h5py.File(h5path, "w") as f:
        grp = linelist_to_hdf5(ll, f, info=info)
        assert "linelist" in f
        assert "columns" in grp

    # read back
    with h5py.File(h5path, "r") as f:
        ll2 = hdf5_to_linelist(f, info=info)

    # ensure essential columns present and length matches
    assert set(ll.columns) == set(ll2.columns)
    assert len(ll) == len(ll2)


def test_linelist_hdf5_force_and_groupname(tmp_path):
    info = Info()
    ll = LineList.read_csv(path=DATA_CSV, info=info)

    h5path = tmp_path / "force.h5"
    with h5py.File(h5path, "w") as f:
        # write to custom group
        linelist_to_hdf5(ll, f, info=info, group_name="mylist")
        assert "mylist" in f

        # writing again without force should raise
        with pytest.raises(ValueError):
            linelist_to_hdf5(ll, f, info=info, group_name="mylist", force=False)

        # writing with force should overwrite
        linelist_to_hdf5(ll, f, info=info, group_name="mylist", force=True)

    # verify reading from custom group_name
    with h5py.File(h5path, "r") as f:
        ll2 = hdf5_to_linelist(f, info=info, group_name="mylist")
        assert len(ll2) == len(ll)


def test_print_disk_usage(tmp_path, subtests: pytest.Subtests):
    """Write an HDF5 file and print sizes of the source CSV and generated HDF5."""
    info = Info()
    ll = LineList.read_csv(path=DATA_CSV, info=info)

    h5path = tmp_path / "size.h5"
    with h5py.File(h5path, "w") as f:
        linelist_to_hdf5(ll, f, info=info)

    csv_size = os.path.getsize(DATA_CSV)
    h5_size = os.path.getsize(h5path)

    with subtests.test(f"CSV: {csv_size}, HDF5: {h5_size} -> ratio: {h5_size/csv_size:.2f}"):
        assert csv_size > 0
        assert h5_size > 0