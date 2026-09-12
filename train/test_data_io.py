import pandas as pd
import pytest
import os
import warnings

# Suppress warnings for the entire module
warnings.filterwarnings("ignore")
# Or for specific warning types
# warnings.filterwarnings("ignore", category=UserWarning)

from .data_io import Disk
from enum import Enum

class SampleEnum(Enum):
    COLUMN_ID = 1
    COLUMN_FEATURE = 2
    COLUMN_TARGET = 3

FOLDER = "test"
FILENAME = "sample"

@pytest.fixture(scope="session")
def sample_csv_path():
    filepath = f"{FOLDER}/{FILENAME}.csv"
    csv_sample = """
    id,name,value
    1,hello,300
    """
    if not os.path.exists(FOLDER):
        os.makedirs(FOLDER) 
    with open(filepath, 'w') as f: 
        f.write(csv_sample)
    yield filepath
    if os.path.exists(filepath):
        os.remove(filepath)

@pytest.mark.filterwarnings("ignore")
class TestDisk:
    """Test suite for disk-based data I/O operations using the Disk class."""

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_initialization(self):
        """Test Disk class initialization with folder path and filename."""
        disk = Disk(None, FOLDER, FILENAME)
        assert disk.path == FOLDER 
        assert disk.name == FILENAME

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_load_dataframe_via_csv(self, sample_csv_path):
        """Test loading CSV file into DataFrame with column name mapping using Enums."""
        disk = Disk(None, FOLDER, FILENAME)
        disk.load_dataframe_via_csv(SampleEnum, {})
        df = disk.load_data()
        
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (1, 3)
        assert list(df.columns) == [SampleEnum.COLUMN_ID.name, SampleEnum.COLUMN_FEATURE.name, SampleEnum.COLUMN_TARGET.name]
        assert df.iloc[0, 0] == 1
        assert df.iloc[0, 1] == 'hello'
        assert df.iloc[0, 2] == 300

    def test_replace_columns(self):
        """Test DataFrame column replacement using Enum mapping to standardize column names."""
        sample_dict = {
            "id": [1, 2, 3],
            "ffaster": ["asd", "asd", "bds"],
            "ttarget": [100, 200, 100],
        }
        disk = Disk(None, FOLDER, FILENAME)
        disk.raw_data = pd.DataFrame(sample_dict)
        disk.raw_data = disk._replace_columns(disk.raw_data, SampleEnum)
        
        assert disk.raw_data.columns[0] == SampleEnum.COLUMN_ID.name
        assert disk.raw_data.columns[1] == SampleEnum.COLUMN_FEATURE.name
        assert disk.raw_data.columns[2] == SampleEnum.COLUMN_TARGET.name

    def test_save_data_via_csv(self):
        """Test saving DataFrame to CSV file and verify the saved file contents."""
        output = "output"
        output_path = os.path.join(FOLDER, f"{output}.csv")
        sample_data = pd.DataFrame({
            "id": [1, 2], 
            "name": ["foo", "bar"], 
            "value": [100, 200]
        })
        
        disk = Disk(None, FOLDER, output)
        disk.save_via_csv()
        disk.save_data(sample_data)
        
        assert os.path.exists(output_path)
        loaded_data = pd.read_csv(output_path)
        assert loaded_data.shape == (2, 3)
        assert list(loaded_data.columns) == ["id", "name", "value"]


class TestRowCounting:
    """F-15: counting data rows without decoding the whole file."""

    def _write(self, tmp_path, content, name="rows.csv"):
        path = tmp_path / name
        path.write_text(content)
        return str(path)

    def test_counts_excluding_header(self, tmp_path):
        path = self._write(tmp_path, "h\n1\n2\n3\n")

        assert Disk._count_data_rows(path) == 3

    def test_counts_a_file_without_a_trailing_newline(self, tmp_path):
        path = self._write(tmp_path, "h\n1\n2\n3")

        assert Disk._count_data_rows(path) == 3

    def test_header_only_file_has_no_data_rows(self, tmp_path):
        path = self._write(tmp_path, "h\n")

        assert Disk._count_data_rows(path) == 0

    def test_agrees_with_pandas(self, tmp_path):
        body = "\n".join(f"{i},{i * 2}" for i in range(500))
        path = self._write(tmp_path, "id,val\n" + body + "\n")

        assert Disk._count_data_rows(path) == len(pd.read_csv(path))


class TestRandomRowSelection:
    """F-15: skiprows as a predicate cost one Python call per row."""

    def test_keeps_exactly_n_rows(self):
        disk = Disk(None, FOLDER, FILENAME)

        skip = disk.generate_skiprows(100, 10, random_state=42)
        kept = set(range(1, 101)) - set(skip.tolist())

        assert len(kept) == 10

    def test_never_skips_the_header(self):
        disk = Disk(None, FOLDER, FILENAME)

        skip = disk.generate_skiprows(100, 10, random_state=42)

        assert 0 not in skip

    def test_is_deterministic_for_a_seed(self):
        disk = Disk(None, FOLDER, FILENAME)

        assert (disk.generate_skiprows(100, 10, 42) == disk.generate_skiprows(100, 10, 42)).all()

    def test_varies_across_seeds(self):
        disk = Disk(None, FOLDER, FILENAME)

        assert not (disk.generate_skiprows(100, 10, 42) == disk.generate_skiprows(100, 10, 7)).all()

    def test_all_rows_selected_means_nothing_is_skipped(self):
        disk = Disk(None, FOLDER, FILENAME)

        assert list(disk.generate_skiprows(10, 10, random_state=1)) == []

    def test_requesting_more_rows_than_exist_raises(self):
        disk = Disk(None, FOLDER, FILENAME)

        with pytest.raises(ValueError, match="only has"):
            disk.generate_skiprows(10, 11)

    def test_end_to_end_sample_is_spread_across_the_file(self, tmp_path):
        # The point of the loader is a random sample, not the first n rows.
        body = "\n".join(f"{i},v{i},{i * 2}" for i in range(300))
        path = tmp_path / "big.csv"
        path.write_text("id,val,other\n" + body + "\n")

        disk = Disk(None, str(tmp_path), "big")
        disk.load_random_rows_via_csv(SampleEnum, {}, n_rows=30, random_state=1)
        frame = disk.load_data()

        assert frame.shape == (30, 3)
        assert max(frame.iloc[:, 0]) > 150

    def test_end_to_end_matches_a_direct_pandas_read(self, tmp_path):
        body = "\n".join(f"{i},v{i},{i * 2}" for i in range(300))
        path = tmp_path / "big2.csv"
        path.write_text("id,val,other\n" + body + "\n")

        disk = Disk(None, str(tmp_path), "big2")
        disk.load_random_rows_via_csv(SampleEnum, {}, n_rows=30, random_state=7)
        via_disk = disk.load_data()
        via_pandas = pd.read_csv(path, skiprows=disk.generate_skiprows(300, 30, 7))

        assert list(via_disk.iloc[:, 0]) == list(via_pandas.iloc[:, 0])

    def test_does_not_overwrite_the_training_load_metrics(self, tmp_path):
        # post_test runs on a run that already recorded its training load under
        # size.load.*; reporting the post-test sample there would rewrite them.
        class Facade:
            def __init__(self):
                self.calls = {}

            def set_data_loading_time(self, ms):
                self.calls["time"] = ms

            def set_row_size(self, n):
                self.calls["rows"] = n

            def set_column_size(self, n):
                self.calls["cols"] = n

            def set_dataset_name(self, name):
                self.calls["name"] = name

        body = "\n".join(f"{i},v{i},{i * 2}" for i in range(50))
        (tmp_path / "meta.csv").write_text("id,val,other\n" + body + "\n")
        facade = Facade()

        disk = Disk(facade, str(tmp_path), "meta")
        disk.load_random_rows_via_csv(SampleEnum, {}, n_rows=5, random_state=1)
        frame = disk.load_data()

        assert frame.shape == (5, 3)
        assert facade.calls == {}

