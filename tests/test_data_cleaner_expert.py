import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from zenrube.experts.data_cleaner import DataCleanerExpert, detect_format


class TestDataCleanerExpert:
    def test_detect_format_identifies_json_csv_text(self):
        # JSON string
        assert detect_format('{"name": "Vlad"}') == "json"
        # CSV string
        assert detect_format('a,b,c\n1,2,3') == "csv"
        # Text string
        assert detect_format('just some words') == "text"

    def test_clean_text_removes_noise_and_trims(self):
        expert = DataCleanerExpert()
        input_data = "  hello!!!  "
        result = expert.run(input_data)
        assert result == "Hello!!!"

    def test_clean_csv_removes_empty_rows_and_trims_cells(self):
        expert = DataCleanerExpert()
        input_data = " name , age , city \n Alice , 30 , New York \n , , \n Bob , , Boston "
        result = expert.run(input_data)
        # CSV writer uses \r\n line endings and removes empty cells
        expected = "Name,Age,City\r\nAlice,30,New York\r\nBob,Boston\r\n"
        assert result == expected

    def test_clean_json_removes_nulls_and_trims_strings(self):
        expert = DataCleanerExpert()
        input_data = '{"name": " Vlad ", "age": null, "city": "Boston"}'
        result = expert.run(input_data)
        expected = '{\n  "name": "Vlad",\n  "city": "Boston"\n}'
        assert result == expected

    def test_returns_same_type_as_input(self):
        expert = DataCleanerExpert()
        # String input
        result_str = expert.run("some text")
        assert isinstance(result_str, str)
        # Dict input
        result_dict = expert.run({"key": "value"})
        assert isinstance(result_dict, dict)
        # List input
        result_list = expert.run(["item1", "item2"])
        assert isinstance(result_list, list)

    def test_duplicate_removal_in_lists(self):
        expert = DataCleanerExpert()
        input_data = [" apple ", "Apple", "banana", "banana "]
        result = expert.run(input_data)
        expected = ["Apple", "Banana"]
        assert result == expected

    def test_handles_invalid_input_gracefully(self):
        expert = DataCleanerExpert()
        # Integer input
        result_int = expert.run(12345)
        assert result_int == 12345
        # None input
        result_none = expert.run(None)
        assert result_none is None
