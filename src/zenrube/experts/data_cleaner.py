import json
import csv
import io
import re
from typing import Any, Union

EXPERT_METADATA = {
    "name": "data_cleaner",
    "version": "1.0",
    "description": "Cleans messy text, CSV, or JSON data.",
    "author": "vladinc@gmail.com"
}


def detect_format(input_data: Any) -> str:
    """
    Detects the format of the input data based on simple heuristics.

    Args:
        input_data: The input data to analyze (string, list, or dict).

    Returns:
        str: "csv", "json", or "text".
    """
    if isinstance(input_data, dict):
        return "json"
    if isinstance(input_data, list):
        return "csv"  # Assuming list represents CSV rows
    if isinstance(input_data, str):
        # Try to parse as JSON
        try:
            json.loads(input_data)
            return "json"
        except (json.JSONDecodeError, TypeError):
            pass
        # Check for CSV-like structure (commas and newlines)
        if ',' in input_data and '\n' in input_data:
            return "csv"
        else:
            return "text"
    return "text"


class DataCleanerExpert:
    """
    Expert class for cleaning messy data inputs including text, CSV, and JSON.
    """

    def run(self, input_data: Union[str, list, dict]) -> Union[str, list, dict]:
        """
        Cleans the input data by detecting its format and applying appropriate cleaning operations.

        Args:
            input_data: The data to clean (string, list, or dict).

        Returns:
            The cleaned data in the same structure as the input.
        """
        format_type = detect_format(input_data)

        if format_type == "json":
            return self._clean_json(input_data)
        elif format_type == "csv":
            return self._clean_csv(input_data)
        else:  # text
            return self._clean_text(input_data)

    def _clean_json(self, input_data: Union[str, dict]) -> Union[str, dict]:
        """Cleans JSON data (dict or JSON string)."""
        if isinstance(input_data, str):
            try:
                data = json.loads(input_data)
            except (json.JSONDecodeError, TypeError):
                return input_data  # Return as-is if not valid JSON
        else:
            data = input_data

        cleaned_data = self._clean_dict(data)

        if isinstance(input_data, str):
            return json.dumps(cleaned_data, indent=2)
        else:
            return cleaned_data

    def _clean_csv(self, input_data: Union[str, list]) -> Union[str, list]:
        """Cleans CSV data (string or list of rows)."""
        if isinstance(input_data, str):
            try:
                reader = csv.reader(io.StringIO(input_data))
                rows = list(reader)
            except Exception:
                return input_data  # Return as-is if parsing fails
        else:
            rows = input_data

        cleaned_rows = []
        seen = set()

        for row in rows:
            if isinstance(row, list):
                # Clean each cell: strip, remove non-printable, remove empty
                cleaned_row = []
                for cell in row:
                    if isinstance(cell, str):
                        cell = re.sub(r'[^\x20-\x7E\n\r\t]', '', cell)  # Remove non-printable
                        cell = cell.strip()
                        if cell:  # Only add non-empty cells
                            cell = cell[0].upper() + cell[1:] if cell else ''  # Normalize capitalization
                            cleaned_row.append(cell)
                    else:
                        cleaned_row.append(cell)
                if cleaned_row:  # Only add non-empty rows
                    row_tuple = tuple(cleaned_row)
                    if row_tuple not in seen:
                        seen.add(row_tuple)
                        cleaned_rows.append(cleaned_row)
            else:
                # Treat as single item
                if isinstance(row, str):
                    row = re.sub(r'[^\x20-\x7E\n\r\t]', '', row)
                    row = row.strip()
                    if row:
                        row = row[0].upper() + row[1:] if row else ''
                        if row not in seen:
                            seen.add(row)
                            cleaned_rows.append(row)

        if isinstance(input_data, str):
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerows(cleaned_rows)
            return output.getvalue()
        else:
            return cleaned_rows

    def _clean_text(self, input_data: Union[str, list]) -> Union[str, list]:
        """Cleans plain text data (string or list of strings)."""
        if isinstance(input_data, str):
            # Remove non-printable characters
            cleaned = re.sub(r'[^\x20-\x7E\n\r\t]', '', input_data)
            cleaned = cleaned.strip()
            # Process each line
            lines = cleaned.split('\n')
            cleaned_lines = []
            for line in lines:
                line = line.strip()
                if line:
                    # Normalize capitalization: first letter uppercase
                    line = line[0].upper() + line[1:] if line else ''
                    cleaned_lines.append(line)
            return '\n'.join(cleaned_lines)
        elif isinstance(input_data, list):
            cleaned = []
            seen = set()
            for item in input_data:
                if isinstance(item, str):
                    item = re.sub(r'[^\x20-\x7E\n\r\t]', '', item)
                    item = item.strip()
                    if item:
                        item = item[0].upper() + item[1:] if item else ''
                        if item not in seen:
                            seen.add(item)
                            cleaned.append(item)
                elif item is not None and item != '':
                    cleaned.append(item)
            return cleaned
        else:
            return input_data

    def _clean_dict(self, data: Any) -> Any:
        """Recursively cleans a dictionary or list structure."""
        if isinstance(data, dict):
            cleaned = {}
            for k, v in data.items():
                if v is not None and v != '':
                    cleaned_k = k.strip() if isinstance(k, str) else k
                    cleaned_v = self._clean_dict(v)
                    if cleaned_v is not None and cleaned_v != '':
                        cleaned[cleaned_k] = cleaned_v
            return cleaned
        elif isinstance(data, list):
            cleaned = []
            seen = set()
            for item in data:
                cleaned_item = self._clean_dict(item)
                if cleaned_item is not None and cleaned_item != '':
                    if isinstance(cleaned_item, (str, int, float, bool)):
                        if cleaned_item not in seen:
                            seen.add(cleaned_item)
                            cleaned.append(cleaned_item)
                    else:
                        cleaned.append(cleaned_item)
            return cleaned
        elif isinstance(data, str):
            cleaned = re.sub(r'[^\x20-\x7E\n\r\t]', '', data)
            cleaned = cleaned.strip()
            if cleaned:
                cleaned = cleaned[0].upper() + cleaned[1:] if cleaned else ''
            return cleaned if cleaned else None
        else:
            return data
