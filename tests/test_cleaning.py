# tests/test_cleaning.py
import unittest
import pandas as pd
from src.data.clean import load_and_clean_data

class TestCleaning(unittest.TestCase):
    def test_load_clean(self):
        df = load_and_clean_data("data/raw/Sample - Superstore.csv")
        self.assertFalse(df.empty)
        self.assertIn('profit_margin', df.columns)
        self.assertGreater(len(df), 0)

if __name__ == '__main__':
    unittest.main()