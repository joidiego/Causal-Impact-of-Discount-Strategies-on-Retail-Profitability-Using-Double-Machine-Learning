# tests/test_aggregation.py
import unittest
import pandas as pd
from src.data.aggregate import aggregate_by_customer, aggregate_by_order

class TestAggregation(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'customer_id': ['C1', 'C1', 'C2'],
            'Order_ID': ['O1', 'O2', 'O3'],
            'Sales': [100, 200, 150],
            'Profit': [20, 30, 25],
            'Discount': [0.1, 0.2, 0.15],
            'order_date': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03'])
        })

    def test_aggregate_by_customer(self):
        result = aggregate_by_customer(self.df)
        self.assertIn('C1', result.index)
        self.assertEqual(result.loc['C1', ('Profit', 'sum')], 50)

if __name__ == '__main__':
    unittest.main()