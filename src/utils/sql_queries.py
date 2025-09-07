# src/utils/sql_queries.py

CUSTOMER_AGGREGATE_QUERY = """
SELECT 
    "Customer ID",
    "Segment",
    "Region",
    COUNT("Order ID") AS order_count,
    AVG("Discount") AS avg_discount,
    SUM("Profit") AS total_profit,
    MAX("Order Date") AS last_order_date
FROM superstore
GROUP BY "Customer ID", "Segment", "Region";
"""

PRODUCT_AGGREGATE_QUERY = """
SELECT 
    "Product ID",
    "Category",
    "Sub-Category",
    AVG("Discount") AS avg_discount,
    AVG("Profit") AS avg_profit,
    SUM("Quantity") AS total_sold
FROM superstore
GROUP BY "Product ID", "Category", "Sub-Category";
"""