-- Q1: WHERE then HAVING (Completed orders > 500, customer total > 8000)
SELECT customer_id, SUM(order_amount) AS total_spend
FROM orders
WHERE status = 'completed' AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000;

-- Q2: Three-table join revenue by city and category
SELECT c.city, p.category, ROUND(SUM(o.order_amount), 2) AS total_revenue
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN products p ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category;

-- Q3: Products priced above average catalogue price
SELECT product_id, product_name, price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products);

-- Q4: LEFT JOIN for customers with no completed orders
SELECT c.customer_id
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id AND o.status = 'completed'
WHERE o.order_id IS NULL;

-- Q5: CTEs for customer completed spend vs. average
WITH customer_spend AS (
    SELECT customer_id, SUM(order_amount) as total_spend
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
),
avg_spend AS (
    SELECT AVG(total_spend) as mean_spend FROM customer_spend
)
SELECT cs.customer_id, cs.total_spend
FROM customer_spend cs, avg_spend av
WHERE cs.total_spend > av.mean_spend;

-- Q6: Ranking window functions partitioned by city ordered by total spend descending
WITH customer_city_spend AS (
    SELECT c.customer_id, c.city, SUM(o.order_amount) AS total_spend
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    WHERE o.status = 'completed'
    GROUP BY c.customer_id, c.city
)
SELECT customer_id, city, total_spend,
       RANK() OVER(PARTITION BY city ORDER BY total_spend DESC) as rnk,
       DENSE_RANK() OVER(PARTITION BY city ORDER BY total_spend DESC) as dense_rnk,
       ROW_NUMBER() OVER(PARTITION BY city ORDER BY total_spend DESC) as row_num
FROM customer_city_spend;

-- Q7: Running total of daily completed revenue
WITH daily_revenue AS (
    SELECT SUBSTR(order_ts, 1, 10) AS day, SUM(order_amount) AS daily_rev
    FROM orders
    WHERE status = 'completed'
    GROUP BY day
)
SELECT day, daily_rev,
       SUM(daily_rev) OVER(ORDER BY day) AS cumulative_revenue
FROM daily_revenue;
