-- 4 季都有資料的「完整年」（整個資料庫來看，不是每個市區町村各自看）
SELECT price_classification, year
FROM transactions
GROUP BY price_classification, year
HAVING count(DISTINCT quarter) = 4
ORDER BY price_classification, year;
