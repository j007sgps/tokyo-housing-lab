-- 市區町村代碼 → 名稱（中古マンション等有成交的區）
SELECT city, any_value(municipality) AS name
FROM transactions
WHERE starts_with(city, $region)
  AND property_type = '中古マンション等'
GROUP BY city
ORDER BY city;
