-- 某段期間東京都（$region）中古マンション等的成交筆數（「成交 → 車站」的對應率要用）
SELECT count(*) AS n
FROM transactions
WHERE starts_with(city, $region)
  AND property_type = '中古マンション等'
  AND year BETWEEN $first_year AND $last_year;
