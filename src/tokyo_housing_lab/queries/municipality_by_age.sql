-- 詳細面板：$last_year（最新完整年）往前 $years 年、取引價格，依建築年代（10 年一組）的 ㎡ 單價中位數
--   建築年空白或「戦前」（building_year 是 NULL）的不列入
SELECT
    city,
    (building_year // 10) * 10 AS decade,
    count(*) AS n,
    median(unit_price_per_m2) AS median_unit_price
FROM transactions
WHERE starts_with(city, $region)
  AND property_type = '中古マンション等'
  AND price_classification = '01'
  AND unit_price_per_m2_status = 'ok'
  AND building_year IS NOT NULL
  AND year BETWEEN $last_year - $years + 1 AND $last_year
GROUP BY city, decade
ORDER BY city, decade;
