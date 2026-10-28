-- 市區町村 × 年 × 價格種類：中古マンション的 ㎡ 單價中位數、筆數
--   spread_log = 1.4826 × MAD(ln 單價)：用中位數絕對偏差估的離散程度，不受極端值影響
--   （資料裡有 34 日圓／㎡、1 億日圓／㎡ 這種值，標準差會被拉大）。用來估中位數的誤差。
--   $region：'13' = 東京都全域，也可以給 5 碼只看一個市區町村
SELECT
    city,
    any_value(municipality) AS name,
    year,
    price_classification,
    count(*) AS n,
    median(unit_price_per_m2) AS median_unit_price,
    1.4826 * mad(ln(unit_price_per_m2)) AS spread_log
FROM transactions
WHERE starts_with(city, $region)
  AND property_type = '中古マンション等'
  AND unit_price_per_m2_status = 'ok'
GROUP BY city, year, price_classification
ORDER BY city, price_classification, year;
