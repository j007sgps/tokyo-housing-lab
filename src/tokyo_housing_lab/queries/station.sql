-- 車站（load 時從 XKT015 建）
SELECT station_id, name, lines, lon, lat, passengers, passengers_year
FROM station
ORDER BY station_id;
