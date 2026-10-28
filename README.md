# tokyo-housing-lab

東京中古公寓行情分析工具（個人學習用）。資料來源為國土交通省的公開 API（XIT001）。

目前進度：M1（立川市 → DuckDB）完成；資料擴大到東京都全域（2005Q3〜最新季），並加上市區町村地圖（趨勢／購買／投資模式）。

## 環境準備

```bash
cd tokyo-housing-lab
uv sync
cp .env.example .env   # 再把 REINFOLIB_API_KEY 換成自己的金鑰
```

金鑰也可以直接用環境變數 `REINFOLIB_API_KEY` 提供（環境變數優先於 `.env`）。

## 使用方式

```bash
uv run tokyo-housing-lab fetch   # 打 API，東京都全域 2005Q3〜 的原始 JSON 存到 data/raw/（已有的跳過，有資料的最新 4 季重抓）
uv run tokyo-housing-lab load    # 只讀 data/raw/，清理後寫進 data/tokyo_housing_lab.duckdb（不打 API）
uv run tokyo-housing-lab query   # 每季 ㎡ 單價中位數與筆數（中古マンション等）
uv run tokyo-housing-lab analyze buy                       # 各區最新 4 季的 ㎡ 單價中位數排行
uv run tokyo-housing-lab analyze trend --from 2015 --to 2025  # 各區起訖年的漲幅排行
uv run tokyo-housing-lab analyze invest                    # 各區最近 10 年的年成長率排行
uv run tokyo-housing-lab map     # 產生 data/tokyo_map.html，用瀏覽器打開
```

### 地圖

`data/tokyo_map.html` 是一個單檔 HTML（資料和邊界都嵌在裡面，只有 d3 從 cdnjs 載入）：

- **趨勢模式**：選起年和迄年，看各區 ㎡ 單價中位數漲了多少（紅＝漲、藍＝跌）
- **購買模式**：最新 4 季的 ㎡ 單價中位數，顏色越深越貴
- **投資模式**：最近 10 年的年成長率（過去走勢的整理，不是預測）
- 點地圖上的區或排行裡的名字，右側（手機在下方）會顯示詳細資料：最新價格、每年的中位數走勢、建築年代別中位數
- 灰色＝樣本不足（一年少於 30 筆）；斜線＝區間包含 0（看不出漲跌）或區間太寬（不確定）

邊界資料 `src/tokyo_housing_lab/map/tokyo_municipalities.geojson` 由 `scripts/prepare_boundaries.py` 產生：
「国土数値情報（行政区域データ）」（国土交通省）を加工して作成（スマートニュース メディア研究所 japan-topography の簡略化データを使用）。

`analyze` 的數字都附 95% 區間；一年（或最新 4 季）少於 30 筆的區標成「樣本不足」，
區間包含 0 或太寬的標成「看不出漲跌／不確定」。計算方式寫在 `src/tokyo_housing_lab/analysis/trend.py`。
這些只是過去成交資料的整理，不是價格預測，也不是投資建議。

- `fetch --region 13202`：只抓某個市區町村（2 碼＝都道府縣、5 碼＝市區町村，預設 13 東京都全域）
- `fetch --force`：全部重抓；`fetch --refresh-recent 0`：不重抓最新幾季；`fetch --start 2023Q1`：改起始季
- 成約價格只有 2021Q1 以後、取引價格只有 2005Q3 以後的資料，更早的季不會請求
- 原始檔存成 `.json.gz`（解壓後和 API 回傳的 bytes 完全相同）；東京都全域約 25MB
- `load` 每次都把 `data/raw/` 整個重新載入，跑幾次結果都一樣；任何一個檔案有問題就整批不寫入
- 同一季已經有東京都全域的檔案時，該季的市區町村檔案（例如 M1 的立川市）會被跳過，不會重複計算
- `query --city 13104`：查其他市區町村（例：新宿区）

查詢用的 SQL 在 `src/tokyo_housing_lab/queries/quarterly_median.sql`，也可以直接用 DuckDB 開資料庫自己查：

```bash
uv run python -c "import duckdb; print(duckdb.connect('data/tokyo_housing_lab.duckdb', read_only=True).sql('SELECT count(*) FROM transactions'))"
```

資料表：
- `transactions`：中古マンション等的成交資料。數值欄位都有 `<欄位>_raw`（原字串）、`<欄位>`（數值）、`<欄位>_status`（ok / empty / sentinel / unparsable）
- `fetch_log`：每次打 API 的紀錄（含 404 的「該期無資料」）

欄位的實際觀察與清理規則的依據：`docs/xit001_fields.md`

## 測試

```bash
uv run pytest            # 單元測試 + 用 fixture 的整條管線測試（不會打 API）
uv run pytest -m manual  # 真的打 API 的整合測試（需要金鑰與網路）
```

## 資料目錄

- `data/raw/`：API 回傳的原始 JSON（不進 git）
- `data/*.duckdb`：清理後的資料庫（不進 git，可由 raw 重建）

---

このサービスは、国土交通省の不動産情報ライブラリのAPI機能を使用していますが、提供情報の最新性、正確性、完全性等が保証されたものではありません
