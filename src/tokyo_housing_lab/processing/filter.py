"""處理層：篩選。"""

from tokyo_housing_lab.sources.base import SourceRecord

# 步驟 2 確認的實際值（注意有「等」）。Purpose 不篩：02 的 Purpose 全部是空的
USED_CONDO_TYPE = "中古マンション等"


def is_used_condo(record: SourceRecord) -> bool:
    return record.fields["property_type"] == USED_CONDO_TYPE
