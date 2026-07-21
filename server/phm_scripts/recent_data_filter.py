"""
recent_data_filter.py — 최근 N일 이내에 수집된 CSV 파일만 선택하는 공용 유틸리티.

주기적 자동 재학습(phm_auto_retrain)에서 데이터 드리프트를 반영하기 위해,
누적된 전체 히스토리 대신 최근 N일치 데이터만 학습에 사용할 수 있게 한다.
수동 트리거(phm_retrain)는 기본적으로 recent_days를 지정하지 않으므로
영향받지 않는다(전체 히스토리 사용, 기존 동작 그대로).

파일명 앞부분의 타임스탬프(yyyyMMdd_HHmmss — PHM-Motion-Studio C# 수집기의
명명 규칙, 예: 20260720_053200_AllAxes_Continuous_Combined.csv)를 우선 사용하고,
파일명에서 타임스탬프를 못 찾을 때만 mtime으로 폴백한다. 파일이 장비 PC에서
서버로 전송/동기화되는 과정에서 mtime이 실제 수집 시각과 달라질 수 있어
파일명 쪽이 더 신뢰할 수 있기 때문이다.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

_TS_RE = re.compile(r"(\d{8}_\d{6})")


def _extract_file_timestamp(path: Path) -> Optional[datetime]:
    """파일명에서 yyyyMMdd_HHmmss 형태의 타임스탬프를 찾아 반환합니다. 없으면 None."""
    m = _TS_RE.search(path.name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def filter_recent_files(
    files: List[Path],
    recent_days: Optional[float],
    now: Optional[datetime] = None,
) -> List[Path]:
    """recent_days가 None이거나 0 이하면 files를 그대로 반환합니다 (필터 비활성 — 기존 동작).

    그 외에는 (now - recent_days)일 이후에 수집된 파일만 남깁니다.
    파일명에서 타임스탬프를 못 찾은 파일은 mtime을 대신 사용합니다(그마저 실패하면 제외).
    """
    if not files or not recent_days or recent_days <= 0:
        return files

    now = now or datetime.now()
    cutoff = now - timedelta(days=float(recent_days))

    kept: List[Path] = []
    fallback_count = 0
    for f in files:
        ts = _extract_file_timestamp(f)
        if ts is None:
            fallback_count += 1
            try:
                ts = datetime.fromtimestamp(f.stat().st_mtime)
            except OSError:
                continue
        if ts >= cutoff:
            kept.append(f)

    extra = f", 파일명에 타임스탬프 없어 mtime 사용={fallback_count}개" if fallback_count else ""
    print(
        f"[data] 최근 {recent_days}일 필터: {len(files)}개 → {len(kept)}개 "
        f"(기준시각={cutoff.isoformat(timespec='seconds')}{extra})",
        flush=True,
    )
    return kept
