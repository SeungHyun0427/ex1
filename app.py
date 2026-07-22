
from __future__ import annotations

import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn


# ============================================================
# 1. 기본 설정
# ============================================================
st.set_page_config(
    page_title="타이로드 좌굴하중 예측",
    page_icon="🛠️",
    layout="wide",
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DESIGN_FEATURES = [
    "ibj_length_mm",
    "obj_length_mm",
    "fuse_diameter_mm",
    "fuse_length_mm",
    "fuse_position_mm",
    "shaft_diameter_mm",
    "curve_radius_mm",
    "curve_angle_deg",
    "offset_distance_mm",
    "transition_length_mm",
]

FEATURE_LABELS = {
    "ibj_length_mm": "이너 타이로드 길이 (mm)",
    "obj_length_mm": "아우터 타이로드 길이 (mm)",
    "fuse_diameter_mm": "퓨즈부 직경 (mm)",
    "fuse_length_mm": "퓨즈부 길이 (mm)",
    "fuse_position_mm": "퓨즈부 위치 (mm)",
    "shaft_diameter_mm": "기본 축 직경 (mm)",
    "curve_radius_mm": "곡률반경 (mm)",
    "curve_angle_deg": "곡선 각도 (deg)",
    "offset_distance_mm": "오프셋 거리 (mm)",
    "transition_length_mm": "오프셋 전이부 길이 (mm)",
}

FEATURE_MASK_MAP = {
    "fuse_diameter_mm": "mask_fuse_diameter",
    "fuse_length_mm": "mask_fuse_length",
    "fuse_position_mm": "mask_fuse_position",
    "shaft_diameter_mm": "mask_shaft_diameter",
    "curve_radius_mm": "mask_curve_radius",
    "curve_angle_deg": "mask_curve_angle",
    "offset_distance_mm": "mask_offset_distance",
    "transition_length_mm": "mask_transition_length",
}

MASK_COLUMNS = list(FEATURE_MASK_MAP.values())

SHAPE_TO_ID = {"SF": 0, "CF": 1, "OF": 2, "CNF": 3}
SHAPE_NAMES = {
    "SF": "Straight-Fuse",
    "CF": "Curved-Fuse",
    "OF": "Offset-Fuse",
    "CNF": "Curved-Non Fuse",
}

SHAPE_REQUIRED_FEATURES = {
    "SF": [
        "ibj_length_mm",
        "obj_length_mm",
        "fuse_diameter_mm",
        "fuse_length_mm",
        "fuse_position_mm",
    ],
    "CF": [
        "ibj_length_mm",
        "obj_length_mm",
        "fuse_diameter_mm",
        "fuse_length_mm",
        "fuse_position_mm",
        "curve_radius_mm",
        "curve_angle_deg",
    ],
    "OF": [
        "ibj_length_mm",
        "obj_length_mm",
        "fuse_diameter_mm",
        "fuse_length_mm",
        "fuse_position_mm",
        "offset_distance_mm",
        "transition_length_mm",
    ],
    "CNF": [
        "ibj_length_mm",
        "obj_length_mm",
        "shaft_diameter_mm",
        "curve_radius_mm",
        "curve_angle_deg",
    ],
}

DEFAULT_RANGES = {'ibj_length_mm': {'min': 200.0, 'max': 390.0, 'mean': 300.3521126760563}, 'obj_length_mm': {'min': 100.0, 'max': 220.0, 'mean': 158.19718309859155}, 'fuse_diameter_mm': {'min': 11.0, 'max': 14.6, 'mean': 12.799999999999999}, 'fuse_length_mm': {'min': 20.0, 'max': 48.0, 'mean': 33.2112676056338}, 'fuse_position_mm': {'min': 70.0, 'max': 289.0, 'mean': 128.06338028169014}, 'shaft_diameter_mm': {'min': 14.6028, 'max': 18.1957, 'mean': 16.399987323943662}, 'curve_radius_mm': {'min': 180.373, 'max': 519.683, 'mean': 349.99988591549294}, 'curve_angle_deg': {'min': 6.0128, 'max': 19.9973, 'mean': 12.999745618153364}, 'offset_distance_mm': {'min': 4.016, 'max': 27.9929, 'mean': 15.999951330203443}, 'transition_length_mm': {'min': 45.1344, 'max': 139.8801, 'mean': 92.49916635367762}}

EXPERT_NAMES = [
    "공유 전문가 1",
    "공유 전문가 2",
    "Straight-Fuse 전문가",
    "Curved-Fuse 전문가",
    "Offset-Fuse 전문가",
    "Curved-Non Fuse 전문가",
]


# ============================================================
# 2. 모델 정의
# ============================================================
FEATURE_INDEX = {name: idx for idx, name in enumerate(DESIGN_FEATURES)}
COMMON_INDEX = [FEATURE_INDEX["ibj_length_mm"], FEATURE_INDEX["obj_length_mm"]]
FUSE_INDEX = [
    FEATURE_INDEX["fuse_diameter_mm"],
    FEATURE_INDEX["fuse_length_mm"],
    FEATURE_INDEX["fuse_position_mm"],
]
CURVE_INDEX = [
    FEATURE_INDEX["curve_radius_mm"],
    FEATURE_INDEX["curve_angle_deg"],
]
OFFSET_INDEX = [
    FEATURE_INDEX["offset_distance_mm"],
    FEATURE_INDEX["transition_length_mm"],
]
SECTION_INDEX = [FEATURE_INDEX["shaft_diameter_mm"]]


class SmallEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ExpertRegressor(nn.Module):
    def __init__(self, latent_dim: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, 24),
            nn.SiLU(),
            nn.Linear(24, 12),
            nn.SiLU(),
            nn.Linear(12, 1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.network(latent).squeeze(-1)


class HierarchicalMaskedMoE(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.common_encoder = SmallEncoder(2, 16, 8)
        self.fuse_encoder = SmallEncoder(3, 16, 8)
        self.curve_encoder = SmallEncoder(2, 12, 6)
        self.offset_encoder = SmallEncoder(2, 12, 6)
        self.section_encoder = SmallEncoder(1, 8, 4)

        self.shape_embedding = nn.Embedding(4, 4)

        self.fusion = nn.Sequential(
            nn.Linear(36, 48),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(48, 32),
            nn.SiLU(),
            nn.LayerNorm(32),
        )

        self.gate = nn.Sequential(
            nn.Linear(32, 16),
            nn.SiLU(),
            nn.Linear(16, 6),
        )

        self.experts = nn.ModuleList(
            [ExpertRegressor(latent_dim=32) for _ in range(6)]
        )

        allowed_expert_mask = torch.tensor(
            [
                [1, 1, 1, 0, 0, 0],
                [1, 1, 0, 1, 0, 0],
                [1, 1, 0, 0, 1, 0],
                [1, 1, 0, 0, 0, 1],
            ],
            dtype=torch.bool,
        )
        self.register_buffer("allowed_expert_mask", allowed_expert_mask)

    def encode(
        self,
        features: torch.Tensor,
        group_masks: torch.Tensor,
        shape_id: torch.Tensor,
    ) -> torch.Tensor:
        common_z = self.common_encoder(features[:, COMMON_INDEX])
        fuse_z = self.fuse_encoder(features[:, FUSE_INDEX]) * group_masks[:, 0:1]
        curve_z = self.curve_encoder(features[:, CURVE_INDEX]) * group_masks[:, 1:2]
        offset_z = self.offset_encoder(features[:, OFFSET_INDEX]) * group_masks[:, 2:3]
        section_z = self.section_encoder(features[:, SECTION_INDEX]) * group_masks[:, 3:4]
        shape_z = self.shape_embedding(shape_id)

        combined = torch.cat(
            [common_z, fuse_z, curve_z, offset_z, section_z, shape_z],
            dim=1,
        )
        return self.fusion(combined)

    def forward(
        self,
        features: torch.Tensor,
        group_masks: torch.Tensor,
        shape_id: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        latent = self.encode(features, group_masks, shape_id)

        gate_logits = self.gate(latent)
        allowed = self.allowed_expert_mask[shape_id]
        masked_logits = gate_logits.masked_fill(
            ~allowed,
            torch.finfo(gate_logits.dtype).min,
        )
        gate_weights = torch.softmax(masked_logits, dim=1)

        expert_outputs = torch.stack(
            [expert(latent) for expert in self.experts],
            dim=1,
        )
        prediction_norm = torch.sum(gate_weights * expert_outputs, dim=1)

        return {
            "prediction_norm": prediction_norm,
            "gate_weights": gate_weights,
            "expert_outputs": expert_outputs,
            "latent": latent,
        }


# ============================================================
# 3. 전처리 및 체크포인트 로딩
# ============================================================
@dataclass
class LogTargetScaler:
    mean_log: float
    std_log: float

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        return np.exp(values * self.std_log + self.mean_log)


class ActiveFeatureStandardizer:
    def __init__(self, state: Dict) -> None:
        self.feature_names = list(state["feature_names"])
        self.feature_mask_map = dict(state["feature_mask_map"])
        self.mean_ = {k: float(v) for k, v in state["mean"].items()}
        self.std_ = {k: float(v) for k, v in state["std"].items()}

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        result = np.zeros((len(frame), len(self.feature_names)), dtype=np.float32)

        for idx, feature in enumerate(self.feature_names):
            values = frame[feature].astype(float).to_numpy()
            normalized = (values - self.mean_[feature]) / self.std_[feature]

            mask_col = self.feature_mask_map.get(feature)
            if mask_col is not None:
                normalized = normalized * frame[mask_col].astype(float).to_numpy()

            result[:, idx] = normalized.astype(np.float32)

        return result


@st.cache_resource(show_spinner=False)
def load_checkpoint_from_bytes(raw_bytes: bytes):
    # .pt 파일은 임의 코드 실행 위험이 있으므로 본인이 학습한 파일만 사용하세요.
    buffer = io.BytesIO(raw_bytes)
    try:
        checkpoint = torch.load(buffer, map_location=DEVICE, weights_only=False)
    except TypeError:
        buffer.seek(0)
        checkpoint = torch.load(buffer, map_location=DEVICE)

    required_keys = {
        "model_state_dict",
        "feature_scaler",
        "target_scaler",
    }
    missing = required_keys - set(checkpoint)
    if missing:
        raise ValueError(f"체크포인트 필수 키가 없습니다: {sorted(missing)}")

    model = HierarchicalMaskedMoE().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    feature_scaler = ActiveFeatureStandardizer(checkpoint["feature_scaler"])
    target_scaler = LogTargetScaler(**checkpoint["target_scaler"])

    return model, feature_scaler, target_scaler, checkpoint


def build_design_frame(shape_code: str, values: Dict[str, float]) -> pd.DataFrame:
    row = {feature: 0.0 for feature in DESIGN_FEATURES}
    row.update({mask: 0 for mask in MASK_COLUMNS})

    for feature in SHAPE_REQUIRED_FEATURES[shape_code]:
        row[feature] = float(values[feature])

    if shape_code in {"SF", "CF", "OF"}:
        row["mask_fuse_diameter"] = 1
        row["mask_fuse_length"] = 1
        row["mask_fuse_position"] = 1

    if shape_code in {"CF", "CNF"}:
        row["mask_curve_radius"] = 1
        row["mask_curve_angle"] = 1

    if shape_code == "OF":
        row["mask_offset_distance"] = 1
        row["mask_transition_length"] = 1

    if shape_code == "CNF":
        row["mask_shaft_diameter"] = 1

    row["shape_code"] = shape_code
    return pd.DataFrame([row])


def add_masks_to_batch(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()

    for feature in DESIGN_FEATURES:
        if feature not in output.columns:
            output[feature] = 0.0

    for mask in MASK_COLUMNS:
        output[mask] = 0

    for idx, row in output.iterrows():
        shape = str(row["shape_code"]).upper().strip()
        if shape not in SHAPE_TO_ID:
            continue

        if shape in {"SF", "CF", "OF"}:
            output.loc[idx, ["mask_fuse_diameter", "mask_fuse_length", "mask_fuse_position"]] = 1

        if shape in {"CF", "CNF"}:
            output.loc[idx, ["mask_curve_radius", "mask_curve_angle"]] = 1

        if shape == "OF":
            output.loc[idx, ["mask_offset_distance", "mask_transition_length"]] = 1

        if shape == "CNF":
            output.loc[idx, "mask_shaft_diameter"] = 1

        # 비활성 변수는 강제로 0
        active = set(SHAPE_REQUIRED_FEATURES[shape])
        for feature in DESIGN_FEATURES:
            if feature not in active:
                output.loc[idx, feature] = 0.0

    return output


@torch.no_grad()
def predict_frame(
    model: nn.Module,
    frame: pd.DataFrame,
    feature_scaler: ActiveFeatureStandardizer,
    target_scaler: LogTargetScaler,
) -> pd.DataFrame:
    frame = add_masks_to_batch(frame)

    invalid_shapes = sorted(set(frame["shape_code"]) - set(SHAPE_TO_ID))
    if invalid_shapes:
        raise ValueError(f"지원하지 않는 shape_code: {invalid_shapes}")

    for idx, row in frame.iterrows():
        shape = row["shape_code"]
        missing = [
            feature
            for feature in SHAPE_REQUIRED_FEATURES[shape]
            if pd.isna(row[feature])
        ]
        if missing:
            raise ValueError(f"{idx}행의 필수값 누락: {missing}")

    features_np = feature_scaler.transform(frame)

    fuse_mask = frame[
        ["mask_fuse_diameter", "mask_fuse_length", "mask_fuse_position"]
    ].max(axis=1)
    curve_mask = frame[
        ["mask_curve_radius", "mask_curve_angle"]
    ].max(axis=1)
    offset_mask = frame[
        ["mask_offset_distance", "mask_transition_length"]
    ].max(axis=1)
    section_mask = frame["mask_shaft_diameter"]

    group_masks_np = np.column_stack(
        [fuse_mask, curve_mask, offset_mask, section_mask]
    ).astype(np.float32)

    shape_ids_np = frame["shape_code"].map(SHAPE_TO_ID).astype(np.int64).to_numpy()

    features = torch.tensor(features_np, dtype=torch.float32, device=DEVICE)
    group_masks = torch.tensor(group_masks_np, dtype=torch.float32, device=DEVICE)
    shape_ids = torch.tensor(shape_ids_np, dtype=torch.long, device=DEVICE)

    outputs = model(features, group_masks, shape_ids)

    final_n = target_scaler.inverse_transform(
        outputs["prediction_norm"].cpu().numpy()
    )
    expert_n = target_scaler.inverse_transform(
        outputs["expert_outputs"].cpu().numpy()
    )
    gates = outputs["gate_weights"].cpu().numpy()

    result = frame[["shape_code", *DESIGN_FEATURES]].copy()
    result["predicted_buckling_load_N"] = final_n
    result["predicted_buckling_load_kN"] = final_n / 1000.0

    for j, name in enumerate(EXPERT_NAMES):
        result[f"gate_{j}"] = gates[:, j]
        result[f"expert_{j}_kN"] = expert_n[:, j] / 1000.0

    return result


def range_warnings(shape_code: str, values: Dict[str, float]) -> pd.DataFrame:
    rows = []

    for feature in SHAPE_REQUIRED_FEATURES[shape_code]:
        info = DEFAULT_RANGES.get(feature)
        if not info:
            continue

        value = float(values[feature])
        status = (
            "학습범위 내"
            if info["min"] <= value <= info["max"]
            else "학습범위 밖"
        )
        rows.append(
            {
                "설계변수": FEATURE_LABELS[feature],
                "입력값": value,
                "학습 최소": info["min"],
                "학습 최대": info["max"],
                "상태": status,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# 4. 화면
# ============================================================
st.title("🛠️ 타이로드 좌굴하중 예측 프로그램")
st.caption("형상 기반 하드 마스킹 계층적 MoE 정방향 대리모델")

with st.sidebar:
    st.header("모델 설정")
    st.warning(
        "보안을 위해 본인이 직접 생성한 신뢰할 수 있는 .pt 파일만 업로드하세요."
    )

    local_pt = Path("tie_rod_hierarchical_moe.pt")
    uploaded_pt = st.file_uploader(
        "학습된 체크포인트(.pt)",
        type=["pt", "pth"],
    )

    raw_checkpoint = None
    source_text = None

    if uploaded_pt is not None:
        raw_checkpoint = uploaded_pt.getvalue()
        source_text = f"업로드: {uploaded_pt.name}"
    elif local_pt.exists():
        raw_checkpoint = local_pt.read_bytes()
        source_text = f"로컬: {local_pt}"

    if raw_checkpoint is None:
        st.info(
            "먼저 학습 노트북에서 생성한 "
            "`tie_rod_hierarchical_moe.pt` 파일을 업로드하세요."
        )
        st.stop()

    try:
        with st.spinner("모델을 불러오는 중..."):
            model, feature_scaler, target_scaler, checkpoint = (
                load_checkpoint_from_bytes(raw_checkpoint)
            )
        st.success("모델 로딩 완료")
        st.caption(source_text)
        st.caption(f"실행 장치: {DEVICE}")
    except Exception as exc:
        st.error(f"모델 로딩 실패: {exc}")
        st.stop()

tab_single, tab_batch, tab_info = st.tabs(
    ["단일 설계 예측", "CSV 일괄 예측", "모델 정보"]
)

with tab_single:
    shape_name = st.selectbox(
        "타이로드 형상",
        options=list(SHAPE_NAMES.values()),
    )
    shape_code = next(
        code for code, name in SHAPE_NAMES.items() if name == shape_name
    )

    st.subheader(f"{shape_name} 설계변수")

    required = SHAPE_REQUIRED_FEATURES[shape_code]
    values: Dict[str, float] = {}

    columns = st.columns(2)
    for i, feature in enumerate(required):
        info = DEFAULT_RANGES.get(feature, {"min": 0.0, "max": 1000.0, "mean": 1.0})
        low = float(info["min"])
        high = float(info["max"])
        default = float(info["mean"])

        # number_input은 범위 밖 외삽 입력도 허용하되 도움말로 학습범위를 표시
        with columns[i % 2]:
            values[feature] = st.number_input(
                FEATURE_LABELS[feature],
                value=default,
                step=max((high - low) / 100.0, 0.1),
                format="%.4f",
                help=f"학습 활성범위: {low:.3f} ~ {high:.3f}",
                key=f"single_{shape_code}_{feature}",
            )

    st.divider()

    if st.button("좌굴하중 예측", type="primary", use_container_width=True):
        try:
            input_frame = build_design_frame(shape_code, values)
            prediction = predict_frame(
                model,
                input_frame,
                feature_scaler,
                target_scaler,
            )
            row = prediction.iloc[0]

            col1, col2, col3 = st.columns(3)
            col1.metric(
                "예측 좌굴하중",
                f"{row['predicted_buckling_load_kN']:.3f} kN",
            )
            col2.metric(
                "예측 좌굴하중",
                f"{row['predicted_buckling_load_N']:.1f} N",
            )

            warnings = range_warnings(shape_code, values)
            out_count = int((warnings["상태"] == "학습범위 밖").sum())
            col3.metric("범위 밖 변수", f"{out_count}개")

            if out_count:
                st.warning(
                    "학습범위를 벗어난 입력이 있습니다. "
                    "이 예측은 외삽이므로 신뢰도가 낮습니다."
                )
            else:
                st.success("모든 입력이 현재 학습범위 안에 있습니다.")

            st.subheader("학습범위 확인")
            st.dataframe(warnings, use_container_width=True, hide_index=True)

            gate_data = pd.DataFrame(
                {
                    "전문가": EXPERT_NAMES,
                    "게이트 가중치": [row[f"gate_{i}"] for i in range(6)],
                    "전문가 후보 예측 (kN)": [
                        row[f"expert_{i}_kN"] for i in range(6)
                    ],
                }
            )
            gate_data = gate_data[gate_data["게이트 가중치"] > 1e-8]

            st.subheader("허용 전문가 조합")
            st.dataframe(
                gate_data.style.format(
                    {
                        "게이트 가중치": "{:.4f}",
                        "전문가 후보 예측 (kN)": "{:.3f}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
            st.bar_chart(
                gate_data.set_index("전문가")["게이트 가중치"]
            )

            st.info(
                "현재 CF·OF·CNF 모델은 가상 데이터로 학습되었습니다. "
                "실제 설계 확정 전에는 반드시 FEA 또는 시험으로 재검증하세요."
            )
        except Exception as exc:
            st.error(f"예측 중 오류가 발생했습니다: {exc}")

with tab_batch:
    st.subheader("CSV 일괄 예측")

    template = pd.DataFrame(
        [
            {
                "shape_code": "SF",
                "ibj_length_mm": 272.5,
                "obj_length_mm": 156.5,
                "fuse_diameter_mm": 12.35,
                "fuse_length_mm": 36.5,
                "fuse_position_mm": 118.5,
                "shaft_diameter_mm": 0.0,
                "curve_radius_mm": 0.0,
                "curve_angle_deg": 0.0,
                "offset_distance_mm": 0.0,
                "transition_length_mm": 0.0,
            },
            {
                "shape_code": "CF",
                "ibj_length_mm": 272.5,
                "obj_length_mm": 156.5,
                "fuse_diameter_mm": 12.35,
                "fuse_length_mm": 36.5,
                "fuse_position_mm": 118.5,
                "shaft_diameter_mm": 0.0,
                "curve_radius_mm": 337.5,
                "curve_angle_deg": 13.7,
                "offset_distance_mm": 0.0,
                "transition_length_mm": 0.0,
            },
        ]
    )

    st.download_button(
        "입력 CSV 템플릿 다운로드",
        data=template.to_csv(index=False).encode("utf-8-sig"),
        file_name="tie_rod_prediction_template.csv",
        mime="text/csv",
    )

    batch_file = st.file_uploader(
        "예측할 CSV 파일",
        type=["csv"],
        key="batch_csv",
    )

    if batch_file is not None:
        try:
            batch_df = pd.read_csv(batch_file)
            st.write("입력 미리보기")
            st.dataframe(batch_df.head(20), use_container_width=True)

            if st.button("일괄 예측 실행", type="primary"):
                result = predict_frame(
                    model,
                    batch_df,
                    feature_scaler,
                    target_scaler,
                )
                st.success(f"{len(result):,}건 예측 완료")
                st.dataframe(result, use_container_width=True)

                st.download_button(
                    "예측 결과 CSV 다운로드",
                    data=result.to_csv(index=False).encode("utf-8-sig"),
                    file_name="tie_rod_prediction_results.csv",
                    mime="text/csv",
                )
        except Exception as exc:
            st.error(f"CSV 처리 오류: {exc}")

with tab_info:
    st.subheader("체크포인트 정보")

    model_info = {
        "모델 클래스": checkpoint.get("model_class", "정보 없음"),
        "모델 버전": checkpoint.get("model_version", "정보 없음"),
        "전문가 수": 6,
        "형상 수": 4,
        "입력 변수 수": 10,
        "실행 장치": str(DEVICE),
    }
    st.json(model_info)

    training_config = checkpoint.get("training_config")
    if training_config:
        st.subheader("학습 설정")
        st.json(training_config)

    st.subheader("프로그램 사용 한계")
    st.markdown(
        """
- 모델은 입력 설계변수가 학습범위 안에 있을 때 가장 신뢰할 수 있습니다.
- Straight-Fuse 외 형상은 현재 가상 데이터 기반입니다.
- 예측값은 FEA를 대체하는 최종 인증값이 아니라 설계 후보를 빠르게 선별하는 참고값입니다.
- 기준 좌굴하중에 가까운 설계는 반드시 FEA 또는 시험으로 재검증해야 합니다.
        """
    )
