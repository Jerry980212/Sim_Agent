# -*- coding: utf-8 -*-
"""
cfd_knowledge_mcp_server.py — CFD 领域知识 MCP Server
为 Agent 提供 CFD 仿真的最佳实践和参数建议。
避免 AI 幻觉生成不合理的仿真参数。
"""

import json
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "CFD Knowledge Server",
    description="提供 CFD 仿真最佳实践、参数建议和故障诊断",
)

# ══════════════════════════════════════════════════════════
#  领域知识库
# ══════════════════════════════════════════════════════════

TURBULENCE_GUIDE = {
    "internal-flow": {
        "recommended": "k-omega-sst",
        "alternatives": ["k-epsilon-realizable", "k-omega-standard"],
        "reason": "SST 模型结合了 k-ω 在近壁面的优势和 k-ε 在远场的稳定性",
    },
    "external-aerodynamics": {
        "recommended": "k-omega-sst",
        "alternatives": ["spalart-allmaras", "k-epsilon-realizable"],
        "reason": "外部气动力学推荐 SST 模型或 SA 模型（单方程，计算量小）",
    },
    "heat-transfer": {
        "recommended": "k-epsilon-realizable",
        "alternatives": ["k-omega-sst"],
        "reason": "Realizable k-ε 对热传递问题有较好的预测能力",
    },
    "combustion": {
        "recommended": "k-epsilon-realizable",
        "alternatives": ["k-epsilon-rng"],
        "reason": "燃烧问题推荐 Realizable k-ε，RNG 变体适合旋流燃烧器",
    },
    "turbomachinery": {
        "recommended": "k-omega-sst",
        "alternatives": ["k-epsilon-rng"],
        "reason": "旋转机械推荐 SST 模型处理逆压梯度和分离流",
    },
    "free-surface": {
        "recommended": "k-omega-sst",
        "alternatives": ["k-epsilon-realizable"],
        "reason": "自由表面流推荐 SST 以准确捕捉边界层和分离",
    },
    "high-speed-compressible": {
        "recommended": "k-omega-sst",
        "alternatives": ["k-epsilon-realizable", "spalart-allmaras"],
        "reason": "高速可压缩流推荐 SST 或 Realizable k-ε",
    },
}

MESH_QUALITY_CRITERIA = {
    "orthogonal_quality": {"min": 0.1, "good": 0.7, "description": "正交质量"},
    "skewness":           {"max": 0.95, "good": 0.25, "description": "偏斜度"},
    "aspect_ratio":       {"max": 100, "good": 10, "description": "长宽比"},
    "y_plus_guidelines": {
        "k-epsilon":       {"wall_function": "30-300", "enhanced": "<1"},
        "k-omega-sst":     {"recommended": "<1", "acceptable": "<5"},
        "spalart-allmaras": {"recommended": "<1"},
    },
}

CONVERGENCE_DEFAULTS = {
    "continuity":     1e-4,
    "x-velocity":     1e-4,
    "y-velocity":     1e-4,
    "z-velocity":     1e-4,
    "energy":         1e-6,
    "k":              1e-4,
    "epsilon":        1e-4,
    "omega":          1e-4,
}

COMMON_ERRORS = {
    "floating_point": {
        "symptoms": ["Floating point exception", "divergence", "NaN"],
        "causes": [
            "网格质量差（高偏斜度、低正交质量）",
            "边界条件不合理（如入口速度过高）",
            "时间步长过大（瞬态计算）",
            "初始条件与边界条件冲突",
            "欠松弛因子过大",
        ],
        "solutions": [
            "降低欠松弛因子（动量 0.3-0.5，压力 0.1-0.3）",
            "使用 First Order Upwind 先获得初步收敛",
            "检查并改善网格质量",
            "减小 Courant 数（密度基求解器）",
            "使用渐进式边界条件加载",
        ],
    },
    "convergence_stall": {
        "symptoms": ["残差停滞不降", "残差振荡"],
        "causes": [
            "网格不够细密",
            "离散格式阶数过高不稳定",
            "物理不稳定（如涡脱落）需要瞬态计算",
        ],
        "solutions": [
            "尝试切换到瞬态求解器",
            "加密关键区域网格",
            "先用 First Order 稳定后切换到 Second Order",
        ],
    },
    "reverse_flow": {
        "symptoms": ["Reverse flow at pressure outlet"],
        "causes": [
            "出口域不够长，回流未充分发展",
            "出口边界条件不合理",
        ],
        "solutions": [
            "延长出口段（建议 10-20 倍特征长度）",
            "调整出口压力值",
            "考虑使用 outflow 边界条件",
        ],
    },
}


# ── Tool: 推荐湍流模型 ──

@mcp.tool()
def recommend_turbulence_model(application_type: str) -> str:
    """
    根据应用场景推荐最合适的湍流模型。

    Args:
        application_type: 应用场景，如 "internal-flow", "external-aerodynamics",
                         "heat-transfer", "combustion", "turbomachinery"

    Returns:
        推荐结果的 JSON 字符串
    """
    if application_type in TURBULENCE_GUIDE:
        return json.dumps({
            "success": True,
            **TURBULENCE_GUIDE[application_type],
            "application": application_type,
        }, ensure_ascii=False)

    return json.dumps({
        "success": False,
        "error": f"未知的应用场景: {application_type}",
        "available_types": list(TURBULENCE_GUIDE.keys()),
    }, ensure_ascii=False)


# ── Tool: 获取收敛标准建议 ──

@mcp.tool()
def get_convergence_criteria(
    has_energy_equation: bool = False,
    turbulence_model: str = "k-omega-sst",
) -> str:
    """
    根据仿真配置推荐收敛标准。

    Args:
        has_energy_equation: 是否启用了能量方程
        turbulence_model: 当前使用的湍流模型

    Returns:
        收敛标准建议的 JSON 字符串
    """
    criteria = {
        "continuity": CONVERGENCE_DEFAULTS["continuity"],
        "x-velocity": CONVERGENCE_DEFAULTS["x-velocity"],
        "y-velocity": CONVERGENCE_DEFAULTS["y-velocity"],
        "z-velocity": CONVERGENCE_DEFAULTS["z-velocity"],
    }

    if has_energy_equation:
        criteria["energy"] = CONVERGENCE_DEFAULTS["energy"]

    if "epsilon" in turbulence_model:
        criteria["k"] = CONVERGENCE_DEFAULTS["k"]
        criteria["epsilon"] = CONVERGENCE_DEFAULTS["epsilon"]
    elif "omega" in turbulence_model:
        criteria["k"] = CONVERGENCE_DEFAULTS["k"]
        criteria["omega"] = CONVERGENCE_DEFAULTS["omega"]

    return json.dumps({
        "success": True,
        "criteria": criteria,
        "note": "能量方程通常需要更严格的收敛标准 (1e-6)",
    }, ensure_ascii=False)


# ── Tool: 诊断仿真错误 ──

@mcp.tool()
def diagnose_error(error_message: str) -> str:
    """
    根据错误信息诊断仿真问题并提供解决建议。

    Args:
        error_message: 错误信息或症状描述

    Returns:
        诊断结果和解决建议的 JSON 字符串
    """
    error_lower = error_message.lower()
    matches = []

    for error_type, info in COMMON_ERRORS.items():
        for symptom in info["symptoms"]:
            if symptom.lower() in error_lower:
                matches.append({
                    "error_type": error_type,
                    **info,
                })
                break

    if matches:
        return json.dumps({
            "success": True,
            "diagnosis": matches,
        }, ensure_ascii=False)

    return json.dumps({
        "success": True,
        "diagnosis": [],
        "message": "未找到完全匹配的已知错误模式，建议检查网格质量和边界条件设置",
    }, ensure_ascii=False)


# ── Tool: 网格质量评估 ──

@mcp.tool()
def evaluate_mesh_quality(
    min_orthogonal_quality: float,
    max_skewness: float,
    max_aspect_ratio: float,
) -> str:
    """
    评估网格质量并给出改进建议。

    Args:
        min_orthogonal_quality: 最小正交质量 (0-1)
        max_skewness: 最大偏斜度 (0-1)
        max_aspect_ratio: 最大长宽比

    Returns:
        网格质量评估结果的 JSON 字符串
    """
    issues = []
    overall = "good"

    criteria = MESH_QUALITY_CRITERIA

    if min_orthogonal_quality < criteria["orthogonal_quality"]["min"]:
        issues.append({
            "metric": "正交质量",
            "value": min_orthogonal_quality,
            "threshold": criteria["orthogonal_quality"]["min"],
            "severity": "critical",
            "suggestion": "正交质量过低，可能导致计算发散。建议重新划分网格。",
        })
        overall = "poor"
    elif min_orthogonal_quality < criteria["orthogonal_quality"]["good"]:
        issues.append({
            "metric": "正交质量",
            "value": min_orthogonal_quality,
            "severity": "warning",
            "suggestion": "正交质量一般，建议局部优化。",
        })
        overall = "acceptable"

    if max_skewness > criteria["skewness"]["max"]:
        issues.append({
            "metric": "偏斜度",
            "value": max_skewness,
            "threshold": criteria["skewness"]["max"],
            "severity": "critical",
            "suggestion": "偏斜度过高，强烈建议改善网格。",
        })
        overall = "poor"

    if max_aspect_ratio > criteria["aspect_ratio"]["max"]:
        issues.append({
            "metric": "长宽比",
            "value": max_aspect_ratio,
            "threshold": criteria["aspect_ratio"]["max"],
            "severity": "warning",
            "suggestion": "长宽比偏高，可能影响计算精度。",
        })
        if overall == "good":
            overall = "acceptable"

    return json.dumps({
        "success": True,
        "overall_quality": overall,
        "issues": issues,
        "y_plus_guidelines": criteria["y_plus_guidelines"],
    }, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run(transport="stdio")
