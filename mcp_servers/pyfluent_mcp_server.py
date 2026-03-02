# -*- coding: utf-8 -*-
"""
pyfluent_mcp_server.py — PyFluent MCP Server
提供标准化的 CFD 仿真工具，供 Agent 通过 MCP 协议调用。
每个工具都有严格的参数校验和错误处理。
"""

import json
import logging
from typing import Any, Optional
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from mcp.server.fastmcp import FastMCP

# ── PyFluent 延迟导入（仅在实际调用时加载） ──
_fluent_session = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pyfluent-mcp")

# ══════════════════════════════════════════════════════════
#  参数校验 — 这是 Agent+MCP 相比直接生成代码的核心优势
# ══════════════════════════════════════════════════════════

VALID_TURBULENCE_MODELS = {
    "k-epsilon-standard", "k-epsilon-rng", "k-epsilon-realizable",
    "k-omega-standard", "k-omega-sst", "k-omega-bsl",
    "spalart-allmaras",
    "reynolds-stress-linear", "reynolds-stress-quadratic",
    "les-smagorinsky", "les-wale", "les-dynamic",
    "des-sa", "des-sst",
    "laminar", "inviscid",
}

VALID_BOUNDARY_TYPES = {
    "velocity-inlet", "pressure-inlet", "mass-flow-inlet",
    "pressure-outlet", "outflow",
    "wall", "symmetry", "periodic", "axis",
    "pressure-far-field", "inlet-vent", "outlet-vent",
}

VALID_SOLVER_TYPES = {"pressure-based", "density-based"}

VALID_SCHEMES = {
    "pressure-velocity-coupling": ["SIMPLE", "SIMPLEC", "PISO", "Coupled"],
    "spatial-discretization-pressure": ["Standard", "PRESTO!", "Second Order", "Body Force Weighted"],
    "spatial-discretization-momentum": ["First Order Upwind", "Second Order Upwind", "QUICK", "Power Law"],
    "transient-formulation": ["First Order Implicit", "Second Order Implicit", "Bounded Second Order Implicit"],
}


def validate_param(name: str, value: Any, valid_set: set | dict) -> tuple[bool, str]:
    """通用参数校验函数"""
    if isinstance(valid_set, set):
        if value not in valid_set:
            suggestions = [v for v in valid_set if any(
                part in v for part in str(value).lower().split("-")
            )]
            return False, (
                f"参数 '{name}' 的值 '{value}' 无效。"
                f"可选值: {sorted(valid_set)}"
                f"{f'  您是否想用: {suggestions}' if suggestions else ''}"
            )
    elif isinstance(valid_set, dict):
        if name not in valid_set:
            return False, f"未知的参数类别 '{name}'，可选: {list(valid_set.keys())}"
        if value not in valid_set[name]:
            return False, (
                f"'{name}' 不支持 '{value}'。"
                f"可选值: {valid_set[name]}"
            )
    return True, "OK"


# ══════════════════════════════════════════════════════════
#  MCP Server 定义
# ══════════════════════════════════════════════════════════

mcp = FastMCP(
    "PyFluent Simulation Server",
    description="提供 ANSYS Fluent CFD 仿真的标准化工具接口",
)


@dataclass
class FluentState:
    """跟踪当前 Fluent 会话状态"""
    is_launched: bool = False
    mesh_loaded: bool = False
    models_set: bool = False
    materials_set: bool = False
    boundary_conditions_set: bool = False
    initialized: bool = False
    solved: bool = False
    case_file: str = ""
    turbulence_model: str = ""
    solver_type: str = ""
    iteration_count: int = 0
    errors: list = field(default_factory=list)


_state = FluentState()


# ── Tool 1: 启动 Fluent ──

@mcp.tool()
def launch_fluent(
    version: str = "3d",
    precision: str = "double",
    processor_count: int = 4,
    show_gui: bool = True,
) -> str:
    """
    启动 ANSYS Fluent 求解器会话。

    Args:
        version: 求解器维度，"2d" 或 "3d"
        precision: 计算精度，"single" 或 "double"
        processor_count: 并行核心数，建议 2-16
        show_gui: 是否显示 Fluent GUI 界面

    Returns:
        启动结果状态的 JSON 字符串
    """
    global _fluent_session, _state

    # 参数校验
    if version not in ("2d", "3d"):
        return json.dumps({
            "success": False,
            "error": f"version 必须是 '2d' 或 '3d'，收到: '{version}'"
        })
    if precision not in ("single", "double"):
        return json.dumps({
            "success": False,
            "error": f"precision 必须是 'single' 或 'double'，收到: '{precision}'"
        })
    if not 1 <= processor_count <= 128:
        return json.dumps({
            "success": False,
            "error": f"processor_count 应在 1-128 之间，收到: {processor_count}"
        })

    try:
        import ansys.fluent.core as pyfluent

        _fluent_session = pyfluent.launch_fluent(
            version=version,
            precision=precision,
            processor_count=processor_count,
            show_gui=show_gui,
            mode="solver",
        )

        _state = FluentState(is_launched=True)
        logger.info(f"Fluent launched: {version}, {precision}, {processor_count} cores")

        return json.dumps({
            "success": True,
            "message": f"Fluent 已启动 ({version}, {precision} precision, {processor_count} cores)",
            "session_id": str(id(_fluent_session)),
        })

    except Exception as e:
        error_msg = str(e)
        # 提供常见错误的友好提示
        if "license" in error_msg.lower():
            error_msg += " | 提示：请检查 ANSYS 许可证服务器配置 (ANSYSLMD_LICENSE_FILE 环境变量)"
        elif "not found" in error_msg.lower():
            error_msg += " | 提示：请确认 PyFluent 和 ANSYS Fluent 已正确安装"

        return json.dumps({"success": False, "error": error_msg})


# ── Tool 2: 读取网格 ──

@mcp.tool()
def read_mesh(file_path: str) -> str:
    """
    读取网格文件 (.msh, .msh.h5, .cas, .cas.h5)。

    Args:
        file_path: 网格文件的完整路径

    Returns:
        读取结果，包含网格统计信息的 JSON 字符串
    """
    global _state

    if not _state.is_launched:
        return json.dumps({
            "success": False,
            "error": "Fluent 尚未启动，请先调用 launch_fluent"
        })

    import os
    if not os.path.exists(file_path):
        return json.dumps({
            "success": False,
            "error": f"文件不存在: {file_path}"
        })

    valid_extensions = (".msh", ".msh.h5", ".cas", ".cas.h5", ".msh.gz")
    if not any(file_path.endswith(ext) for ext in valid_extensions):
        return json.dumps({
            "success": False,
            "error": f"不支持的文件格式。支持: {valid_extensions}"
        })

    try:
        if file_path.endswith((".cas", ".cas.h5")):
            _fluent_session.file.read_case(file_name=file_path)
        else:
            _fluent_session.file.read_mesh(file_name=file_path)

        _state.mesh_loaded = True
        _state.case_file = file_path

        # 获取网格信息
        mesh_info = {
            "success": True,
            "file": file_path,
            "message": "网格读取成功",
        }

        # 尝试获取网格统计信息
        try:
            _fluent_session.tui.mesh.check()
            mesh_info["mesh_check"] = "网格检查已执行，请查看日志"
        except Exception:
            pass

        return json.dumps(mesh_info)

    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


# ── Tool 3: 设置湍流模型 ──

@mcp.tool()
def set_turbulence_model(model_name: str) -> str:
    """
    设置湍流模型。

    Args:
        model_name: 湍流模型名称，如 "k-epsilon-realizable", "k-omega-sst" 等

    Returns:
        设置结果的 JSON 字符串
    """
    global _state

    if not _state.mesh_loaded:
        return json.dumps({
            "success": False,
            "error": "请先读取网格文件"
        })

    # 参数校验 — 这是 MCP 方案的核心优势
    valid, msg = validate_param("turbulence_model", model_name, VALID_TURBULENCE_MODELS)
    if not valid:
        return json.dumps({"success": False, "error": msg})

    try:
        viscous = _fluent_session.setup.models.viscous

        # 根据模型名称映射到 PyFluent API
        model_mapping = {
            "k-epsilon-standard":    lambda: setattr(viscous.model, "value", "k-epsilon") or
                                             setattr(viscous.k_epsilon_model, "value", "standard"),
            "k-epsilon-rng":         lambda: setattr(viscous.model, "value", "k-epsilon") or
                                             setattr(viscous.k_epsilon_model, "value", "rng"),
            "k-epsilon-realizable":  lambda: setattr(viscous.model, "value", "k-epsilon") or
                                             setattr(viscous.k_epsilon_model, "value", "realizable"),
            "k-omega-standard":      lambda: setattr(viscous.model, "value", "k-omega") or
                                             setattr(viscous.k_omega_model, "value", "standard"),
            "k-omega-sst":           lambda: setattr(viscous.model, "value", "k-omega") or
                                             setattr(viscous.k_omega_model, "value", "sst"),
            "k-omega-bsl":           lambda: setattr(viscous.model, "value", "k-omega") or
                                             setattr(viscous.k_omega_model, "value", "bsl"),
            "spalart-allmaras":      lambda: setattr(viscous.model, "value", "spalart-allmaras"),
            "laminar":               lambda: setattr(viscous.model, "value", "laminar"),
            "inviscid":              lambda: setattr(viscous.model, "value", "inviscid"),
        }

        if model_name in model_mapping:
            model_mapping[model_name]()
        else:
            # 对于 LES/DES 等高级模型，走 TUI
            _fluent_session.tui.define.models.viscous.__getattr__(model_name)()

        _state.turbulence_model = model_name
        _state.models_set = True

        return json.dumps({
            "success": True,
            "message": f"湍流模型已设置为: {model_name}",
            "model": model_name,
        })

    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


# ── Tool 4: 设置边界条件 ──

@mcp.tool()
def set_boundary_condition(
    zone_name: str,
    boundary_type: str,
    parameters: dict,
) -> str:
    """
    设置指定边界区域的边界条件。

    Args:
        zone_name: 边界区域名称（如 "inlet", "outlet", "wall-top"）
        boundary_type: 边界类型（如 "velocity-inlet", "pressure-outlet"）
        parameters: 边界参数字典，如 {"velocity_magnitude": 10.0, "temperature": 300}

    Returns:
        设置结果的 JSON 字符串
    """
    global _state

    if not _state.mesh_loaded:
        return json.dumps({"success": False, "error": "请先读取网格文件"})

    # 校验边界类型
    valid, msg = validate_param("boundary_type", boundary_type, VALID_BOUNDARY_TYPES)
    if not valid:
        return json.dumps({"success": False, "error": msg})

    # 校验参数合理性
    param_errors = _validate_boundary_params(boundary_type, parameters)
    if param_errors:
        return json.dumps({"success": False, "error": "; ".join(param_errors)})

    try:
        bc = _fluent_session.setup.boundary_conditions

        # 获取可用的边界区域列表，校验 zone_name 是否存在
        try:
            available_zones = list(bc.get_zone_names())
            if zone_name not in available_zones:
                return json.dumps({
                    "success": False,
                    "error": f"边界区域 '{zone_name}' 不存在。可用区域: {available_zones}"
                })
        except Exception:
            pass  # 某些版本可能不支持此 API

        # 设置边界条件
        zone = getattr(bc, zone_name, None)
        if zone is None:
            zone = bc[zone_name]

        for param_name, param_value in parameters.items():
            try:
                # PyFluent settings API
                param_obj = getattr(zone, param_name, None)
                if param_obj is not None:
                    if hasattr(param_obj, "value"):
                        param_obj.value = param_value
                    else:
                        param_obj = param_value
                else:
                    zone[param_name] = param_value
            except Exception as e:
                _state.errors.append(f"设置 {zone_name}.{param_name} 失败: {e}")

        _state.boundary_conditions_set = True

        return json.dumps({
            "success": True,
            "message": f"边界条件已设置: {zone_name} ({boundary_type})",
            "zone": zone_name,
            "type": boundary_type,
            "parameters": parameters,
        })

    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


def _validate_boundary_params(boundary_type: str, params: dict) -> list[str]:
    """校验边界条件参数的物理合理性"""
    errors = []

    if boundary_type == "velocity-inlet":
        vel = params.get("velocity_magnitude", params.get("vmag"))
        if vel is not None and vel < 0:
            errors.append(f"速度入口的速度大小不应为负值: {vel}")
        temp = params.get("temperature")
        if temp is not None and temp < 0:
            errors.append(f"温度不应低于 0K: {temp}")

    elif boundary_type == "pressure-outlet":
        pressure = params.get("gauge_pressure", params.get("pressure"))
        if pressure is not None and pressure < -1e6:
            errors.append(f"表压值异常偏低: {pressure} Pa")

    return errors


# ── Tool 5: 求解设置与运行 ──

@mcp.tool()
def run_solver(
    iterations: int = 100,
    solver_type: str = "pressure-based",
    convergence_criteria: Optional[dict] = None,
) -> str:
    """
    配置并运行求解器。

    Args:
        iterations: 迭代次数（稳态）或时间步数（瞬态），范围 1-100000
        solver_type: 求解器类型，"pressure-based" 或 "density-based"
        convergence_criteria: 收敛标准字典，如 {"continuity": 1e-4, "x-velocity": 1e-4}

    Returns:
        求解结果的 JSON 字符串
    """
    global _state

    if not _state.mesh_loaded:
        return json.dumps({"success": False, "error": "请先读取网格文件"})

    # 参数校验
    valid, msg = validate_param("solver_type", solver_type, VALID_SOLVER_TYPES)
    if not valid:
        return json.dumps({"success": False, "error": msg})

    if not 1 <= iterations <= 100000:
        return json.dumps({
            "success": False,
            "error": f"迭代次数应在 1-100000 之间，收到: {iterations}"
        })

    try:
        solver = _fluent_session.setup.general
        solver.solver_type = solver_type

        # 设置收敛标准
        if convergence_criteria:
            monitors = _fluent_session.solution.monitors.residual
            for residual_name, threshold in convergence_criteria.items():
                try:
                    res = getattr(monitors, residual_name, None)
                    if res is not None:
                        res.absolute_criteria = threshold
                except Exception as e:
                    _state.errors.append(
                        f"设置收敛标准 {residual_name}={threshold} 失败: {e}"
                    )

        # 初始化
        if not _state.initialized:
            _fluent_session.solution.initialization.hybrid_initialize()
            _state.initialized = True

        # 运行计算
        _fluent_session.solution.run_calculation.iterate(
            number_of_iterations=iterations
        )

        _state.solved = True
        _state.iteration_count += iterations

        return json.dumps({
            "success": True,
            "message": f"求解完成: {iterations} 次迭代",
            "total_iterations": _state.iteration_count,
            "solver_type": solver_type,
        })

    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


# ── Tool 6: 导出结果 ──

@mcp.tool()
def export_results(
    output_dir: str,
    export_contours: list[str] = None,
    export_data: bool = True,
    image_format: str = "png",
    image_resolution: tuple[int, int] = (1920, 1080),
) -> str:
    """
    导出仿真结果（云图、数据文件等）。

    Args:
        output_dir: 输出目录路径
        export_contours: 要导出的云图变量列表，如 ["pressure", "velocity-magnitude", "temperature"]
        export_data: 是否导出 .cas/.dat 文件
        image_format: 图片格式，"png" 或 "jpg"
        image_resolution: 图片分辨率 (width, height)

    Returns:
        导出结果的 JSON 字符串
    """
    global _state

    if not _state.solved:
        return json.dumps({"success": False, "error": "请先运行求解器"})

    import os
    os.makedirs(output_dir, exist_ok=True)

    exported_files = []
    errors = []

    try:
        # 导出 case 和 data 文件
        if export_data:
            case_path = os.path.join(output_dir, "result.cas.h5")
            _fluent_session.file.write_case_data(file_name=case_path)
            exported_files.append(case_path)

        # 导出云图
        if export_contours:
            graphics = _fluent_session.results.graphics.contour
            for var_name in export_contours:
                try:
                    contour_name = f"contour-{var_name}"
                    contour = graphics.create(contour_name)
                    contour.field = var_name
                    contour.display()

                    img_path = os.path.join(
                        output_dir, f"{var_name}.{image_format}"
                    )
                    _fluent_session.results.graphics.picture.save_picture(
                        file_name=img_path,
                        x_resolution=image_resolution[0],
                        y_resolution=image_resolution[1],
                    )
                    exported_files.append(img_path)

                except Exception as e:
                    errors.append(f"导出 {var_name} 云图失败: {e}")

        return json.dumps({
            "success": True,
            "message": f"导出完成，共 {len(exported_files)} 个文件",
            "files": exported_files,
            "errors": errors if errors else None,
        })

    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


# ── Tool 7: 查询当前状态 ──

@mcp.tool()
def get_session_status() -> str:
    """
    获取当前 Fluent 会话的完整状态信息。

    Returns:
        当前状态的 JSON 字符串
    """
    import dataclasses
    return json.dumps(dataclasses.asdict(_state), ensure_ascii=False)


# ── Tool 8: 参数验证工具（Agent 可在规划阶段调用） ──

@mcp.tool()
def validate_simulation_parameter(
    category: str,
    value: str,
) -> str:
    """
    在执行前验证仿真参数是否合法。Agent 应在设置参数前调用此工具。

    Args:
        category: 参数类别，如 "turbulence_model", "boundary_type", "solver_type"
        value: 要验证的参数值

    Returns:
        验证结果的 JSON 字符串
    """
    category_map = {
        "turbulence_model": VALID_TURBULENCE_MODELS,
        "boundary_type": VALID_BOUNDARY_TYPES,
        "solver_type": VALID_SOLVER_TYPES,
    }

    if category not in category_map:
        return json.dumps({
            "valid": False,
            "error": f"未知的参数类别: {category}",
            "available_categories": list(category_map.keys()),
        })

    valid, msg = validate_param(category, value, category_map[category])
    return json.dumps({
        "valid": valid,
        "message": msg,
        "category": category,
        "value": value,
    })


# ── 启动 MCP Server ──

if __name__ == "__main__":
    mcp.run(transport="stdio")
