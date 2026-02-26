"""
test_operator_base.py – Tests for operator_base.py:
  - Port and PortType
  - Operator registration and discovery
  - Operator serialisation round‑trip
  - Connection data‑class
"""
from __future__ import annotations

import pytest

from engine.operator_base import (
    Connection,
    Operator,
    OpCategory,
    ParamKind,
    ParamSpec,
    Port,
    PortType,
    get_operator_class,
    list_operator_types,
    operators_by_category,
    register_operator,
)


# ═══════════════════════════════════════════════════════════════════════════
# PortType / Port
# ═══════════════════════════════════════════════════════════════════════════

class TestPortType:
    def test_enum_values(self):
        assert PortType.EXAMPLE_SET.value == "ExampleSet"
        assert PortType.MODEL.value == "Model"
        assert PortType.PERFORMANCE.value == "Performance"
        assert PortType.ANY.value == "Any"

    def test_port_defaults(self):
        p = Port("test_port")
        assert p.name == "test_port"
        assert p.port_type == PortType.EXAMPLE_SET
        assert p.optional is False
        assert p.value is None

    def test_port_custom(self):
        p = Port("model_out", PortType.MODEL, optional=True)
        assert p.port_type == PortType.MODEL
        assert p.optional is True


# ═══════════════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════════════

class TestRegistry:
    def test_all_operators_registered(self):
        types = list_operator_types()
        assert len(types) >= 60  # 61 expected

    def test_get_operator_class(self):
        cls = get_operator_class("Read CSV")
        assert issubclass(cls, Operator)

    def test_unknown_operator_raises(self):
        with pytest.raises(KeyError):
            get_operator_class("NonExistent Operator")

    def test_operators_by_category(self):
        groups = operators_by_category()
        assert OpCategory.DATA in groups
        assert OpCategory.MODEL in groups
        assert OpCategory.TRANSFORM in groups
        assert len(groups[OpCategory.DATA]) >= 7
        assert len(groups[OpCategory.MODEL]) >= 13

    def test_operator_instantiation(self):
        for op_type in list_operator_types():
            cls = get_operator_class(op_type)
            op = cls()
            assert op.op_type == op_type
            assert isinstance(op.op_id, str)
            assert len(op.op_id) > 0


# ═══════════════════════════════════════════════════════════════════════════
# Operator base class
# ═══════════════════════════════════════════════════════════════════════════

class TestOperator:
    def test_serialisation_roundtrip(self):
        cls = get_operator_class("Read CSV")
        op = cls()
        op.x, op.y = 150, 200
        op.display_name = "My CSV Reader"
        op.set_param("filepath", "/some/path.csv")

        d = op.to_dict()
        assert d["type"] == "Read CSV"
        assert d["display_name"] == "My CSV Reader"
        assert d["position"] == [150, 200]
        assert d["params"]["filepath"] == "/some/path.csv"

    def test_reset_execution(self):
        cls = get_operator_class("Read CSV")
        op = cls()
        op.executed = True
        op.error = "something"
        op.exec_time = 1.5
        op.reset_execution()
        assert op.executed is False
        assert op.error is None
        assert op.exec_time == 0.0

    def test_get_set_param(self):
        cls = get_operator_class("Split Data")
        op = cls()
        op.set_param("ratio", 0.8)
        assert op.get_param("ratio") == 0.8

    def test_param_specs_exist(self):
        cls = get_operator_class("Random Forest")
        op = cls()
        assert len(op.param_specs) > 0
        names = [ps.name for ps in op.param_specs]
        assert "n_estimators" in names


# ═══════════════════════════════════════════════════════════════════════════
# Connection
# ═══════════════════════════════════════════════════════════════════════════

class TestConnection:
    def test_roundtrip(self):
        c = Connection("op_a", "out", "op_b", "in")
        d = c.to_dict()
        c2 = Connection.from_dict(d)
        assert c2.from_op_id == "op_a"
        assert c2.from_port == "out"
        assert c2.to_op_id == "op_b"
        assert c2.to_port == "in"

    def test_fields(self):
        c = Connection("x", "port1", "y", "port2")
        assert c.from_op_id == "x"
        assert c.to_port == "port2"
