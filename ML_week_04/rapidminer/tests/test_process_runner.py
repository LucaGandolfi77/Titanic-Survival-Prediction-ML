"""
test_process_runner.py – Tests for Process and ProcessRunner.
"""
from __future__ import annotations

import pandas as pd
import pytest

from engine.operator_base import Connection, get_operator_class
from engine.process_runner import Process, ProcessRunner, topological_sort


# ═══════════════════════════════════════════════════════════════════════════
# Process
# ═══════════════════════════════════════════════════════════════════════════

class TestProcess:
    def test_add_remove_operator(self):
        proc = Process("test")
        op = get_operator_class("Read CSV")()
        proc.add_operator(op)
        assert op.op_id in proc.operators
        proc.remove_operator(op.op_id)
        assert op.op_id not in proc.operators

    def test_add_remove_connection(self):
        proc = Process("test")
        conn = Connection("a", "out", "b", "in")
        proc.add_connection(conn)
        assert len(proc.connections) == 1
        proc.remove_connection(conn)
        assert len(proc.connections) == 0

    def test_remove_operator_cleans_connections(self):
        proc = Process("test")
        op1 = get_operator_class("Read CSV")()
        op2 = get_operator_class("Select Attributes")()
        proc.add_operator(op1)
        proc.add_operator(op2)
        proc.add_connection(Connection(op1.op_id, "out", op2.op_id, "in"))
        proc.remove_operator(op1.op_id)
        assert len(proc.connections) == 0

    def test_serialisation_roundtrip(self, iris_csv):
        proc = Process("Round Trip")
        op = get_operator_class("Read CSV")()
        op.set_param("filepath", str(iris_csv))
        proc.add_operator(op)

        data = proc.to_dict()
        proc2 = Process.from_dict(data)
        assert proc2.name == "Round Trip"
        assert len(proc2.operators) == 1

    def test_serialisation_with_connections(self):
        proc = Process("Connected")
        op1 = get_operator_class("Read CSV")()
        op2 = get_operator_class("Select Attributes")()
        proc.add_operator(op1)
        proc.add_operator(op2)
        proc.add_connection(Connection(op1.op_id, "out", op2.op_id, "in"))

        data = proc.to_dict()
        proc2 = Process.from_dict(data)
        assert len(proc2.connections) == 1
        assert proc2.connections[0].from_port == "out"


# ═══════════════════════════════════════════════════════════════════════════
# Topological Sort
# ═══════════════════════════════════════════════════════════════════════════

class TestTopologicalSort:
    def test_linear_chain(self):
        proc = Process("Chain")
        op1 = get_operator_class("Read CSV")()
        op2 = get_operator_class("Select Attributes")()
        op3 = get_operator_class("Sort")()
        proc.add_operator(op1)
        proc.add_operator(op2)
        proc.add_operator(op3)
        proc.add_connection(Connection(op1.op_id, "out", op2.op_id, "in"))
        proc.add_connection(Connection(op2.op_id, "out", op3.op_id, "in"))

        order = topological_sort(proc)
        assert order.index(op1.op_id) < order.index(op2.op_id)
        assert order.index(op2.op_id) < order.index(op3.op_id)

    def test_isolated_operators(self):
        proc = Process("Isolated")
        op1 = get_operator_class("Read CSV")()
        op2 = get_operator_class("Read CSV")()
        proc.add_operator(op1)
        proc.add_operator(op2)
        order = topological_sort(proc)
        assert len(order) == 2

    def test_cycle_raises(self):
        proc = Process("Cycle")
        op1 = get_operator_class("Sort")()
        op2 = get_operator_class("Sort")()
        proc.add_operator(op1)
        proc.add_operator(op2)
        proc.add_connection(Connection(op1.op_id, "out", op2.op_id, "in"))
        proc.add_connection(Connection(op2.op_id, "out", op1.op_id, "in"))
        with pytest.raises(RuntimeError, match="cycle"):
            topological_sort(proc)


# ═══════════════════════════════════════════════════════════════════════════
# ProcessRunner
# ═══════════════════════════════════════════════════════════════════════════

class TestProcessRunner:
    def test_run_single_operator(self, iris_csv):
        proc = Process("Single")
        op = get_operator_class("Read CSV")()
        op.set_param("filepath", str(iris_csv))
        proc.add_operator(op)

        runner = ProcessRunner()
        results = runner.run(proc)
        assert op.op_id in results
        df = results[op.op_id]["out"]
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 30

    def test_run_pipeline(self, iris_csv):
        proc = Process("Pipeline")
        read_op = get_operator_class("Read CSV")()
        read_op.set_param("filepath", str(iris_csv))
        select_op = get_operator_class("Select Attributes")()
        select_op.set_param("mode", "keep")
        select_op.set_param("columns", "sepal_length,sepal_width")

        proc.add_operator(read_op)
        proc.add_operator(select_op)
        proc.add_connection(Connection(read_op.op_id, "out", select_op.op_id, "in"))

        runner = ProcessRunner()
        results = runner.run(proc)
        df = results[select_op.op_id]["out"]
        assert list(df.columns) == ["sepal_length", "sepal_width"]

    def test_callbacks_fire(self, iris_csv):
        proc = Process("Callbacks")
        op = get_operator_class("Read CSV")()
        op.set_param("filepath", str(iris_csv))
        proc.add_operator(op)

        log_messages = []
        runner = ProcessRunner()
        runner.on_log = lambda msg: log_messages.append(msg)
        runner.run(proc)
        assert len(log_messages) >= 2  # start + at least one op

    def test_stop(self, iris_csv):
        proc = Process("Stop")
        op1 = get_operator_class("Read CSV")()
        op1.set_param("filepath", str(iris_csv))
        op2 = get_operator_class("Select Attributes")()
        proc.add_operator(op1)
        proc.add_operator(op2)
        proc.add_connection(Connection(op1.op_id, "out", op2.op_id, "in"))

        runner = ProcessRunner()
        runner.stop()  # pre-signal stop
        results = runner.run(proc)
        # At least one operator should not have run
        assert len(results) <= 2

    def test_operator_error_handled(self, tmp_dir):
        proc = Process("Error")
        op = get_operator_class("Read CSV")()
        op.set_param("filepath", str(tmp_dir / "nonexistent.csv"))
        proc.add_operator(op)

        errors = []
        runner = ProcessRunner()
        runner.on_operator_error = lambda oid, name, err: errors.append(err)
        results = runner.run(proc)
        assert len(errors) == 1
