"""Unit tests for Expression system."""

import pytest
from src.poffertjes.expression import ExpressionOp, Expression
from src.poffertjes.variable import VariableBuilder
import pandas as pd


class TestExpressionOp:
    """Test suite for ExpressionOp enum."""

    def test_enum_values_exist(self):
        """Test that all required operators are defined in the enum."""
        assert ExpressionOp.EQ.value == "=="
        assert ExpressionOp.NE.value == "!="
        assert ExpressionOp.LT.value == "<"
        assert ExpressionOp.LE.value == "<="
        assert ExpressionOp.GT.value == ">"
        assert ExpressionOp.GE.value == ">="
        assert ExpressionOp.BETWEEN.value == "between"
        assert ExpressionOp.IN.value == "in"

    def test_enum_member_count(self):
        """Test that the enum has exactly 8 operators."""
        assert len(ExpressionOp) == 8

    def test_enum_from_string(self):
        """Test that enum can be created from string values."""
        assert ExpressionOp("==") == ExpressionOp.EQ
        assert ExpressionOp("!=") == ExpressionOp.NE
        assert ExpressionOp("<") == ExpressionOp.LT
        assert ExpressionOp("<=") == ExpressionOp.LE
        assert ExpressionOp(">") == ExpressionOp.GT
        assert ExpressionOp(">=") == ExpressionOp.GE
        assert ExpressionOp("between") == ExpressionOp.BETWEEN
        assert ExpressionOp("in") == ExpressionOp.IN

    def test_enum_invalid_string_raises_error(self):
        """Test that invalid operator strings raise ValueError."""
        with pytest.raises(ValueError):
            ExpressionOp("invalid")
        with pytest.raises(ValueError):
            ExpressionOp("=")
        with pytest.raises(ValueError):
            ExpressionOp("and")

    def test_enum_equality(self):
        """Test that enum members can be compared for equality."""
        assert ExpressionOp.EQ == ExpressionOp.EQ
        assert ExpressionOp.EQ != ExpressionOp.NE
        assert ExpressionOp.LT != ExpressionOp.LE

    def test_enum_identity(self):
        """Test that enum members are singletons."""
        op1 = ExpressionOp.EQ
        op2 = ExpressionOp.EQ
        assert op1 is op2

    def test_enum_in_collection(self):
        """Test that enum members can be used in collections."""
        ops = {ExpressionOp.EQ, ExpressionOp.NE, ExpressionOp.LT}
        assert ExpressionOp.EQ in ops
        assert ExpressionOp.GT not in ops

    def test_enum_iteration(self):
        """Test that we can iterate over all enum members."""
        all_ops = list(ExpressionOp)
        assert len(all_ops) == 8
        assert ExpressionOp.EQ in all_ops
        assert ExpressionOp.IN in all_ops

    def test_enum_name_attribute(self):
        """Test that enum members have correct name attributes."""
        assert ExpressionOp.EQ.name == "EQ"
        assert ExpressionOp.NE.name == "NE"
        assert ExpressionOp.BETWEEN.name == "BETWEEN"
        assert ExpressionOp.IN.name == "IN"

    def test_enum_value_attribute(self):
        """Test that enum members have correct value attributes."""
        assert ExpressionOp.EQ.value == "=="
        assert ExpressionOp.BETWEEN.value == "between"

    def test_enum_repr(self):
        """Test that enum members have useful string representations."""
        assert "ExpressionOp.EQ" in repr(ExpressionOp.EQ)
        assert "ExpressionOp.IN" in repr(ExpressionOp.IN)


class TestExpressionWithEnum:
    """Test that Expression class works correctly with ExpressionOp enum."""

    def test_expression_accepts_string_operator(self):
        """Test that Expression can be created with string operator."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        vb = VariableBuilder.from_data(df)
        x = vb.get_variables("x")
        
        expr = Expression(x, "==", 5)
        assert expr.operator == ExpressionOp.EQ

    def test_expression_accepts_enum_operator(self):
        """Test that Expression can be created with ExpressionOp enum."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        vb = VariableBuilder.from_data(df)
        x = vb.get_variables("x")
        
        expr = Expression(x, ExpressionOp.EQ, 5)
        assert expr.operator == ExpressionOp.EQ

    def test_expression_converts_all_operators(self):
        """Test that Expression correctly converts all operator strings to enums."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        vb = VariableBuilder.from_data(df)
        x = vb.get_variables("x")
        
        operators = ["==", "!=", "<", "<=", ">", ">=", "in"]
        expected_enums = [
            ExpressionOp.EQ,
            ExpressionOp.NE,
            ExpressionOp.LT,
            ExpressionOp.LE,
            ExpressionOp.GT,
            ExpressionOp.GE,
            ExpressionOp.IN,
        ]
        
        for op_str, expected_enum in zip(operators, expected_enums):
            expr = Expression(x, op_str, 5)
            assert expr.operator == expected_enum

    def test_expression_repr_uses_enum(self):
        """Test that Expression repr works correctly with enum operators."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        vb = VariableBuilder.from_data(df)
        x = vb.get_variables("x")
        
        expr = Expression(x, "==", 5)
        assert "==" in repr(expr)
        
        expr_in = Expression(x, "in", [1, 2, 3])
        assert "in" in repr(expr_in)
