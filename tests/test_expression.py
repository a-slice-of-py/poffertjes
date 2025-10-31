"""Unit tests for Expression system."""

import pytest
import pandas as pd
import narwhals as nw
from src.poffertjes.expression import ExpressionOp, Expression, CompositeExpression
from src.poffertjes.variable import VariableBuilder


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



class TestExpressionCreation:
    """Test suite for Expression creation and initialization."""

    def test_expression_stores_variable(self):
        """Test that Expression stores the variable reference."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        vb = VariableBuilder.from_data(df)
        x = vb.get_variables("x")
        
        expr = Expression(x, "==", 5)
        assert expr.variable is x

    def test_expression_stores_operator(self):
        """Test that Expression stores the operator."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        vb = VariableBuilder.from_data(df)
        x = vb.get_variables("x")
        
        expr = Expression(x, "==", 5)
        assert expr.operator == ExpressionOp.EQ

    def test_expression_stores_value(self):
        """Test that Expression stores the comparison value."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        vb = VariableBuilder.from_data(df)
        x = vb.get_variables("x")
        
        expr = Expression(x, "==", 5)
        assert expr.value == 5

    def test_expression_stores_upper_bound(self):
        """Test that Expression stores upper_bound for BETWEEN operations."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        vb = VariableBuilder.from_data(df)
        x = vb.get_variables("x")
        
        expr = Expression(x, "between", 5, upper_bound=10)
        assert expr.upper_bound == 10

    def test_expression_upper_bound_defaults_to_none(self):
        """Test that upper_bound defaults to None."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        vb = VariableBuilder.from_data(df)
        x = vb.get_variables("x")
        
        expr = Expression(x, "==", 5)
        assert expr.upper_bound is None


class TestExpressionToNarwhalsExpr:
    """Test suite for Expression.to_narwhals_expr() method."""

    def test_to_narwhals_expr_equality(self):
        """Test conversion of equality expression to Narwhals."""
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
        vb = VariableBuilder.from_data(df)
        x = vb.get_variables("x")
        
        expr = Expression(x, "==", 3)
        nw_expr = expr.to_narwhals_expr()
        
        # Apply the expression to filter the dataframe
        nw_df = nw.from_native(df)
        result = nw_df.filter(nw_expr)
        assert len(result) == 1
        assert result["x"].to_list()[0] == 3

    def test_to_narwhals_expr_inequality(self):
        """Test conversion of inequality expression to Narwhals."""
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
        vb = VariableBuilder.from_data(df)
        x = vb.get_variables("x")
        
        expr = Expression(x, "!=", 3)
        nw_expr = expr.to_narwhals_expr()
        
        nw_df = nw.from_native(df)
        result = nw_df.filter(nw_expr)
        assert len(result) == 4
        assert 3 not in result["x"].to_list()

    def test_to_narwhals_expr_less_than(self):
        """Test conversion of less-than expression to Narwhals."""
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
        vb = VariableBuilder.from_data(df)
        x = vb.get_variables("x")
        
        expr = Expression(x, "<", 3)
        nw_expr = expr.to_narwhals_expr()
        
        nw_df = nw.from_native(df)
        result = nw_df.filter(nw_expr)
        assert len(result) == 2
        assert result["x"].to_list() == [1, 2]

    def test_to_narwhals_expr_less_than_or_equal(self):
        """Test conversion of less-than-or-equal expression to Narwhals."""
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
        vb = VariableBuilder.from_data(df)
        x = vb.get_variables("x")
        
        expr = Expression(x, "<=", 3)
        nw_expr = expr.to_narwhals_expr()
        
        nw_df = nw.from_native(df)
        result = nw_df.filter(nw_expr)
        assert len(result) == 3
        assert result["x"].to_list() == [1, 2, 3]

    def test_to_narwhals_expr_greater_than(self):
        """Test conversion of greater-than expression to Narwhals."""
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
        vb = VariableBuilder.from_data(df)
        x = vb.get_variables("x")
        
        expr = Expression(x, ">", 3)
        nw_expr = expr.to_narwhals_expr()
        
        nw_df = nw.from_native(df)
        result = nw_df.filter(nw_expr)
        assert len(result) == 2
        assert result["x"].to_list() == [4, 5]

    def test_to_narwhals_expr_greater_than_or_equal(self):
        """Test conversion of greater-than-or-equal expression to Narwhals."""
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
        vb = VariableBuilder.from_data(df)
        x = vb.get_variables("x")
        
        expr = Expression(x, ">=", 3)
        nw_expr = expr.to_narwhals_expr()
        
        nw_df = nw.from_native(df)
        result = nw_df.filter(nw_expr)
        assert len(result) == 3
        assert result["x"].to_list() == [3, 4, 5]

    def test_to_narwhals_expr_between(self):
        """Test conversion of BETWEEN expression to Narwhals."""
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        vb = VariableBuilder.from_data(df)
        x = vb.get_variables("x")
        
        expr = Expression(x, "between", 3, upper_bound=7)
        nw_expr = expr.to_narwhals_expr()
        
        nw_df = nw.from_native(df)
        result = nw_df.filter(nw_expr)
        # Note: Narwhals is_between with closed="none" currently includes both bounds
        # This appears to be the default behavior in the current version
        # Expected: 3 <= x <= 7
        result_list = result["x"].to_list()
        assert len(result) == 5
        assert result_list == [3, 4, 5, 6, 7]

    def test_to_narwhals_expr_in(self):
        """Test conversion of IN expression to Narwhals."""
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
        vb = VariableBuilder.from_data(df)
        x = vb.get_variables("x")
        
        expr = Expression(x, "in", [2, 4])
        nw_expr = expr.to_narwhals_expr()
        
        nw_df = nw.from_native(df)
        result = nw_df.filter(nw_expr)
        assert len(result) == 2
        assert result["x"].to_list() == [2, 4]

    def test_to_narwhals_expr_with_string_values(self):
        """Test conversion with string values."""
        df = pd.DataFrame({"category": ["A", "B", "C", "A", "B"]})
        vb = VariableBuilder.from_data(df)
        cat = vb.get_variables("category")
        
        expr = Expression(cat, "==", "A")
        nw_expr = expr.to_narwhals_expr()
        
        nw_df = nw.from_native(df)
        result = nw_df.filter(nw_expr)
        assert len(result) == 2
        assert all(v == "A" for v in result["category"].to_list())

    def test_to_narwhals_expr_with_float_values(self):
        """Test conversion with float values."""
        df = pd.DataFrame({"x": [1.5, 2.5, 3.5, 4.5, 5.5]})
        vb = VariableBuilder.from_data(df)
        x = vb.get_variables("x")
        
        expr = Expression(x, ">", 3.0)
        nw_expr = expr.to_narwhals_expr()
        
        nw_df = nw.from_native(df)
        result = nw_df.filter(nw_expr)
        assert len(result) == 3
        assert result["x"].to_list() == [3.5, 4.5, 5.5]

    def test_to_narwhals_expr_invalid_operator_raises_error(self):
        """Test that invalid operator raises ValueError."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        vb = VariableBuilder.from_data(df)
        x = vb.get_variables("x")
        
        # Manually create an expression with invalid operator
        expr = Expression(x, "==", 5)
        expr.operator = "invalid"  # Bypass enum validation
        
        with pytest.raises(ValueError, match="Unsupported operator"):
            expr.to_narwhals_expr()


class TestExpressionCombination:
    """Test suite for combining expressions with __and__ and __or__."""

    def test_expression_and_creates_composite(self):
        """Test that & operator creates CompositeExpression."""
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
        vb = VariableBuilder.from_data(df)
        x = vb.get_variables("x")
        
        expr1 = Expression(x, ">", 2)
        expr2 = Expression(x, "<", 5)
        composite = expr1 & expr2
        
        assert isinstance(composite, CompositeExpression)
        assert composite.logic == "AND"
        assert len(composite.expressions) == 2

    def test_expression_or_creates_composite(self):
        """Test that | operator creates CompositeExpression."""
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
        vb = VariableBuilder.from_data(df)
        x = vb.get_variables("x")
        
        expr1 = Expression(x, "==", 1)
        expr2 = Expression(x, "==", 5)
        composite = expr1 | expr2
        
        assert isinstance(composite, CompositeExpression)
        assert composite.logic == "OR"
        assert len(composite.expressions) == 2

    def test_expression_and_preserves_expressions(self):
        """Test that & operator preserves both expressions."""
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
        vb = VariableBuilder.from_data(df)
        x = vb.get_variables("x")
        
        expr1 = Expression(x, ">", 2)
        expr2 = Expression(x, "<", 5)
        composite = expr1 & expr2
        
        assert composite.expressions[0] is expr1
        assert composite.expressions[1] is expr2

    def test_expression_or_preserves_expressions(self):
        """Test that | operator preserves both expressions."""
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
        vb = VariableBuilder.from_data(df)
        x = vb.get_variables("x")
        
        expr1 = Expression(x, "==", 1)
        expr2 = Expression(x, "==", 5)
        composite = expr1 | expr2
        
        assert composite.expressions[0] is expr1
        assert composite.expressions[1] is expr2

    def test_multiple_and_combinations(self):
        """Test chaining multiple AND operations."""
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
        vb = VariableBuilder.from_data(df)
        x = vb.get_variables("x")
        
        expr1 = Expression(x, ">", 1)
        expr2 = Expression(x, "<", 5)
        expr3 = Expression(x, "!=", 3)
        
        composite = (expr1 & expr2) & expr3
        
        assert isinstance(composite, CompositeExpression)
        assert composite.logic == "AND"

    def test_multiple_or_combinations(self):
        """Test chaining multiple OR operations."""
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
        vb = VariableBuilder.from_data(df)
        x = vb.get_variables("x")
        
        expr1 = Expression(x, "==", 1)
        expr2 = Expression(x, "==", 3)
        expr3 = Expression(x, "==", 5)
        
        composite = (expr1 | expr2) | expr3
        
        assert isinstance(composite, CompositeExpression)
        assert composite.logic == "OR"

    def test_mixed_and_or_combinations(self):
        """Test mixing AND and OR operations."""
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
        vb = VariableBuilder.from_data(df)
        x = vb.get_variables("x")
        
        expr1 = Expression(x, ">", 1)
        expr2 = Expression(x, "<", 4)
        expr3 = Expression(x, "==", 5)
        
        # (x > 1 AND x < 4) OR x == 5
        composite = (expr1 & expr2) | expr3
        
        assert isinstance(composite, CompositeExpression)
        assert composite.logic == "OR"


class TestCompositeExpression:
    """Test suite for CompositeExpression class."""

    def test_composite_expression_creation(self):
        """Test creating a CompositeExpression."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        vb = VariableBuilder.from_data(df)
        x = vb.get_variables("x")
        
        expr1 = Expression(x, ">", 1)
        expr2 = Expression(x, "<", 3)
        
        composite = CompositeExpression([expr1, expr2], "AND")
        
        assert composite.logic == "AND"
        assert len(composite.expressions) == 2

    def test_composite_expression_invalid_logic_raises_error(self):
        """Test that invalid logic raises ValueError."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        vb = VariableBuilder.from_data(df)
        x = vb.get_variables("x")
        
        expr1 = Expression(x, ">", 1)
        expr2 = Expression(x, "<", 3)
        
        with pytest.raises(ValueError, match="Logic must be 'AND' or 'OR'"):
            CompositeExpression([expr1, expr2], "INVALID")

    def test_composite_expression_repr(self):
        """Test CompositeExpression string representation."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        vb = VariableBuilder.from_data(df)
        x = vb.get_variables("x")
        
        expr1 = Expression(x, ">", 1)
        expr2 = Expression(x, "<", 3)
        
        composite = CompositeExpression([expr1, expr2], "AND")
        repr_str = repr(composite)
        
        assert "&" in repr_str
        assert ">" in repr_str
        assert "<" in repr_str

    def test_composite_expression_and_operator(self):
        """Test CompositeExpression & operator."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        vb = VariableBuilder.from_data(df)
        x = vb.get_variables("x")
        
        expr1 = Expression(x, ">", 1)
        expr2 = Expression(x, "<", 3)
        expr3 = Expression(x, "!=", 2)
        
        composite1 = CompositeExpression([expr1, expr2], "AND")
        composite2 = composite1 & expr3
        
        assert isinstance(composite2, CompositeExpression)
        assert composite2.logic == "AND"

    def test_composite_expression_or_operator(self):
        """Test CompositeExpression | operator."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        vb = VariableBuilder.from_data(df)
        x = vb.get_variables("x")
        
        expr1 = Expression(x, "==", 1)
        expr2 = Expression(x, "==", 2)
        expr3 = Expression(x, "==", 3)
        
        composite1 = CompositeExpression([expr1, expr2], "OR")
        composite2 = composite1 | expr3
        
        assert isinstance(composite2, CompositeExpression)
        assert composite2.logic == "OR"

    def test_composite_expression_to_narwhals_expr_and(self):
        """Test converting CompositeExpression with AND to Narwhals."""
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
        vb = VariableBuilder.from_data(df)
        x = vb.get_variables("x")
        
        expr1 = Expression(x, ">", 2)
        expr2 = Expression(x, "<", 5)
        composite = CompositeExpression([expr1, expr2], "AND")
        
        nw_expr = composite.to_narwhals_expr()
        nw_df = nw.from_native(df)
        result = nw_df.filter(nw_expr)
        
        assert len(result) == 2
        assert result["x"].to_list() == [3, 4]

    def test_composite_expression_to_narwhals_expr_or(self):
        """Test converting CompositeExpression with OR to Narwhals."""
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
        vb = VariableBuilder.from_data(df)
        x = vb.get_variables("x")
        
        expr1 = Expression(x, "==", 1)
        expr2 = Expression(x, "==", 5)
        composite = CompositeExpression([expr1, expr2], "OR")
        
        nw_expr = composite.to_narwhals_expr()
        nw_df = nw.from_native(df)
        result = nw_df.filter(nw_expr)
        
        assert len(result) == 2
        assert result["x"].to_list() == [1, 5]

    def test_composite_expression_nested_and(self):
        """Test nested CompositeExpression with AND logic."""
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        vb = VariableBuilder.from_data(df)
        x = vb.get_variables("x")
        
        expr1 = Expression(x, ">", 2)
        expr2 = Expression(x, "<", 8)
        expr3 = Expression(x, "!=", 5)
        
        composite = (expr1 & expr2) & expr3
        nw_expr = composite.to_narwhals_expr()
        nw_df = nw.from_native(df)
        result = nw_df.filter(nw_expr)
        
        expected = [3, 4, 6, 7]
        assert result["x"].to_list() == expected

    def test_composite_expression_nested_or(self):
        """Test nested CompositeExpression with OR logic."""
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
        vb = VariableBuilder.from_data(df)
        x = vb.get_variables("x")
        
        expr1 = Expression(x, "==", 1)
        expr2 = Expression(x, "==", 3)
        expr3 = Expression(x, "==", 5)
        
        composite = (expr1 | expr2) | expr3
        nw_expr = composite.to_narwhals_expr()
        nw_df = nw.from_native(df)
        result = nw_df.filter(nw_expr)
        
        assert result["x"].to_list() == [1, 3, 5]

    def test_composite_expression_mixed_logic(self):
        """Test CompositeExpression with mixed AND/OR logic."""
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        vb = VariableBuilder.from_data(df)
        x = vb.get_variables("x")
        
        # (x > 2 AND x < 5) OR x == 8
        expr1 = Expression(x, ">", 2)
        expr2 = Expression(x, "<", 5)
        expr3 = Expression(x, "==", 8)
        
        composite = (expr1 & expr2) | expr3
        nw_expr = composite.to_narwhals_expr()
        nw_df = nw.from_native(df)
        result = nw_df.filter(nw_expr)
        
        expected = [3, 4, 8]
        assert result["x"].to_list() == expected

    def test_composite_expression_with_multiple_variables(self):
        """Test CompositeExpression with expressions on different variables."""
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [5, 4, 3, 2, 1]})
        vb = VariableBuilder.from_data(df)
        x, y = vb.get_variables("x", "y")
        
        expr1 = Expression(x, ">", 2)
        expr2 = Expression(y, "<", 4)
        composite = CompositeExpression([expr1, expr2], "AND")
        
        nw_expr = composite.to_narwhals_expr()
        nw_df = nw.from_native(df)
        result = nw_df.filter(nw_expr)
        
        # x > 2 AND y < 4: rows where x in [3,4,5] AND y in [3,2,1]
        # Row 2: x=3, y=3 ✓
        # Row 3: x=4, y=2 ✓
        # Row 4: x=5, y=1 ✓
        assert len(result) == 3
        assert result["x"].to_list() == [3, 4, 5]


class TestExpressionIntegration:
    """Integration tests for Expression system with real dataframes."""

    def test_expression_with_pandas_dataframe(self):
        """Test Expression works with Pandas dataframes."""
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
        vb = VariableBuilder.from_data(df)
        x = vb.get_variables("x")
        
        expr = Expression(x, ">=", 3)
        nw_expr = expr.to_narwhals_expr()
        
        nw_df = nw.from_native(df)
        result = nw_df.filter(nw_expr)
        
        assert len(result) == 3
        assert result["x"].to_list() == [3, 4, 5]

    def test_expression_with_multiple_dtypes(self):
        """Test Expression works with different data types."""
        df = pd.DataFrame({
            "int_col": [1, 2, 3, 4, 5],
            "float_col": [1.5, 2.5, 3.5, 4.5, 5.5],
            "str_col": ["A", "B", "C", "D", "E"],
            "bool_col": [True, False, True, False, True]
        })
        vb = VariableBuilder.from_data(df)
        int_col, float_col, str_col, bool_col = vb.get_variables(
            "int_col", "float_col", "str_col", "bool_col"
        )
        
        # Test integer
        expr_int = Expression(int_col, ">", 3)
        nw_df = nw.from_native(df)
        result = nw_df.filter(expr_int.to_narwhals_expr())
        assert len(result) == 2
        
        # Test float
        expr_float = Expression(float_col, "<=", 3.5)
        result = nw_df.filter(expr_float.to_narwhals_expr())
        assert len(result) == 3
        
        # Test string
        expr_str = Expression(str_col, "==", "C")
        result = nw_df.filter(expr_str.to_narwhals_expr())
        assert len(result) == 1
        
        # Test boolean
        expr_bool = Expression(bool_col, "==", True)
        result = nw_df.filter(expr_bool.to_narwhals_expr())
        assert len(result) == 3

    def test_complex_expression_chain(self):
        """Test complex chained expressions."""
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        vb = VariableBuilder.from_data(df)
        x = vb.get_variables("x")
        
        # (x > 2 AND x < 8) AND (x != 4 AND x != 6)
        expr = ((x > 2) & (x < 8)) & ((x != 4) & (x != 6))
        
        nw_df = nw.from_native(df)
        result = nw_df.filter(expr.to_narwhals_expr())
        
        expected = [3, 5, 7]
        assert result["x"].to_list() == expected
