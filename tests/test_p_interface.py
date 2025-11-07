"""Unit tests for P singleton and API."""

import pytest
import pandas as pd

from poffertjes.p_interface import P, p
from poffertjes.variable import VariableBuilder
from poffertjes.exceptions import DataframeError, VariableError


class TestPSingleton:
    """Test the singleton pattern implementation of P class."""

    def test_singleton_same_instance(self):
        """Test that multiple instantiations return the same instance."""
        p1 = P()
        p2 = P()
        assert p1 is p2, "P() should return the same instance"

    def test_singleton_exported_instance(self):
        """Test that the exported p instance is the singleton."""
        p1 = P()
        assert p is p1, "Exported p should be the singleton instance"

    def test_singleton_multiple_imports(self):
        """Test that importing p multiple times gives the same instance."""
        from poffertjes import p as p1
        from poffertjes.p_interface import p as p2
        
        assert p1 is p2, "Multiple imports should reference the same instance"

    def test_singleton_identity(self):
        """Test that singleton maintains identity across different access patterns."""
        p1 = P()
        from poffertjes import p as p2
        p3 = P()
        
        assert p1 is p2 is p3, "All references should point to the same instance"


class TestPCallMethod:
    """Test the __call__ method of P class."""

    def test_call_with_no_arguments_raises_error(self):
        """Test that calling p() with no arguments raises ValueError."""
        with pytest.raises(VariableError, match="requires at least one argument"):
            p()

    def test_call_validates_before_query_execution(self):
        """Test that validation happens before attempting to execute query."""
        df = pd.DataFrame({'x': [1, 2, 3, 1, 2]})
        vb = VariableBuilder.from_data(df)
        x = vb.get_variables('x')
        
        # Validation should pass, and QueryBuilder should return placeholder result
        # This tests that the validation logic works correctly
        result = p(x)
        # QueryBuilder is now implemented and returns placeholder strings
        assert isinstance(result, str)
        assert "DistributionQuery" in result

    def test_call_with_single_expression(self):
        """Test that p(x == value) validates correctly."""
        df = pd.DataFrame({'x': [1, 2, 3, 1, 2]})
        vb = VariableBuilder.from_data(df)
        x = vb.get_variables('x')
        
        # Validation should pass, and QueryBuilder should return placeholder result
        result = p(x == 1)
        # QueryBuilder is now implemented and returns placeholder strings
        assert isinstance(result, str)
        assert "ScalarQuery" in result

    def test_call_with_multiple_variables(self):
        """Test that p(x, y) validates correctly."""
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        vb = VariableBuilder.from_data(df)
        x, y = vb.get_variables('x', 'y')
        
        # Validation should pass, and QueryBuilder should return placeholder result
        result = p(x, y)
        # QueryBuilder is now implemented and returns placeholder strings
        assert isinstance(result, str)
        assert "DistributionQuery" in result


class TestPVariableExtraction:
    """Test the _extract_variables method."""

    def test_extract_from_single_variable(self):
        """Test extracting variables from a single Variable argument."""
        df = pd.DataFrame({'x': [1, 2, 3]})
        vb = VariableBuilder.from_data(df)
        x = vb.get_variables('x')
        
        variables = p._extract_variables((x,))
        assert len(variables) == 1
        assert variables[0] is x

    def test_extract_from_single_expression(self):
        """Test extracting variables from a single Expression argument."""
        df = pd.DataFrame({'x': [1, 2, 3]})
        vb = VariableBuilder.from_data(df)
        x = vb.get_variables('x')
        expr = x == 1
        
        variables = p._extract_variables((expr,))
        assert len(variables) == 1
        assert variables[0] is x

    def test_extract_from_multiple_variables(self):
        """Test extracting variables from multiple Variable arguments."""
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        vb = VariableBuilder.from_data(df)
        x, y = vb.get_variables('x', 'y')
        
        variables = p._extract_variables((x, y))
        assert len(variables) == 2
        assert variables[0] is x
        assert variables[1] is y

    def test_extract_from_composite_expression(self):
        """Test extracting variables from CompositeExpression."""
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        vb = VariableBuilder.from_data(df)
        x, y = vb.get_variables('x', 'y')
        composite = (x > 1) & (y < 6)
        
        variables = p._extract_variables((composite,))
        assert len(variables) == 2
        # Variables should be extracted from the composite expression
        assert any(v.name == 'x' for v in variables)
        assert any(v.name == 'y' for v in variables)

    def test_extract_from_mixed_arguments(self):
        """Test extracting variables from mixed Variable and Expression arguments."""
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6], 'z': [7, 8, 9]})
        vb = VariableBuilder.from_data(df)
        x, y, z = vb.get_variables('x', 'y', 'z')
        
        variables = p._extract_variables((x, y == 5, z))
        assert len(variables) == 3
        assert variables[0] is x
        assert variables[1] is y
        assert variables[2] is z


class TestPDataframeValidation:
    """Test the _validate_same_dataframe method."""

    def test_validate_single_variable(self):
        """Test validation with a single variable (should pass)."""
        df = pd.DataFrame({'x': [1, 2, 3]})
        vb = VariableBuilder.from_data(df)
        x = vb.get_variables('x')
        
        # Should not raise an error
        p._validate_same_dataframe([x])

    def test_validate_multiple_variables_same_dataframe(self):
        """Test validation with multiple variables from the same dataframe (should pass)."""
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        vb = VariableBuilder.from_data(df)
        x, y = vb.get_variables('x', 'y')
        
        # Should not raise an error
        p._validate_same_dataframe([x, y])

    def test_validate_empty_list(self):
        """Test validation with empty list (should pass)."""
        # Should not raise an error
        p._validate_same_dataframe([])

    def test_validate_different_dataframes_raises_error(self):
        """Test that mixing variables from different dataframes raises ValueError."""
        df1 = pd.DataFrame({'x': [1, 2, 3]})
        df2 = pd.DataFrame({'y': [4, 5, 6]})
        
        vb1 = VariableBuilder.from_data(df1)
        vb2 = VariableBuilder.from_data(df2)
        
        x = vb1.get_variables('x')
        y = vb2.get_variables('y')
        
        with pytest.raises(DataframeError, match="different dataframes cannot be mixed"):
            p._validate_same_dataframe([x, y])

    def test_validate_error_message_includes_variable_names(self):
        """Test that error message includes the names of conflicting variables."""
        df1 = pd.DataFrame({'x': [1, 2, 3]})
        df2 = pd.DataFrame({'y': [4, 5, 6]})
        
        vb1 = VariableBuilder.from_data(df1)
        vb2 = VariableBuilder.from_data(df2)
        
        x = vb1.get_variables('x')
        y = vb2.get_variables('y')
        
        with pytest.raises(DataframeError, match="'x'.*'y'"):
            p._validate_same_dataframe([x, y])

    def test_validate_three_variables_two_dataframes(self):
        """Test validation with three variables where two are from different dataframes."""
        df1 = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        df2 = pd.DataFrame({'z': [7, 8, 9]})
        
        vb1 = VariableBuilder.from_data(df1)
        vb2 = VariableBuilder.from_data(df2)
        
        x, y = vb1.get_variables('x', 'y')
        z = vb2.get_variables('z')
        
        # x and y are from same dataframe, but z is from different one
        with pytest.raises(DataframeError, match="different dataframes cannot be mixed"):
            p._validate_same_dataframe([x, y, z])


class TestPIntegrationWithVariables:
    """Integration tests for P with Variable objects."""

    def test_p_accepts_variable_from_builder(self):
        """Test that p() accepts variables created from VariableBuilder."""
        df = pd.DataFrame({'x': [1, 2, 3, 1, 2]})
        vb = VariableBuilder.from_data(df)
        x = vb.get_variables('x')
        
        # Should not raise validation errors, QueryBuilder should return placeholder result
        result = p(x)
        # QueryBuilder is now implemented and returns placeholder strings
        assert isinstance(result, str)
        assert "DistributionQuery" in result

    def test_p_rejects_mixed_dataframe_variables_before_execution(self):
        """Test that p() rejects variables from different dataframes during validation.
        
        This test verifies that validation happens BEFORE attempting to execute the query,
        so we should get a ValueError about mixed dataframes, not an ImportError about QueryBuilder.
        """
        df1 = pd.DataFrame({'x': [1, 2, 3]})
        df2 = pd.DataFrame({'y': [4, 5, 6]})
        
        vb1 = VariableBuilder.from_data(df1)
        vb2 = VariableBuilder.from_data(df2)
        
        x = vb1.get_variables('x')
        y = vb2.get_variables('y')
        
        # Should raise DataframeError about mixed dataframes, not ImportError
        with pytest.raises(DataframeError, match="different dataframes"):
            p(x, y)


class TestPImportIntegration:
    """Integration tests for importing p from poffertjes package.
    
    These tests verify that the p singleton can be imported correctly
    and maintains its singleton behavior across different import patterns.
    This satisfies Requirements 2.1 and 14.6.
    """

    def test_import_p_from_poffertjes(self):
        """Test that 'from poffertjes import p' works correctly.
        
        Requirement 2.1: WHEN a user imports poffertjes THEN they SHALL be able 
        to use `from poffertjes import p`
        """
        # This import should work without errors
        from poffertjes import p as imported_p
        
        # Verify it's the P singleton instance
        assert isinstance(imported_p, P)
        assert imported_p is p

    def test_import_p_is_singleton(self):
        """Test that imported p is the singleton instance.
        
        Requirement 2.2: WHEN the P class is instantiated THEN it SHALL follow 
        the singleton pattern
        """
        from poffertjes import p as p1
        
        # Create a new instance
        p2 = P()
        
        # They should be the same object
        assert p1 is p2

    def test_import_p_multiple_times_same_instance(self):
        """Test that importing p multiple times gives the same instance.
        
        Requirement 2.4: WHEN multiple imports of `p` occur THEN they SHALL 
        reference the same singleton instance
        """
        # Import from different locations
        from poffertjes import p as p1
        from poffertjes.p_interface import p as p2
        
        # Both should be the same instance
        assert p1 is p2
        
        # And both should be the module-level p
        assert p1 is p
        assert p2 is p

    def test_import_p_callable(self):
        """Test that imported p is callable.
        
        Requirement 2.3: WHEN a user calls `p(...)` THEN the `__call__` method 
        SHALL be invoked
        """
        from poffertjes import p as imported_p
        
        # Verify it's callable
        assert callable(imported_p)
        
        # Verify calling it invokes __call__ (will fail without arguments)
        with pytest.raises(VariableError, match="requires at least one argument"):
            imported_p()

    def test_import_p_works_in_user_code_pattern(self):
        """Test the typical user import pattern works correctly.
        
        This simulates how users will actually use the library:
        1. Import p from poffertjes
        2. Create variables from a dataframe
        3. Use p with those variables
        
        Requirements 2.1, 14.6: User-friendly import and usage pattern
        """
        # Typical user code pattern
        from poffertjes import p as user_p
        
        df = pd.DataFrame({'x': [1, 2, 3, 1, 2], 'y': [5, 6, 7, 5, 6]})
        vb = VariableBuilder.from_data(df)
        x, y = vb.get_variables('x', 'y')
        
        # Verify p is callable with variables, QueryBuilder should return placeholder result
        result = user_p(x)
        # QueryBuilder is now implemented and returns placeholder strings
        assert isinstance(result, str)
        assert "DistributionQuery" in result
        
        # Verify validation works
        df2 = pd.DataFrame({'z': [8, 9, 10]})
        vb2 = VariableBuilder.from_data(df2)
        z = vb2.get_variables('z')
        
        with pytest.raises(DataframeError, match="different dataframes"):
            user_p(x, z)

    def test_import_p_available_in_all_attribute(self):
        """Test that p is listed in __all__ for proper export.
        
        Requirement 14.6: Proper API exposure through __all__
        """
        import poffertjes
        
        # Verify p is in __all__
        assert 'p' in poffertjes.__all__
        
        # Verify p is accessible as attribute
        assert hasattr(poffertjes, 'p')
        assert poffertjes.p is p

    def test_import_star_includes_p(self):
        """Test that 'from poffertjes import *' includes p.
        
        Requirement 14.6: Proper API exposure
        """
        # Create a new namespace to test import *
        namespace = {}
        exec("from poffertjes import *", namespace)
        
        # Verify p is in the namespace
        assert 'p' in namespace
        assert isinstance(namespace['p'], P)
        assert namespace['p'] is p
