from typing import Any, List, Tuple, Union, Callable

import pandas as pd
import numpy as np

from loguru import logger


def singleton(cls: Any) -> Callable:
    instances = {}

    def get_instance(*args, **kwargs) -> Any:
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


class Condition:
    """Class representing a condition on a variable."""

    def __init__(self, series: pd.Series, variable: "Variable", value: Any, operator: str = "=="):
        self.series = series
        self.variable = variable
        self.value = value
        self.operator = operator

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return f"Condition({self.variable} {self.operator} {self.value})"

    def __or__(self, other: "Condition") -> "ConditionalProbability":
        return ConditionalProbability(
            self.variable, (self.variable == self.value, other), pointwise=True
        )


class TernaryCondition:
    """Class representing a ternary condition (lower < x < upper)."""
    
    def __init__(self, variable: "Variable", lower: Any, upper: Any, lower_inclusive: bool, upper_inclusive: bool):
        self.variable = variable
        self.lower = lower
        self.upper = upper
        self.lower_inclusive = lower_inclusive
        self.upper_inclusive = upper_inclusive
        
        # Create the series based on the range condition
        data = variable.data
        name = variable.name
        
        if lower_inclusive and upper_inclusive:
            self.series = data.loc[(data[name] >= lower) & (data[name] <= upper), name]
        elif lower_inclusive and not upper_inclusive:
            self.series = data.loc[(data[name] >= lower) & (data[name] < upper), name]
        elif not lower_inclusive and upper_inclusive:
            self.series = data.loc[(data[name] > lower) & (data[name] <= upper), name]
        else:
            self.series = data.loc[(data[name] > lower) & (data[name] < upper), name]
    
    def __str__(self) -> str:
        return self.__repr__()
    
    def __repr__(self) -> str:
        lower_op = ">=" if self.lower_inclusive else ">"
        upper_op = "<=" if self.upper_inclusive else "<"
        return f"Condition({self.lower} {lower_op} {self.variable} {upper_op} {self.upper})"
        
    def __or__(self, other: Union["Condition", "TernaryCondition"]) -> "ConditionalProbability":
        """Handle the conditional probability bar operator '|'."""
        return ConditionalProbability(self.variable, self)


class Variable:
    """Class representing a random variable in the dataset."""

    def __init__(self, name: str, data: pd.DataFrame) -> None:
        self.name = name
        self.data = data
        self._dtype = data[name].dtype

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return f"Variable({self.name})"

    def __eq__(self, other: Any) -> Condition:
        """Create a condition where this variable equals the given value."""
        series = self.data.loc[self.data[self.name] == other, self.name]
        return Condition(series, self, other, operator="==")

    def __ne__(self, other: Any) -> Condition:
        """Create a condition where this variable does not equal the given value."""
        series = self.data.loc[self.data[self.name] != other, self.name]
        return Condition(series, self, other, operator="!=")

    def __lt__(self, other: Any) -> Condition:
        """Create a condition where this variable is less than the given value."""
        series = self.data.loc[self.data[self.name] < other, self.name]
        return Condition(series, self, other, operator="<")

    def __le__(self, other: Any) -> Condition:
        """Create a condition where this variable is less than or equal to the given value."""
        series = self.data.loc[self.data[self.name] <= other, self.name]
        return Condition(series, self, other, operator="<=")

    def __gt__(self, other: Any) -> Condition:
        """Create a condition where this variable is greater than the given value."""
        series = self.data.loc[self.data[self.name] > other, self.name]
        return Condition(series, self, other, operator=">")

    def __ge__(self, other: Any) -> Condition:
        """Create a condition where this variable is greater than or equal to the given value."""
        series = self.data.loc[self.data[self.name] >= other, self.name]
        return Condition(series, self, other, operator=">=")

    def __or__(self, other: Union[Condition, TernaryCondition]) -> "ConditionalProbability":
        """Handle the conditional probability bar operator '|'."""
        return ConditionalProbability(self, other)

    @property
    def dtype(self) -> str:
        """Return the data type of this variable."""
        if pd.api.types.is_categorical_dtype(self._dtype):
            return "categorical"
        elif pd.api.types.is_bool_dtype(self._dtype):
            return "boolean"
        elif pd.api.types.is_float_dtype(self._dtype):
            return "float"
        elif pd.api.types.is_integer_dtype(self._dtype):
            return "integer"
        else:
            return str(self._dtype)


class VariableBuilder:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def Variable(self, name: str) -> Variable:
        return Variable(name=name, data=self.data)

    def get_variables(self, *args) -> List["Variable"]:
        if not args:
            return [self.Variable(name) for name in self.data.columns]
        return [self.Variable(name) for name in args]

    @staticmethod
    def from_data(data: pd.DataFrame) -> "VariableBuilder":
        return VariableBuilder(data)


class ConditionalProbability:
    """Class representing a conditional probability expression."""

    def __init__(
        self,
        variable: Variable,
        conditions: Union[Condition, TernaryCondition, Tuple[Union[Condition, TernaryCondition]]],
        pointwise: bool = False,
    ):
        self.variable = variable
        self.conditions = conditions
        self.pointwise = pointwise

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return f"{self.variable} | {self.conditions}"

    def __or__(self, other: Union[Condition, TernaryCondition]):
        """Handle additional conditions after the first one."""
        if isinstance(self.conditions, tuple):
            # Already have multiple conditions, add another
            return ConditionalProbability(self.variable, self.conditions + (other,))
        else:
            # Have a single condition, convert to tuple and add another
            return ConditionalProbability(self.variable, (self.conditions, other))


def intersect_conditions(conditions: List[Union[Condition, TernaryCondition]]) -> pd.Index:
    combined_indices = None
    for condition in conditions:
        if combined_indices is None:
            combined_indices = condition.series.index
        else:
            combined_indices = combined_indices.intersection(condition.series.index)
    return combined_indices


def compute_probability(data: pd.DataFrame, indices: pd.Index, column: str) -> Any:
    if indices.empty:
        return pd.Series(
            index=pd.Index([], dtype=np.float64, name=column),
            dtype=np.float64,
            name="proportion",
        )

    # Get the distribution of the variable given the conditions
    return data.loc[indices, column].value_counts(normalize=True)


class RangeExpression:
    """Class for handling ternary range expressions like 'lower < x < upper'."""
    
    def __init__(self, lower: Any, variable: Variable, upper: Any, 
                 lower_inclusive: bool = False, upper_inclusive: bool = False):
        self.lower = lower
        self.variable = variable
        self.upper = upper
        self.lower_inclusive = lower_inclusive
        self.upper_inclusive = upper_inclusive
        
        # Create the condition
        data = variable.data
        name = variable.name
        
        if lower_inclusive and upper_inclusive:
            self.series = data.loc[(data[name] >= lower) & (data[name] <= upper), name]
        elif lower_inclusive and not upper_inclusive:
            self.series = data.loc[(data[name] >= lower) & (data[name] < upper), name]
        elif not lower_inclusive and upper_inclusive:
            self.series = data.loc[(data[name] > lower) & (data[name] <= upper), name]
        else:
            self.series = data.loc[(data[name] > lower) & (data[name] < upper), name]
    
    def __str__(self) -> str:
        return self.__repr__()
    
    def __repr__(self) -> str:
        lower_op = ">=" if self.lower_inclusive else ">"
        upper_op = "<=" if self.upper_inclusive else "<"
        return f"{self.lower} {lower_op} {self.variable} {upper_op} {self.upper}"
    
    def __or__(self, other: Union[Condition, TernaryCondition, "RangeExpression"]) -> ConditionalProbability:
        """Handle the conditional probability bar operator '|'."""
        return ConditionalProbability(self.variable, self)


# Add methods to enable ternary operators
def lt_and_lt(lower: Any, variable: Variable, upper: Any) -> RangeExpression:
    """Handle expressions like 'lower < x < upper'."""
    return RangeExpression(lower, variable, upper, lower_inclusive=False, upper_inclusive=False)

def lt_and_le(lower: Any, variable: Variable, upper: Any) -> RangeExpression:
    """Handle expressions like 'lower < x <= upper'."""
    return RangeExpression(lower, variable, upper, lower_inclusive=False, upper_inclusive=True)

def le_and_lt(lower: Any, variable: Variable, upper: Any) -> RangeExpression:
    """Handle expressions like 'lower <= x < upper'."""
    return RangeExpression(lower, variable, upper, lower_inclusive=True, upper_inclusive=False)

def le_and_le(lower: Any, variable: Variable, upper: Any) -> RangeExpression:
    """Handle expressions like 'lower <= x <= upper'."""
    return RangeExpression(lower, variable, upper, lower_inclusive=True, upper_inclusive=True)


@singleton
class P:
    """Probability operator class."""

    def __init__(self) -> None:
        logger.info("Probability operator instantiated.")

    def __call__(self, *args):
        logger.debug(args)

        """Handle P(target | conditions) syntax."""
        if len(args) == 0:
            raise ValueError("Empty probability expression.")

        # Check if we have P(target | x_cond, z_cond) syntax
        if len(args) > 1 and isinstance(args[0], ConditionalProbability):
            logger.debug("p(target | cond1, cond2)")
            # We have P((target | x_cond), z_cond) - need to combine conditions
            cond_prob = args[0]
            variable = cond_prob.variable
            first_condition = cond_prob.conditions
            other_conditions = args[1:]

            # Combine all conditions
            if isinstance(first_condition, tuple):
                all_conditions = first_condition + other_conditions
            else:
                all_conditions = (first_condition,) + other_conditions

            # Calculate with combined conditions indices
            combined_indices = intersect_conditions(all_conditions)

            # Calculate probability
            return compute_probability(variable.data, combined_indices, variable.name)

        if len(args) > 1 and (isinstance(args[0], Condition) or isinstance(args[0], RangeExpression)):
            logger.debug("p(cond1, cond2")
            if len(args) > 2:
                raise NotImplementedError("Chain rule...")
            return self.__call__(args[0] | args[1]) * self.__call__(args[1])

        # Handle single argument cases
        expr = args[0]

        if isinstance(expr, ConditionalProbability):
            logger.debug("p(target | conds)")
            # Calculate the conditional probability
            variable = expr.variable
            conditions = expr.conditions

            # Process multiple conditions
            if isinstance(conditions, tuple):
                logger.debug("p(target | (cond1, cond2))")
                # Combine multiple conditions
                combined_indices = intersect_conditions(conditions)
                if expr.pointwise:
                    logger.debug("p(cond0 | (cond1, cond2))")
                    if len(conditions) > 2:
                        raise NotImplementedError("More conditions...")
                    if conditions[-1].series.empty:
                        return 0
                    return len(combined_indices) / len(conditions[-1].series.index)
                # Calculate probability
                return compute_probability(
                    variable.data, combined_indices, variable.name
                )

            elif isinstance(conditions, (Condition, RangeExpression)):
                logger.debug("p(target | cond1)")
                # Single condition
                combined_indices = conditions.series.index

                # Calculate probability
                return compute_probability(
                    variable.data, combined_indices, variable.name
                )

        elif isinstance(expr, Variable):
            logger.debug("p(target)")
            # Handle unconditional probability P(target)
            return compute_probability(expr.data, expr.data.index, expr.name)

        elif isinstance(expr, (Condition, RangeExpression)):
            logger.debug("p(cond)")
            return len(expr.series) / len(expr.variable.data)

        raise ValueError("Invalid probability expression.")


# Create a singleton instance of P
p = P()