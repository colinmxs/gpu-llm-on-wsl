"""
Example Calculator Tool - Strands SDK Compatible

This is an example of a properly formatted tool for the Strands SDK.
Tools should be Python functions decorated with @strands.tool.
"""

import strands


@strands.tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression and return the result.
    
    This tool allows the agent to perform mathematical calculations
    by evaluating Python expressions safely.
    
    Args:
        expression: A mathematical expression as a string (e.g., "2 + 2", "10 * 5")
    
    Returns:
        The result of the calculation as a string
    
    Example:
        >>> calculator("2 + 2")
        "4"
        >>> calculator("(10 + 5) * 2")
        "30"
    """
    try:
        # Evaluate the expression (note: in production, use a safer eval method)
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"


@strands.tool
def convert_temperature(value: float, from_unit: str, to_unit: str) -> str:
    """
    Convert temperature between Celsius, Fahrenheit, and Kelvin.
    
    Args:
        value: The temperature value to convert
        from_unit: Source unit ('C', 'F', or 'K')
        to_unit: Target unit ('C', 'F', or 'K')
    
    Returns:
        Converted temperature as a string with unit
    
    Example:
        >>> convert_temperature(0, 'C', 'F')
        "32.0°F"
        >>> convert_temperature(100, 'C', 'K')
        "373.15K"
    """
    try:
        # Convert to Celsius first
        if from_unit.upper() == 'F':
            celsius = (value - 32) * 5/9
        elif from_unit.upper() == 'K':
            celsius = value - 273.15
        else:  # Already Celsius
            celsius = value
        
        # Convert from Celsius to target
        if to_unit.upper() == 'F':
            result = (celsius * 9/5) + 32
            return f"{result:.2f}°F"
        elif to_unit.upper() == 'K':
            result = celsius + 273.15
            return f"{result:.2f}K"
        else:  # To Celsius
            return f"{celsius:.2f}°C"
            
    except Exception as e:
        return f"Error converting temperature: {str(e)}"


if __name__ == "__main__":
    # Test the tools
    print("Testing calculator:")
    print(calculator("2 + 2"))
    print(calculator("10 * 5 + 3"))
    
    print("\nTesting temperature converter:")
    print(convert_temperature(0, 'C', 'F'))
    print(convert_temperature(100, 'C', 'K'))
    print(convert_temperature(32, 'F', 'C'))
