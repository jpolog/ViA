def solve_sudoku(puzzle):
    """
    Solve a Sudoku puzzle given in a 9x9 NumPy array.
    Returns the solved puzzle as a 9x9 NumPy array.
    """
    # Helper function to find empty cells in the puzzle
    def find_empty_cell(puzzle):
        for row in range(9):
            for col in range(9):
                if puzzle[row][col] == 0:
                    return (row, col)
        return None
    
    # Helper function to check if a value can be placed in a cell
    def is_valid(puzzle, row, col, value):
        # Check row
        if value in puzzle[row]:
            return False
        
        # Check column
        if value in puzzle[:,col]:
            return False
        
        # Check box
        box_row = (row // 3) * 3
        box_col = (col // 3) * 3
        box = puzzle[box_row:box_row+3, box_col:box_col+3]
        if value in box:
            return False
        
        return True
    
    # Helper function to solve the puzzle using backtracking
    def solve_backtrack(puzzle):
        # Find the next empty cell
        empty_cell = find_empty_cell(puzzle)
        
        # If there are no empty cells, the puzzle is solved
        if not empty_cell:
            return True
        
        # Try values from 1 to 9 in the empty cell
        row, col = empty_cell
        for value in range(1, 10):
            # Check if the value is valid in the cell
            if is_valid(puzzle, row, col, value):
                # Place the value in the cell
                puzzle[row][col] = value
                
                # Recursively solve the rest of the puzzle
                if solve_backtrack(puzzle):
                    return True
                
                # If the puzzle cannot be solved with the current value, backtrack
                puzzle[row][col] = 0
        
        # If no value can be placed in the cell, the puzzle is unsolvable
        return False
    
    # Make a copy of the puzzle so the original is not modified
    puzzle_copy = puzzle.copy()
    
    # Solve the puzzle using backtracking
    solve_backtrack(puzzle_copy)
    
    return puzzle_copy
