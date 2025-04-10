import numpy as np

def is_valid(board, row, col, num):
    """Checks if placing `num` at (row, col) is valid in Sudoku."""
    # Check row & column
    if num in board[row, :] or num in board[:, col]:
        return False

    # Check 3x3 grid
    start_row, start_col = (row // 3) * 3, (col // 3) * 3
    if num in board[start_row:start_row+3, start_col:start_col+3]:
        return False

    return True

def find_empty_cell(board):
    """Finds the next empty cell (0) in the board."""
    for i in range(9):
        for j in range(9):
            if board[i, j] == 0:
                return i, j  # Return empty cell position
    return None

def solve_sudoku(board):
    """Solves Sudoku using backtracking."""
    empty = find_empty_cell(board)
    if not empty:
        return True  # No empty cells, Sudoku solved

    row, col = empty

    for num in range(1, 10):  # Try numbers 1 to 9
        if is_valid(board, row, col, num):
            board[row, col] = num

            if solve_sudoku(board):  # Recursively solve
                return True

            board[row, col] = 0  # Undo move (Backtrack)

    return False  # No solution found

# Load recognized Sudoku board
sudoku_board = np.load("sudoku_board.npy")

print("Recognized Sudoku Grid:\n", sudoku_board)

# Solve the Sudoku puzzle
if solve_sudoku(sudoku_board):
    print("\nSolved Sudoku Grid:\n", sudoku_board)
else:
    print("\nNo solution exists.")

# Save the solved Sudoku board
np.save("solved_sudoku.npy", sudoku_board)
