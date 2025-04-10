import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

def extract_cells(image):
    """ Splits the Sudoku grid into 81 individual cells (9x9). """
    grid_size = image.shape[0]
    cell_size = grid_size // 9
    cells = []

    for y in range(9):
        for x in range(9):
            x1, y1 = x * cell_size, y * cell_size
            x2, y2 = (x + 1) * cell_size, (y + 1) * cell_size
            cell = image[y1:y2, x1:x2]
            cells.append(cell)

    return np.array(cells)

def preprocess_cell(cell):
    """ Prepares a Sudoku cell for digit recognition. """
    cell = cv2.resize(cell, (28, 28))
    cell = cell.astype("float32") / 255.0  # Normalize
    cell = np.expand_dims(cell, axis=-1)  # Add channel dimension
    return cell

def recognize_digits(image_path):
    """ Loads Sudoku grid and recognizes digits. """
    model = load_model("sudoku_digit_model.h5")  # Load trained model
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (450, 450))

    cells = extract_cells(image)
    board = np.zeros((9, 9), dtype=int)

    for i, cell in enumerate(cells):
        processed = preprocess_cell(cell)
        processed = np.expand_dims(processed, axis=0)  # Add batch dimension

        prediction = model.predict(processed)
        digit = np.argmax(prediction)

        # If confidence is too low, treat as empty cell
        if np.max(prediction) < 0.7:
            digit = 0

        board[i // 9, i % 9] = digit

    return board

# Example usage
sudoku_image = "processed_image.jpg"  # Use preprocessed grid image
sudoku_board = recognize_digits(sudoku_image)

# Save Sudoku board in both .npy and .npz formats
np.save("sudoku_board.npy", sudoku_board)
np.savez("sudoku_board.npz", X_train=sudoku_board, X_test=sudoku_board, y_train=sudoku_board, y_test=sudoku_board)

print("Recognized Sudoku Grid:\n", sudoku_board)
print("Sudoku board saved as 'sudoku_board.npy' and 'sudoku_board.npz'.")
