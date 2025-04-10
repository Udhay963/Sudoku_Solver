import cv2
import numpy as np

def overlay_solution(original_image, recognized_board, solved_board):
    """Overlays the solved Sudoku numbers onto the original image."""
    image = cv2.imread(original_image)

    # Define grid size
    size = image.shape[0]
    cell_size = size // 9

    for i in range(9):
        for j in range(9):
            if recognized_board[i, j] == 0:  # Only overlay missing numbers
                text = str(solved_board[i, j])
                x, y = j * cell_size + cell_size // 3, i * cell_size + 2 * cell_size // 3
                cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0, 255, 0), 2, cv2.LINE_AA)

    return image

# Load recognized and solved Sudoku boards
recognized_board = np.load("sudoku_board.npy")
solved_board = np.load("solved_sudoku.npy")

# Load and overlay the solution onto the original Sudoku image
original_image = "processed_image.jpg"  # Use preprocessed image
solved_image = overlay_solution(original_image, recognized_board, solved_board)

# Save the final solved image
cv2.imwrite("solved_sudoku.jpg", solved_image)

# Load images for display
unsolved_img = cv2.imread(original_image)
solved_img = cv2.imread("solved_sudoku.jpg")

# Resize images to be same height
height = max(unsolved_img.shape[0], solved_img.shape[0])
width = unsolved_img.shape[1] + solved_img.shape[1] + 40  # Add 4 cm gap (~40 pixels)
side_by_side = np.ones((height + 50, width, 3), dtype=np.uint8) * 255  # White background

# Place images with a 4 cm (40px) gap
side_by_side[:unsolved_img.shape[0], :unsolved_img.shape[1]] = unsolved_img
side_by_side[:solved_img.shape[0], unsolved_img.shape[1] + 40:] = solved_img

# Add Labels
cv2.putText(side_by_side, "Unsolved Sudoku", (50, height + 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
cv2.putText(side_by_side, "Solved Sudoku", (unsolved_img.shape[1] + 80, height + 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

# Display both images
cv2.imshow("Sudoku Comparison", side_by_side)
cv2.waitKey(0)  # Wait for key press to close
cv2.destroyAllWindows()

print("✅ Solved Sudoku image displayed successfully.")
