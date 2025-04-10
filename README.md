# 🧩 Sudoku Solver Using Image Processing Techniques

A Python-based solution that uses **OpenCV**, **Convolutional Neural Networks (CNNs)**, and a **Backtracking Algorithm** to solve Sudoku puzzles from raw images.

---

## 📌 Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [System Requirements](#system-requirements)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [How to Run](#how-to-run)
- [Results](#results)
- [Applications](#applications)
- [Future Scope](#future-scope)
- [Author](#author)

---

## 🧠 Introduction

This project aims to automate the process of solving Sudoku puzzles using computer vision and AI. The system reads an image of a Sudoku puzzle, detects the grid, recognizes the digits using a CNN model, solves the puzzle, and displays the solved puzzle alongside the original for easy comparison.

---

## 🚀 Features

- Extracts Sudoku grid from image using OpenCV.
- Recognizes digits using a custom-trained CNN model.
- Solves Sudoku using a recursive backtracking algorithm.
- Overlays solved numbers onto the original image.
- Displays both unsolved and solved puzzles side-by-side.

---

## ⚙️ Technologies Used

- **Python 3.7+**
- **OpenCV** – for image processing
- **NumPy** – for matrix operations
- **TensorFlow/Keras** – for training and using the CNN model
- **Matplotlib (Optional)** – for visualizations and debugging

---

## 💻 System Requirements

### Software
- Python 3.7 or higher
- TensorFlow, OpenCV, NumPy
- Any IDE (VS Code, PyCharm, Jupyter)

### Hardware
- Minimum 4 GB RAM
- Webcam (optional for real-time extension)
- GPU (optional for faster model training)

---
