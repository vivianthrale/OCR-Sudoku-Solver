from flask import Flask, request, jsonify, render_template_string
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import tempfile

# Import sudoku solving functions
from utils.image_processor import locate_puzzle, extract_digit
from utils.sudoku import Sudoku
from keras.preprocessing.image import img_to_array
from keras.models import load_model

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Load the digit classifier model
MODEL_PATH = 'trained_model/digit_classifier.h5'
model = load_model(MODEL_PATH)

HTML_TEMPLATE = '''<!DOCTYPE html>
<html>
<head>
    <title>OCR Sudoku Solver</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .upload-form {
            margin: 30px 0;
        }
        input[type="file"] {
            display: block;
            margin: 20px auto;
        }
        button {
            background-color: #4CAF50;
            color: white;
            padding: 15px 32px;
            text-align: center;
            text-decoration: none;
            display: block;
            font-size: 16px;
            margin: 20px auto;
            cursor: pointer;
            border: none;
            border-radius: 5px;
        }
        button:hover {
            background-color: #45a049;
        }
        .result {
            margin-top: 30px;
            padding: 20px;
            background-color: #f9f9f9;
            border-radius: 5px;
        }
        pre {
            background-color: #272822;
            color: #f8f8f2;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }
        .error {
            color: #d32f2f;
            padding: 10px;
            background-color: #ffebee;
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔢 OCR Sudoku Solver</h1>
        <p style="text-align: center;">Upload an image of a Sudoku puzzle to solve it automatically!</p>
        
        <form class="upload-form" method="POST" action="/solve" enctype="multipart/form-data">
            <input type="file" name="image" accept="image/*" required>
            <button type="submit">Solve Sudoku</button>
        </form>
        
        {% if result %}
        <div class="result">
            <h2>✅ Solved!</h2>
            <pre>{{ result }}</pre>
        </div>
        {% endif %}
        
        {% if error %}
        <div class="error">
            <strong>Error:</strong> {{ error }}
        </div>
        {% endif %}
    </div>
</body>
</html>'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/solve', methods=['POST'])
def solve():
    try:
        # Check if file was uploaded
        if 'image' not in request.files:
            return render_template_string(HTML_TEMPLATE, error='No image file provided')
        
        file = request.files['image']
        if file.filename == '':
            return render_template_string(HTML_TEMPLATE, error='No file selected')
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            file.save(temp_file.name)
            temp_path = temp_file.name
        
        try:
            # Read the image
            image = cv2.imread(temp_path)
            if image is None:
                return render_template_string(HTML_TEMPLATE, error='Invalid image file')
            
            # Process the sudoku image
            # 1. Detect and extract sudoku board
            puzzleImage, warped = locate_puzzle(image, debug=False)
            
            # 2. Initialize 9x9 board
            board = np.zeros((9, 9), dtype='int')
            
            # 3. Extract digits from each cell
            stepX = warped.shape[1] // 9
            stepY = warped.shape[0] // 9
            
            for y in range(9):
                for x in range(9):
                    startX = x * stepX
                    startY = y * stepY
                    endX = (x + 1) * stepX
                    endY = (y + 1) * stepY
                    
                    cell = warped[startY:endY, startX:endX]
                    digit = extract_digit(cell, debug=False)
                    
                    if digit is not None:
                        # Resize and prepare for classification
                        roi = cv2.resize(digit, (32, 32))
                        roi = roi.astype('float') / 255.0
                        roi = img_to_array(roi)
                        roi = np.expand_dims(roi, axis=0)
                        
                        # Classify the digit
                        pred = model.predict(roi, verbose=0).argmax(axis=1)[0]
                        board[y, x] = pred
            
            # 4. Solve the sudoku
            puzzle = Sudoku(board.tolist(), 9, 9)
            solved = puzzle.solve()
            
            if not solved:
                return render_template_string(HTML_TEMPLATE, error='Could not solve the Sudoku puzzle. Please check if the puzzle is valid.')
            
            # Format the result as a string
            result = format_board(puzzle.board)
            
            return render_template_string(HTML_TEMPLATE, result=result)
        
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, error=f'An error occurred: {str(e)}')

def format_board(board):
    """Format the sudoku board for display"""
    result = "+-----------------------+\n"
    for i in range(9):
        if i % 3 == 0 and i != 0:
            result += "+-----------------------+\n"
        for j in range(9):
            if j % 3 == 0:
                result += "| "
            if j == 8:
                result += f"{board[i][j]} |\n"
            else:
                result += f"{board[i][j]} "
    result += "+-----------------------+"
    return result

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
