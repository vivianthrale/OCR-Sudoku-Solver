from flask import Flask, request, jsonify, render_template_string
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import tempfile

# Import sudoku solving functions
from utils.sudoku_solver import solve_sudoku
from utils.sudoku_board_detector import get_sudoku_board_contour, extract_sudoku_board
from utils.process_sudoku_board import get_individual_cells, ocr_digits
from utils.display import display_sudoku_board, display_solved_sudoku
from tensorflow import keras

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Load the digit classifier model
MODEL_PATH = 'trained_model/digit_classifier.h5'
model = keras.models.load_model(MODEL_PATH)

HTML_TEMPLATE = '''
<!DOCTYPE html>
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
</html>
'''

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
            contour = get_sudoku_board_contour(image)
            if contour is None:
                return render_template_string(HTML_TEMPLATE, error='Could not detect Sudoku board in image')
            
            board_image = extract_sudoku_board(image, contour)
            
            # 2. Get individual cells
            cells = get_individual_cells(board_image)
            
            # 3. OCR the digits
            board = ocr_digits(cells, model)
            
            # 4. Solve the sudoku
            solved_board = solve_sudoku(board)
            
            if solved_board is None:
                return render_template_string(HTML_TEMPLATE, error='Could not solve the Sudoku puzzle. Please check if the puzzle is valid.')
            
            # Format the result
            result = display_sudoku_board(solved_board)
            
            return render_template_string(HTML_TEMPLATE, result=result)
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, error=f'An error occurred: {str(e)}')

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
