import logging
import os
import numpy as np
import torch
from logging.handlers import RotatingFileHandler
from flask import Flask, render_template, request, jsonify

from model import CNN


# Set up the Flask app
app = Flask(__name__)

# Configure logging
if not app.debug:
    if not os.path.exists("logs"):
        os.mkdir("logs")
    file_handler = RotatingFileHandler(
        "logs/app.log",
        maxBytes=10240,
        backupCount=10,
    )
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]"
        )
    )
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info("Digit classifier startup")

TORCH_DEVICE = torch.device("cpu")

# Load model
MODEL = CNN().to(TORCH_DEVICE)
MODEL.load_state_dict(
    torch.load(
        "cnn_weights.pt",
        map_location=TORCH_DEVICE,
        weights_only=True,
    )
)


# Default route
@app.route("/")
def index():
    return render_template("index.html")


# Process the drawing and classify the digit
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    grid_data = data["grid"]

    # Convert the grid data to a numpy array
    grid_array = np.array(grid_data, dtype=np.float32) / 1.0

    data = torch.reshape(torch.tensor(grid_array), (1, 1, 28, 28))

    # Here you would pass the grid data to your model
    # As requested, model code is removed and would be in a separate file
    # This is a placeholder for demonstration
    output = MODEL(data)
    prediction = output.argmax(dim=1, keepdim=True)
    confidence, _ = torch.max(torch.nn.functional.softmax(output, dim=1), 1)

    # Return results
    return jsonify(
        {
            "prediction": prediction.item(),
            "confidence": f"{confidence.item():.2f}%",
            "status": "Success.",
        }
    )


# HTML template for the web interface
@app.route("/templates/index.html")
def get_template():
    return render_template("index.html")


# Create HTML template
def create_template():
    if not os.path.exists("templates"):
        os.makedirs("templates")

    with open("templates/index.html", "w") as f:
        f.write(
            """<!DOCTYPE html>
<html>
<head>
    <title>Digit Classifier</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            text-align: center;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
        }
        .drawing-container {
            display: flex;
            justify-content: center;
            align-items: center;
            margin: 20px auto;
        }
        .grid-container {
            display: grid;
            grid-template-columns: repeat(28, 10px);
            grid-template-rows: repeat(28, 10px);
            gap: 1px;
            background-color: #333;
            padding: 1px;
            border: 2px solid #333;
        }
        .grid-cell {
            width: 10px;
            height: 10px;
            background-color: black;
            cursor: pointer;
        }
        .active {
            background-color: white;
        }
        .button-container {
            margin: 20px 0;
        }
        button {
            padding: 10px 20px;
            margin: 0 10px;
            font-size: 16px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        #predict-btn {
            background-color: #4CAF50;
            color: white;
        }
        #clear-btn {
            background-color: #f44336;
            color: white;
        }
        button:hover {
            opacity: 0.8;
        }
        .result {
            margin: 20px 0;
            padding: 20px;
            border-radius: 5px;
            background-color: #e9f7ef;
            font-size: 18px;
        }
        .digit {
            font-size: 48px;
            font-weight: bold;
            margin: 10px 0;
        }
        .confidence {
            color: #666;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Handwritten Digit Classifier</h1>
        <p>Click or drag to activate cells in the 28×28 grid below, then click "Predict"</p>

        <div class="drawing-container">
            <div id="grid-container" class="grid-container"></div>
        </div>

        <div class="button-container">
            <button id="predict-btn">Predict</button>
            <button id="clear-btn">Clear</button>
        </div>

        <div class="result" id="result-container" style="display: none;">
            <h2>Result:</h2>
            <div class="digit" id="prediction">?</div>
            <div class="confidence" id="confidence"></div>
            <div id="status" style="font-size: 12px; color: #999;"></div>
        </div>
    </div>

    <script>
        // Grid dimensions
        const GRID_SIZE = 28;

        // Create the grid
        const gridContainer = document.getElementById('grid-container');
        const grid = [];

        function createGrid() {
            gridContainer.innerHTML = '';

            // Create a 2D array to represent the grid
            for (let i = 0; i < GRID_SIZE; i++) {
                grid[i] = [];
                for (let j = 0; j < GRID_SIZE; j++) {
                    const cell = document.createElement('div');
                    cell.className = 'grid-cell';
                    cell.dataset.row = i;
                    cell.dataset.col = j;

                    // Add event listener for individual click
                    cell.addEventListener('mousedown', function() {
                        toggleCell(this);
                    });

                    // Add event listener for click and drag
                    cell.addEventListener('mouseenter', function(e) {
                        if (e.buttons === 1) {
                            toggleCell(this);
                        }
                    });

                    gridContainer.appendChild(cell);
                    grid[i][j] = 0;  // Initialize as inactive
                }
            }

            // Prevent default drag behavior
            gridContainer.addEventListener('dragstart', function(e) {
                e.preventDefault();
            });
        }

        // Function to toggle cell state
        function toggleCell(cell) {
            const row = parseInt(cell.dataset.row);
            const col = parseInt(cell.dataset.col);

            if (!cell.classList.contains('active')) {
                cell.classList.add('active');
                grid[row][col] = 1;  // Set as active
            }
        }

        // Touch support for mobile devices
        gridContainer.addEventListener('touchstart', function(e) {
            e.preventDefault();
            const touch = e.touches[0];
            const element = document.elementFromPoint(touch.clientX, touch.clientY);
            if (element && element.classList.contains('grid-cell')) {
                toggleCell(element);
            }
        });

        gridContainer.addEventListener('touchmove', function(e) {
            e.preventDefault();
            const touch = e.touches[0];
            const element = document.elementFromPoint(touch.clientX, touch.clientY);
            if (element && element.classList.contains('grid-cell')) {
                toggleCell(element);
            }
        });

        // Clear grid
        function clearGrid() {
            const cells = document.querySelectorAll('.grid-cell');
            cells.forEach(cell => {
                cell.classList.remove('active');
                const row = parseInt(cell.dataset.row);
                const col = parseInt(cell.dataset.col);
                grid[row][col] = 0;
            });
            document.getElementById('result-container').style.display = 'none';
        }

        document.getElementById('clear-btn').addEventListener('click', clearGrid);

        // Predict function
        function predict() {
            // Show loading state
            document.getElementById('prediction').textContent = 'Processing...';
            document.getElementById('confidence').textContent = '';
            document.getElementById('status').textContent = '';
            document.getElementById('result-container').style.display = 'block';

            // Send to server for prediction
            fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ grid: grid })
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('prediction').textContent = data.prediction;
                document.getElementById('confidence').textContent = `Confidence: ${data.confidence}`;
                document.getElementById('status').textContent = data.status;
            })
            .catch(error => {
                console.error('Error:', error);
                document.getElementById('prediction').textContent = 'Error';
                document.getElementById('confidence').textContent = 'Failed to get prediction';
            });
        }

        document.getElementById('predict-btn').addEventListener('click', predict);

        // Initialize the grid
        createGrid();
    </script>
</body>
</html>"""
        )


if __name__ == "__main__":
    # Create template file
    create_template()
    # Run the Flask application
    app.run(debug=False)
