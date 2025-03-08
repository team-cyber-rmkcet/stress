# Stress Level Prediction

This project is a web application that predicts stress levels based on various input factors. It uses a machine learning model to make predictions and provides a user-friendly interface for input and result display.

## Features

- User input form for various stress-related factors
- Machine learning model (Decision Tree Classifier) for stress level prediction
- Responsive web design with custom styling
- Input validation to ensure data integrity
- Error handling for invalid inputs

## Technologies Used

- Python
- Flask
- scikit-learn
- pandas
- numpy
- HTML/CSS
- JavaScript

## Project Structure

- `app.py`: Main Flask application file containing the server-side logic and machine learning model
- `templates/`: Directory containing HTML templates
  - `login.html`: Input form for user data
  - `result.html`: Displays the predicted stress level
  - `error.html`: Error page for invalid inputs
- `static/`: Directory for static files
  - `styles.css`: Custom CSS styles for the application
- `StressLevelDataset.csv`: Dataset used for training the model (not included in the repository)

## Setup and Running the Application

1. Clone the repository:
   ```sh
   https://github.com/venky-1710/stress-level-predection.git
   ```
2. Install the required dependencies:
   ```sh
   pip install flask pandas numpy scikit-learn
   ```
3. Ensure you have the `StressLevelDataset.csv` file in the project root directory.

4. Run the Flask application:
   ```sh
   python app.py
   ```
5. Open a web browser and navigate to `http://localhost:5000` to use the application.

## How to Use

1. Fill in the form with your stress-related factors. Each field has a specified range of values.
2. Click the "Submit" button to get your predicted stress level.
3. The result page will display your predicted stress level based on the input factors.

## Future Improvements

- Implement user authentication and data storage
- Add more detailed explanations for each input factor
- Incorporate additional machine learning models for comparison
- Develop a feature to track stress levels over time

## Contributing

Contributions to improve the project are welcome. Please feel free to fork the repository and submit pull requests.

## License

This project is open source and available under the [MIT License](LICENSE).
