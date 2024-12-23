## Student Performance Prediction Using Linear Regression

This project builds a ML model to predict students' final grades (`G3`) based on various factors such as demographics, lifestyle habits, and academic performance. The dataset is preprocessed, and a linear regression model is trained to make predictions. 
For my own learning, I have included model persistence using `pickle` and visualizations of the results.

---

### Features
1. **Dataset Preprocessing**:
   - Encodes categorical variables like `school`, `sex`, and `internet` using `LabelEncoder`.
   - Selects relevant features for training, including grades (`G1`, `G2`) and lifestyle attributes (`absences`, `studytime`).

2. **Model Training**:
   - Trains a linear regression model multiple times and saves the best-performing model based on accuracy.

3. **Model Persistence**:
   - Saves the trained model to a file (`studentmodel.pickle`) for reuse.

4. **Visualizations**:
   - Scatter plots showing the impact of individual variables (e.g., `absences`, `studytime`) on final grades.
   - A performance plot comparing predicted grades (`G3`) with actual grades, highlighting model accuracy.

---

### How to Use
1. **Dependencies**:
   Install the required libraries using:
   ```bash
   pip install pandas numpy scikit-learn matplotlib
   ```

2. **Run the Script**:
   Ensure the dataset (`student-mat.csv`) is in the correct location and execute the script.

3. **Results**:
   - The best-performing model will be saved as `studentmodel.pickle`.
   - Visualizations will be displayed to analyze the model's performance and variable relationships.

4. **Predicted vs. Actual Values**:
   A DataFrame is printed to compare predicted grades with actual grades for further evaluation.

---

### Key Visualizations
- **Impact of Variables on Grades**: Scatter plots for key variables like `absences` and `studytime` against `G3`.
- **Model Accuracy**: A scatter plot comparing predicted vs. true values with an ideal line for reference.

---

### Future Improvements
- Add additional models for comparison (e.g., Random Forest or Gradient Boosting).
- Include more advanced feature engineering and hyperparameter tuning
