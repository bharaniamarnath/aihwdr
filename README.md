Project:
PTID-CDS-DEC-25-3624_PRCP-1002-HandwrittenDigits

Configuration:

Environment: Anaconda3 Virtual Environment
Python: v3.7.16
Jupyter: 6.5.7
Tensorflow: v2.11.0
Gradio: v3.7

Files:

PTID-CDS-DEC-25-3624_PRCP-1002-HandwrittenDigits.ipynb:
*Jupyter notebook file containing project source code.
*Code includes 3 tasks of project:
1.Data Analysis - Exploration, Visualization, Distribution, Correlation.
2.Build Models - SVM, Random Forest Classifer, Keras Sequential with prediction.
3.Compare models - Evaluate models' accuracy, classification report, best classifier.

PTID-CDS-DEC-25-3624_PRCP-1002-HandwrittenDigits.py:
*Python file containing project source code.
*Data Analysis - Data Distribution, Prediction - SVM, Random Forest Classifer, Keras Sequential models.
*Gradio v3.7 User Interface implemented for data analysis plot, image upload and display models prediction result.

Modules:
main.py - Module with main() function
process.py - Load dataset, process data
model.py - Build models SVM, Random Forest, Keras Sequential
evaluate.py - Evaluate models accuracy, classification report
output.py - Set of functions that returns output for data analysis, visualization, correlation and prediction
webui.py - Build web user interface using Gradio and function from output.py module

Execution:

Main project module:
*Required: Python v3.7.16.
*Run command: In terminal, while in the project root directory, run the command "run.bat".
*Open the generated URL (FastAPI Server default IP: http://127.0.0.1:7860/) displayed in the prompt using a compatible web browser to view the project user interface.
*Upload a sample handwritten digit image under the "Prediction" tab. (Find sample images in project subfolder - datasets/images/handwritten_numbers/).

PTID-CDS-DEC-25-3624_PRCP-1002-HandwrittenDigits.ipynb:
*Open and run the ipynb source file in compatible Jupyter Notebook.

PTID-CDS-DEC-25-3624_PRCP-1002-HandwrittenDigits.py:
*Single source file containing complete project code
*Open terminal in project root, and run command "python PTID-CDS-DEC-25-3624_PRCP-1002-HandwrittenDigits.py"
