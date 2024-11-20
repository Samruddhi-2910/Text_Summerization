# Importing the libraries
from flask import Flask, render_template, request
from model import summarize_with_t5

# Creating an instance of the Flask class
app = Flask(__name__)

# Defining the routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    if request.method == 'POST':
        user_input = request.form['user_input']
        selected_language = request.form['language'] 
        summary = summarize_with_t5(user_input, language=selected_language)
        return render_template('result.html', user_input=user_input, summary=summary, selected_language_name=selected_language)

# Running the app
if __name__ == '__main__':
    app.run(debug=True)
