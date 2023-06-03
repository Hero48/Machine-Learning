from flask import Flask, render_template, request
import subprocess
import os
import sys


app = Flask(__name__)
@app.route('/')
def home():
    return render_template('terminal.html')

@app.route('/execute', methods=['POST'])
def execute():
    command = request.form['command']
    output = subprocess.check_output(command, shell=True)
    return output

if __name__ == '__main__':
    app.run(debug=True)