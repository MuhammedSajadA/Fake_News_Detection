from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/')
def form():
    return render_template('home.html')


@app.route('/data/', methods=['POST', 'GET'])
def data():
    if request.method == 'GET':
        return f"The URL /data is accessed directly. Try going to '/form' to submit form"
    if request.method == 'POST':
        form_data = request.form
        return render_template('home.html')


if __name__ == "__main__":
    app.run(debug=True)