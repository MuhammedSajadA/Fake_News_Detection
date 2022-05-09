from flask import Flask, render_template
import main

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def man():
    main()


if __name__ == "__main__":
    app.run(debug=True)
