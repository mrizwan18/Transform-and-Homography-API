import os

from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename

from module import transform

app = Flask(__name__, static_url_path='/static')
os.makedirs(os.path.join(app.instance_path, 'uploads'), exist_ok=True)


@app.route("/", methods=['POST'])
def startProcess():
    try:
        src = request.files['source']
        trg = request.files['target']
    except:
        return "Either source or target is empty"
    params = [request.args.get("x"), request.args.get("y"), request.args.get("z")]
    plength = sum(1 for x in params if x is not None)
    if src is None or trg is None or plength < 3:
        return "Either source or target is empty or some parameter is empty"

    src.save(os.path.join(app.instance_path,
                          'uploads', secure_filename(src.filename)))
    trg.save(os.path.join(app.instance_path,
                          'uploads', secure_filename(trg.filename)))
    try:
        morph = transform.ManipulateSelfie(src.filename, trg.filename, params)
        response = morph.apply_transformation()
        return send_file(response, mimetype='image/jpg')
    except:
        return "Some error occurred while processing", 400


@app.route("/", methods=['GET'])
def index():
    try:
        return render_template('index.html')
    except:
        return "Some error occurred while trying to fetch data"


@app.route("/examples", methods=['GET'])
def getExamples():
    try:
        return render_template('examples.html')
    except:
        return "Some error occurred while trying to fetch data"


if __name__ == "__main__":
    app.run(debug=True)
