from module import transform
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, send_file
import os
import sys
sys.path.append("fyp-morph-api/")
app = Flask(__name__, static_url_path='/static')
os.makedirs(os.path.join(app.instance_path, 'uploads'), exist_ok=True)


@app.route("/", methods=['POST'])
def startProcess():
    try:
        src = request.files['source']
        trg = request.files['target']
        params = []
        params.append(request.args.get("x"))
        params.append(request.args.get("y"))
        params.append(request.args.get("z"))

        src.save(os.path.join(app.instance_path,
                              'uploads', secure_filename(src.filename)))
        trg.save(os.path.join(app.instance_path,
                              'uploads', secure_filename(trg.filename)))

        morph = transform.ManipulateSelfie(src.filename, trg.filename, params)
        timage = morph.apply_transformation()
        return send_file(timage, mimetype='image/jpg')
    except:
        return ("Some error occurred while trying to process")


@app.route("/", methods=['GET'])
def index():
    try:
        return render_template('index.html')
    except:
        return ("Some error occurred while trying to fetch data")


@app.route("/examples", methods=['GET'])
def getExamples():
    try:
        return render_template('examples.html')
    except:
        return ("Some error occurred while trying to fetch data")


if __name__ == "__main__":
    app.run(debug=True)
