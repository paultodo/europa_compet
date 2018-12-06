from flask import Flask
from flask import request, jsonify


app = Flask("dummy-server")


@app.route('/alive')
def alive():
    return 'alive'


@app.route('/prediction', methods=['POST'])
def prediction():
    return jsonify({'status': 'prediction running'})


@app.route('/collect')
def collect():
    return 'finished'


@app.route('/shutdown')
def shutdown():
    "Shutdown server"
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        return jsonify({'status': 'error - Not running with the Werkzeug Server'})
        # raise RuntimeError('Not running with the Werkzeug Server')
    else:
        func()
        return jsonify({'status': 'shutting down'})


if __name__ == '__main__':
    app.run(port=4130)
