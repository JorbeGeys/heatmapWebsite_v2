from flask import Response
import time

@app.route('/progress')
def progress_stream():
    def generate():
        for i in range(101):  # Simulating 0% to 100%
            yield f"data: {i}\n\n"
            time.sleep(0.1)  # Simulate work
    return Response(generate(), mimetype='text/event-stream')
