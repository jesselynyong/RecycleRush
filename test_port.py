from flask import Flask, render_template, request
import serial
import time

app = Flask(__name__)

@app.route('/')
def index():
    # Serve the HTML page
    return render_template('index.html')

@app.route('/control_led', methods=['POST'])
def control_led():
    led_number = request.form.get('led')
    print(led_number)
    try:
        # Dynamically manage the serial connection within the handler
        with serial.Serial('COM4', 9600, timeout=1) as arduino:
            time.sleep(2) # Wait for the connection to establish
            # Send the LED number to the Arduino, including a newline character
            arduino.write((led_number + '\n').encode())
            # Optionally, wait for a response here with arduino.readline()
            return ('', 204)  # Return an empty response to signify success
    except serial.SerialException as e:
        print(f"Failed to communicate with Arduino: {e}")
        return "Failed to communicate with Arduino", 500

if __name__ == '__main__':
    app.run(debug=True)
