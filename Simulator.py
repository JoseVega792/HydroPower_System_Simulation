from flask import Flask, render_template, session, request, jsonify
from Reservoir import Reservoir
import math

app = Flask(__name__)
app.secret_key = "something_random"


# Initial values for reservoir parameters
length = 5
width = 10
height = 5
currHeight = 0

# Inital values for controlled variables
temperature = 0
release = 0
inflow = 0

# Creating reservoir
reservoir = Reservoir(length,height,width,currHeight)

@app.route('/toggle-dark-mode', methods=['POST'])
def toggle_dark_mode():
    session['dark_mode'] = not session.get('dark_mode')
    return 'Dark mode toggled successfully'

@app.route('/get_energy_output', methods=['GET'])
def get_energy_output():
    # Get energy output from the reservoir
    energy_output = reservoir.energy_output()
    return jsonify({'energy_output': energy_output})

@app.route('/', methods = ['GET','POST'])
def index():
    water_level = calculate_water_level(reservoir.current_volume,reservoir.max_volume)
    dark_mode = session.get('dark_mode')
    return render_template('index.html', temperature=temperature, release=release, inflow=inflow, water_level=water_level, dark_mode = dark_mode)

@app.route('/update', methods=['POST'])
def update():
    global temperature, release, inflow
    if request.method == 'POST':
        temperature = int(request.form['temperature'])
        release = int(request.form['release'])
        inflow = int(request.form['inflow'])
        reservoir.inflow(inflow)
        reservoir.release(release)
        water_level = calculate_water_level(reservoir.current_volume,reservoir.max_volume)
        return jsonify({'water_level': water_level})

def calculate_water_level(current_volume,max_volume):
    water_level = math.floor((current_volume / max_volume) * 100)
    water_level = max(0, min(100, water_level))
    print(water_level, max_volume, current_volume)
    return water_level

if __name__ == '__main__':
    app.run(debug=True)