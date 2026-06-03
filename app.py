import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

# Attempt to load pyhailing environment from your local package setup
try:
    from pyhailing import RidehailEnv
except ImportError:
    raise ImportError("Please run 'pip install -e .' within your DriveMate repository to use the pyhailing environment module.")

app = Flask(__name__)
CORS(app)  # Enables cross-origin sharing so your HTML file can call it smoothly

# Instantiate the OpenAI Gym Environment matching your project architecture
env = RidehailEnv()
observation = env.reset()

def parse_environment_state(obs):
    """
    Transforms multi-dimensional numpy arrays from the observation space 
    into standard dictionary objects safe for JSON transfer.
    """
    req_locs = obs.get('request_locs', np.array([]))
    num_reqs = req_locs.shape[0] if req_locs.ndim > 0 else 0
    
    parsed_requests = []
    for i in range(num_reqs):
        parsed_requests.append({
            "origin": {"x": float(req_locs[i][0][0]), "y": float(req_locs[i][0][1])},
            "destination": {"x": float(req_locs[i][1][0]), "y": float(req_locs[i][1][1])}
        })

    v_locs = obs.get('v_locs', np.array([]))
    
    return {
        "time": float(obs.get('time', [0])[0]) if isinstance(obs.get('time'), (list, np.ndarray)) else float(obs.get('time', 0)),
        "day_of_week": int(obs.get('dow', 0)),
        "num_pending_requests": num_reqs,
        "num_vehicles": v_locs.shape[0] if v_locs.ndim > 0 else 0,
        "requests": parsed_requests
    }

@app.route('/api/state', methods=['GET'])
def get_state():
    """Returns the current state profile of the simulator fleet."""
    global observation
    return jsonify(parse_environment_state(observation))

@app.route('/api/step', methods=['POST'])
def step_environment():
    """Progresses the gym environment using a standard random sampling step."""
    global observation
    # Fetch random structural action space dictionary from environment layer
    action = env.get_random_action()
    observation, reward, done, info = env.step(action)
    
    if done:
        observation = env.reset()
        
    return jsonify(parse_environment_state(observation))

@app.route('/api/assign', methods=['POST'])
def assign_dispatch():
    """Manual vehicle override assignment endpoint from dashboard UI actions."""
    global observation
    data = request.json or {}
    req_idx = data.get('request_index', 0)
    vehicle_idx = data.get('vehicle_index', 0)
    
    # Construct base valid empty action dict structures matching environment specs
    action = env.get_random_action()
    
    # Override assignment fields safely matching current dimensions
    if len(action['req_assgts']) > req_idx:
        action['req_assgts'][req_idx] = vehicle_idx
        
    observation, reward, done, info = env.step(action)
    if done:
        observation = env.reset()
        
    return jsonify({"status": "success", "msg": f"Dispatched vehicle {vehicle_idx} to request {req_idx}."})

if __name__ == '__main__':
    print("DriveMate core Engine booting up...")
    app.run(host='127.0.0.1', port=5000, debug=True)
