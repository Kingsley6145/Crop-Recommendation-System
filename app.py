import json
from flask import Flask, request, render_template, redirect, url_for, session, flash
import numpy as np
import pickle
import warnings

app = Flask(__name__, static_folder="./static")
app.secret_key = '1726'

def load_models():
    model_path = '/home/kingsley/crop recommender system/archive/model.pkl'
    standscaler_path = '/home/kingsley/crop recommender system/archive/standscaler.pkl'
    minmaxscaler_path = '/home/kingsley/crop recommender system/archive/minmaxscaler.pkl'

    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(standscaler_path, 'rb') as f:
            standscaler = pickle.load(f)
        with open(minmaxscaler_path, 'rb') as f:
            minmaxscaler = pickle.load(f)
        print("Models loaded successfully.")
        return model, standscaler, minmaxscaler
    except FileNotFoundError as e:
        print(f"File not found error: {e}. Check file paths and ensure files exist.")
        return None, None, None
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None, None

model, standscaler, minmaxscaler = load_models()

# Dictionary to store user credentials and roles
users = {"Kingsley Ombongi": {"password": "Kingsley6", "role": "user"}}
admins = {"Kingsley": {"password": "Kingsley6", "role": "admin"}}

def read_crops():
    try:
        with open('crops.json', 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return []

def write_crops(crops):
    with open('crops.json', 'w') as file:
        json.dump(crops, file, indent=4)

def read_removed_crops():
    try:
        with open('removed_crops.json', 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return []

def write_removed_crops(crops):
    with open('removed_crops.json', 'w') as file:
        json.dump(crops, file, indent=4)

@app.route('/')
def home():
    return redirect(url_for('login'))  # Always redirect to the login page

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = users.get(username)
        
        if user and user['password'] == password:
            session['username'] = username
            session['role'] = 'user'
            return redirect(url_for('index'))
        else:
            flash('Wrong username/password. Try again.', 'error')
    return render_template('login.html')

@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        admin = admins.get(username)
        
        if admin and admin['password'] == password:
            session['username'] = username
            session['role'] = 'admin'
            return redirect(url_for('admin_dashboard'))
        else:
            flash('Wrong username/password. Try again.', 'error')
    return render_template('admin_login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('role', None)
    return redirect(url_for('login'))

@app.route('/admin_logout')
def admin_logout():
    session.pop('username', None)
    session.pop('role', None)
    return redirect(url_for('admin_login'))

@app.route('/index')
def index():
    if 'username' not in session or session.get('role') != 'user':
        return redirect(url_for('login'))
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    if 'username' not in session or session.get('role') != 'user':
        return redirect(url_for('login'))

    N = float(request.form['Nitrogen'])
    P = float(request.form['Phosporus'])
    K = float(request.form['Potassium'])
    temp = float(request.form['Temperature'])
    humidity = float(request.form['Humidity'])
    ph = float(request.form['pH'])
    rainfall = float(request.form['Rainfall'])

    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)
    print(f"Input features: {feature_list}")

    if standscaler is None or minmaxscaler is None or model is None:
        return render_template('index.html', result="Prediction failed. Model not loaded.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            mx_features = standscaler.transform(single_pred)
            sc_mx_features = minmaxscaler.transform(mx_features)
            print(f"Scaled features: {sc_mx_features}")
        except Exception as e:
            print(f"Error transforming data: {e}")
            return render_template('index.html', result="Prediction failed. Unexpected error.")

    try:
        distances = model.predict_proba(sc_mx_features)[0]  # Example using predict_proba; adjust as necessary
        crops = read_crops()
        crop_dict = {crop['id']: crop['name'] for crop in crops}

        top_indices = np.argsort(distances)[-3:][::-1]  # Get indices of top 3 closest crops
        top_crops = [crop_dict[idx + 1] for idx in top_indices]  # Adjust indices based on your crop_dict
        
        result = f" {', '.join(top_crops)}"
    except Exception as e:
        print(f"Prediction failed. Unexpected error: {e}")
        result = "Prediction failed. Unexpected error."

    return render_template('index.html', result=result)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        new_username = request.form['new_username']
        new_password = request.form['new_password']
        confirm_password = request.form['confirm_password']
        user_role = request.form['user_role']

        if new_username in users:
            return render_template('register.html', username_exists=True)
        elif new_password != confirm_password:
            return render_template('register.html', password_mismatch=True)
        elif not user_role:
            flash('Please select a role', 'danger')
            return render_template('register.html')
        else:
            users[new_username] = {'password': new_password, 'role': user_role}
            return render_template('register.html', account_created=True)

    return render_template('register.html')


@app.route('/crops')
def crops_page():
    if 'username' not in session or session.get('role') != 'user':
        return redirect(url_for('login'))
    
    crops = read_crops()
    return render_template('crops.html', crops=crops)

@app.route('/admin_dashboard')
def admin_dashboard():
    if 'username' not in session or session.get('role') != 'admin':
        return redirect(url_for('admin_login'))
    return render_template('admin_dashboard.html')

@app.route('/manage_users', methods=['GET', 'POST'])
def manage_users():
    if 'username' not in session or session.get('role') != 'admin':
        return redirect(url_for('admin_login'))

    if request.method == 'POST':
        if 'add_user' in request.form:
            new_username = request.form['new_username']
            new_password = request.form['new_password']
            user_role = request.form['user_role']
            
            if new_username in users:
                flash('Username already exists. Please choose a different username.', 'danger')
            elif not user_role:
                flash('Please select a role.', 'danger')
            else:
                users[new_username] = {'password': new_password, 'role': user_role}
                flash('User added successfully.', 'success')
                return redirect(url_for('manage_users'))

        if 'remove_user' in request.form:
            username_to_remove = request.form['username_to_remove']
            if username_to_remove in users:
                del users[username_to_remove]
                flash('User removed successfully.', 'success')
                return redirect(url_for('manage_users'))
            else:
                flash('User not found.', 'danger')

    return render_template('manage_users.html', users=users)

@app.route('/manage_crops', methods=['GET', 'POST'])
def manage_crops():
    if 'username' not in session or session.get('role') != 'admin':
        return redirect(url_for('admin_login'))

    crops = read_crops()
    removed_crops = read_removed_crops()

    if request.method == 'POST':
        if 'add_crop' in request.form:
            crop_name = request.form.get('crop_name')
            nitrogen = request.form.get('nitrogen')
            phosphorus = request.form.get('phosphorus')
            potassium = request.form.get('potassium')
            temperature = request.form.get('temperature')
            humidity = request.form.get('humidity')
            ph = request.form.get('ph')
            rainfall = request.form.get('rainfall')

            print(f"crop_name: {crop_name}, nitrogen: {nitrogen}, phosphorus: {phosphorus}, potassium: {potassium}, temperature: {temperature}, humidity: {humidity}, ph: {ph}, rainfall: {rainfall}")

            # Check for any None values
            if not all([crop_name, nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]):
                flash('Please fill in all the fields.', 'danger')
            else:
                try:
                    new_id = max(crop['id'] for crop in crops) + 1 if crops else 1

                    new_crop = {
                        "id": new_id,
                        "name": crop_name,
                        "nitrogen": int(nitrogen),
                        "phosphorus": int(phosphorus),
                        "potassium": int(potassium),
                        "temperature": float(temperature),
                        "humidity": float(humidity),
                        "ph": float(ph),
                        "rainfall": float(rainfall)
                    }
                except ValueError as e:
                    flash(f"Error converting form values: {e}", 'danger')
                    return redirect(url_for('manage_crops'))

                crops.append(new_crop)
                write_crops(crops)
                flash('Crop added successfully.', 'success')
                return redirect(url_for('manage_crops'))

        if 'remove_crop' in request.form:
            crop_id_to_remove = request.form.get('crop_id')
            print(f"crop_id_to_remove: {crop_id_to_remove}")

            if not crop_id_to_remove:
                flash('No crop ID provided.', 'danger')
            else:
                try:
                    crop_id_to_remove = int(crop_id_to_remove)
                except ValueError as e:
                    flash(f"Invalid crop ID: {e}", 'danger')
                    return redirect(url_for('manage_crops'))

                crop_to_remove = next((crop for crop in crops if crop['id'] == crop_id_to_remove), None)
                if crop_to_remove:
                    crops = [crop for crop in crops if crop['id'] != crop_id_to_remove]
                    removed_crops.append(crop_to_remove)
                    write_crops(crops)
                    write_removed_crops(removed_crops)
                    flash('Crop removed successfully.', 'success')
                    return redirect(url_for('manage_crops'))
                else:
                    flash('Crop not found.', 'danger')

        if 'delete_crop' in request.form:
            crop_id_to_delete = request.form.get('crop_id')
            print(f"crop_id_to_delete: {crop_id_to_delete}")

            if not crop_id_to_delete:
                flash('No crop ID provided.', 'danger')
            else:
                try:
                    crop_id_to_delete = int(crop_id_to_delete)
                except ValueError as e:
                    flash(f"Invalid crop ID: {e}", 'danger')
                    return redirect(url_for('manage_crops'))

                crop_to_delete = next((crop for crop in removed_crops if crop['id'] == crop_id_to_delete), None)
                if crop_to_delete:
                    removed_crops = [crop for crop in removed_crops if crop['id'] != crop_id_to_delete]
                    write_removed_crops(removed_crops)
                    flash('Crop deleted permanently.', 'success')
                    return redirect(url_for('manage_crops'))
                else:
                    flash('Crop not found.', 'danger')

    return render_template('manage_crops.html', crops=crops, removed_crops=removed_crops)

@app.route('/kiambu')
def kiambu():
    return render_template('kiambu.html')

if __name__ == '__main__':
    app.run(debug=True)
