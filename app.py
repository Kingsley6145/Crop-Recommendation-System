from flask import Flask, request, render_template, redirect, url_for, session, flash
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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

@app.route('/')
def home():
    if 'username' in session:
        if session.get('role') == 'admin':
            return redirect(url_for('admin_dashboard'))
        else:
            return redirect(url_for('index'))
    return redirect(url_for('login'))

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
        prediction = model.predict(sc_mx_features)
        print(f"Prediction result: {prediction}")

        crop_dict = {
            1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
            8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
            14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
            19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
        }

        if prediction[0] in crop_dict:
            crop = crop_dict[prediction[0]]
            result = f"{crop} is the best crop to be cultivated."
        else:
            result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
    except Exception as e:
        print(f"Prediction failed. Unexpected error: {e}")
        result = "Prediction failed. Unexpected error."

    return render_template('index.html', result=result)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        new_username = request.form['new_username']
        new_password = request.form['new_password']
        if new_username in users:
            flash('Username already exists. Please choose a different username.')
        else:
            users[new_username] = {'password': new_password, 'role': 'user'}
            return render_template('register.html', account_created=True)
    return render_template('register.html')

@app.route('/crops')
def crops():
    if 'username' not in session or session.get('role') != 'user':
        return redirect(url_for('login'))
    return render_template('crops.html')

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
            if new_username in users:
                flash('Username already exists. Please choose a different username.', 'danger')
            else:
                users[new_username] = {'password': new_password, 'role': 'user'}
                flash('User added successfully.', 'success')
        elif 'remove_user' in request.form:
            username_to_remove = request.form['username_to_remove']
            if username_to_remove in users:
                del users[username_to_remove]
                flash('User removed successfully.', 'success')
            else:
                flash('Username not found.', 'danger')

    return render_template('manage_users.html')

if __name__ == "__main__":
    app.run(debug=True)
