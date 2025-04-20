from flask import Flask, render_template, request, jsonify, send_from_directory, url_for, redirect, flash
from werkzeug.utils import secure_filename
import os
from flask_sqlalchemy import SQLAlchemy
from model import predict_disease  # Ensure this is correctly implemented
from flask_cors import CORS

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
app = Flask(__name__)
CORS(app)

# Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///agri_cure.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = "your_secret_key"  # Needed for flash messages
db = SQLAlchemy(app)

# Folder Setup
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Define Database Models
class Contact(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    message = db.Column(db.Text, nullable=False)

class ImageUpload(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    disease_name = db.Column(db.String(100), nullable=True)
    symptoms = db.Column(db.Text, nullable=True)
    causes = db.Column(db.Text, nullable=True)
    cure = db.Column(db.Text, nullable=True)

# Create Database Tables
with app.app_context():
    db.create_all()

# Function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home Page
@app.route('/')
def index():
    return render_template('index.html')

# Contact Page (Store Messages)
@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')

        print(f"Received: {name}, {email}, {message}")  # Debugging output

        if name and email and message:
            # Delete all existing records before adding a new one
            Contact.query.delete()
            db.session.commit()

            # Add the new record
            new_contact = Contact(name=name, email=email, message=message)
            db.session.add(new_contact)
            db.session.commit()  # Commit after adding

            print("All previous contacts deleted, new contact saved!")  # Debugging output

            flash("Thank you for contacting us!", "success")

    return render_template('contact.html')


# Display Stored Contact Messages
@app.route('/result', methods=['GET', 'POST'])
def stored_messages():
    if request.method == 'POST':  # If form is submitted
        admin_token = request.form.get('admin_token')  # Get token from input
        if admin_token == "Agri_Admin@123":
            contacts = Contact.query.all()
            return render_template('result.html', contacts=contacts)
        else:
            flash("Access Denied! Incorrect Token", "danger")
            return redirect(url_for('stored_messages'))

    # If accessed via GET, show the login form
    return render_template('enter_token.html')

# Organic Page

@app.route('/organic')
def organic():
    return render_template('organic.html')

# Upload Page
@app.route('/upload')
def upload_page():
    return render_template('upload.html')

# Image Upload and Disease Prediction
@app.route('/upload-image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Predict disease
        result = predict_disease(filepath)

        if len(result) == 8:
            disease_name, symptoms, causes, cure, fungicide, fungicide_url, image_url, confidence = result
        else:
            return jsonify({'error': 'Unexpected return value from predict_disease()'})

        # Store in database
        new_upload = ImageUpload(
            filename=filename,
            disease_name=disease_name,
            symptoms=symptoms,
            causes=causes,
            cure=cure
        )
        db.session.add(new_upload)
        db.session.commit()

        response_data = {
            "image": url_for('uploaded_file', filename=filename),
            "Name": disease_name,
            "Symptoms": symptoms,
            "Causes": causes,
            "Cure": cure,
            "Fungicide": fungicide,
            "Fungicide_URL": fungicide_url,
            "Supplement_Image": url_for('static', filename=f'images/{image_url}'),
            "Confidence": f"{confidence:.2%}"
        }

        return jsonify(response_data)

    return jsonify({'error': 'Invalid file type'})


# Serve Uploaded Files
@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# View Uploaded Images and Predictions
@app.route('/view-uploads')
def view_uploads():
    uploads = ImageUpload.query.all()
    return render_template('uploads.html', uploads=uploads)

if __name__ == '__main__':
    app.run(debug=True)
