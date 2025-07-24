from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
import bcrypt
from deepface import DeepFace
import numpy as np
import os
from dotenv import load_dotenv
import cv2
from werkzeug.security import generate_password_hash, check_password_hash
import face_recognition
from PIL import Image
import io
import sqlite3

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///attendance.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'your-secret-key')  # Change in production
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=1)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['DATABASE'] = 'attendance.db'

# Initialize extensions
db = SQLAlchemy(app)
jwt = JWTManager(app)

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def get_db():
    db = sqlite3.connect(app.config['DATABASE'])
    db.row_factory = sqlite3.Row
    return db

def init_db():
    with app.app_context():
        db = get_db()
        db.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                face_encoding TEXT
            )
        ''')
        db.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                type TEXT NOT NULL,
                status TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        db.commit()

# Root route
@app.route('/')
def home():
    return render_template('landing.html')

@app.route('/register')
def register_page():
    return render_template('register.html')

@app.route('/attendance')
def attendance_page():
    return render_template('attendance.html')

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    role = db.Column(db.String(20), nullable=False, default='employee')  # 'employee' or 'admin'
    face_embedding = db.Column(db.JSON)  # Store face encoding as JSON
    enrollment_status = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    type = db.Column(db.String(20), nullable=False)  # 'clock_in' or 'clock_out'
    status = db.Column(db.String(20), nullable=False)  # 'on_time' or 'late'
    
    user = db.relationship('User', backref=db.backref('attendances', lazy=True))

# Helper Functions
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def verify_password(password, password_hash):
    return bcrypt.checkpw(password.encode('utf-8'), password_hash)

def get_face_encoding(image_data):
    # Convert image data to numpy array
    image = Image.open(io.BytesIO(image_data))
    image_array = np.array(image)
    
    # Find face locations
    face_locations = face_recognition.face_locations(image_array)
    if not face_locations:
        return None
    
    # Get face encodings
    face_encodings = face_recognition.face_encodings(image_array, face_locations)
    if not face_encodings:
        return None
    
    return face_encodings[0]

def compare_faces(known_encoding, unknown_encoding, tolerance=0.6):
    if known_encoding is None or unknown_encoding is None:
        return False
    return face_recognition.compare_faces([known_encoding], unknown_encoding, tolerance=tolerance)[0]

# Routes
@app.route('/register', methods=['POST'])
def register():
    try:
        # Get form data
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if not all([name, email, password]):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Get face image
        if 'image' not in request.files:
            return jsonify({'error': 'No face image provided'}), 400
        
        image_file = request.files['image']
        image_data = image_file.read()
        
        # Get face encoding
        face_encoding = get_face_encoding(image_data)
        if face_encoding is None:
            return jsonify({'error': 'No face detected in the image'}), 400
        
        # Hash password
        hashed_password = generate_password_hash(password)
        
        # Store in database
        db = get_db()
        cursor = db.cursor()
        
        try:
            cursor.execute(
                'INSERT INTO users (name, email, password, face_encoding) VALUES (?, ?, ?, ?)',
                (name, email, hashed_password, face_encoding.tobytes().hex())
            )
            db.commit()
            
            return jsonify({
                'message': 'Registration successful',
                'user': {
                    'name': name,
                    'email': email
                }
            }), 201
            
        except sqlite3.IntegrityError:
            return jsonify({'error': 'Email already registered'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/login', methods=['POST'])
def login():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No face image provided'}), 400
        
        image_file = request.files['image']
        image_data = image_file.read()
        
        # Get face encoding from login attempt
        unknown_encoding = get_face_encoding(image_data)
        if unknown_encoding is None:
            return jsonify({'error': 'No face detected in the image'}), 400
        
        # Get all users from database
        db = get_db()
        cursor = db.cursor()
        cursor.execute('SELECT id, name, email, face_encoding FROM users')
        users = cursor.fetchall()
        
        # Find matching user
        for user in users:
            known_encoding = np.frombuffer(bytes.fromhex(user['face_encoding']), dtype=np.float64)
            if compare_faces(known_encoding, unknown_encoding):
                # Generate JWT token using flask-jwt-extended
                access_token = create_access_token(identity=user['id'])
                
                # Record attendance with current timestamp
                current_time = datetime.utcnow()
                cursor.execute(
                    'INSERT INTO attendance (user_id, timestamp, type, status) VALUES (?, ?, ?, ?)',
                    (user['id'], current_time, 'clock_in', 'on_time')
                )
                db.commit()
                
                return jsonify({
                    'message': 'Login successful',
                    'access_token': access_token,
                    'user': {
                        'id': user['id'],
                        'name': user['name'],
                        'email': user['email']
                    }
                }), 200
        
        return jsonify({'error': 'Face not recognized'}), 401
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/enroll_face', methods=['POST'])
@jwt_required()
def enroll_face():
    user_id = get_jwt_identity()
    user = User.query.get(user_id)
    
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image_data = request.files['image'].read()
    face_embedding = get_face_encoding(image_data)
    
    if not face_embedding:
        return jsonify({'error': 'No face detected in image'}), 400
    
    user.face_embedding = face_embedding
    user.enrollment_status = True
    db.session.commit()
    
    return jsonify({'message': 'Face enrolled successfully'})

@app.route('/recognize_face', methods=['POST'])
def recognize_face():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image_data = request.files['image'].read()
    face_embedding = get_face_encoding(image_data)
    
    if not face_embedding:
        return jsonify({'error': 'No face detected in image'}), 400
    
    # Compare with all enrolled faces
    users = User.query.filter_by(enrollment_status=True).all()
    best_match = None
    highest_similarity = 0.6  # Minimum threshold for matching
    
    for user in users:
        if user.face_embedding:
            similarity = compare_faces(user.face_embedding, face_embedding)
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = user
    
    if best_match:
        return jsonify({
            'user_id': best_match.id,
            'name': best_match.name,
            'confidence': float(highest_similarity)
        })
    
    return jsonify({'error': 'Face not recognized'}), 404

@app.route('/record_attendance', methods=['POST'])
@jwt_required()
def record_attendance():
    user_id = get_jwt_identity()
    data = request.get_json()
    
    attendance = Attendance(
        user_id=user_id,
        type=data['type'],
        status=data.get('status', 'on_time')
    )
    
    db.session.add(attendance)
    db.session.commit()
    
    return jsonify({'message': 'Attendance recorded successfully'})

@app.route('/get_attendance', methods=['GET'])
@jwt_required()
def get_attendance():
    user_id = get_jwt_identity()
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    query = Attendance.query.filter_by(user_id=user_id)
    
    if start_date:
        query = query.filter(Attendance.timestamp >= datetime.fromisoformat(start_date))
    if end_date:
        query = query.filter(Attendance.timestamp <= datetime.fromisoformat(end_date))
    
    attendances = query.order_by(Attendance.timestamp.desc()).all()
    
    return jsonify([{
        'id': a.id,
        'timestamp': a.timestamp.isoformat(),
        'type': a.type,
        'status': a.status
    } for a in attendances])

@app.route('/attendance', methods=['GET'])
def get_attendance_sqlite():
    try:
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        if not token:
            return jsonify({'error': 'No token provided'}), 401
        
        try:
            payload = jwt.decode(token, app.config['JWT_SECRET_KEY'], algorithms=['HS256'])
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token has expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401
        
        user_id = payload['user_id']
        
        db = get_db()
        cursor = db.cursor()
        cursor.execute('''
            SELECT a.*, u.name 
            FROM attendance a 
            JOIN users u ON a.user_id = u.id 
            WHERE a.user_id = ? 
            ORDER BY a.timestamp DESC
        ''', (user_id,))
        
        attendance_records = cursor.fetchall()
        
        return jsonify({
            'attendance': [dict(record) for record in attendance_records]
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    init_db()
    app.run(debug=True)
