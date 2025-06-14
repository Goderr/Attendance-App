import requests
import json
import os
from PIL import Image
import io

BASE_URL = 'http://127.0.0.1:5000'

def test_register():
    print("\nTesting Registration...")
    response = requests.post(
        f'{BASE_URL}/register',
        json={
            'name': 'John Doe',
            'email': 'john@example.com',
            'password': 'password123'
        }
    )
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 201

def test_login():
    print("\nTesting Login...")
    response = requests.post(
        f'{BASE_URL}/login',
        json={
            'email': 'john@example.com',
            'password': 'password123'
        }
    )
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    if response.status_code == 200:
        return response.json()['access_token']
    return None

def test_enroll_face(token, image_path):
    print("\nTesting Face Enrollment...")
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return False
    
    with open(image_path, 'rb') as img:
        files = {'image': img}
        headers = {'Authorization': f'Bearer {token}'}
        response = requests.post(
            f'{BASE_URL}/enroll_face',
            files=files,
            headers=headers
        )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200

def test_recognize_face(image_path):
    print("\nTesting Face Recognition...")
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return False
    
    with open(image_path, 'rb') as img:
        files = {'image': img}
        response = requests.post(
            f'{BASE_URL}/recognize_face',
            files=files
        )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200

def test_record_attendance(token):
    print("\nTesting Attendance Recording...")
    headers = {'Authorization': f'Bearer {token}'}
    response = requests.post(
        f'{BASE_URL}/record_attendance',
        json={'type': 'clock_in', 'status': 'on_time'},
        headers=headers
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200

def test_get_attendance(token):
    print("\nTesting Get Attendance...")
    headers = {'Authorization': f'Bearer {token}'}
    response = requests.get(
        f'{BASE_URL}/get_attendance',
        headers=headers
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200

def main():
    # Test registration
    if not test_register():
        print("Registration failed!")
        return
    
    # Test login
    token = test_login()
    if not token:
        print("Login failed!")
        return
    
    # Test face enrollment (you need to provide an image path)
    image_path = input("\nEnter the path to a face image for testing: ")
    if test_enroll_face(token, image_path):
        # Test face recognition with the same image
        test_recognize_face(image_path)
        
        # Test attendance recording
        test_record_attendance(token)
        
        # Test getting attendance history
        test_get_attendance(token)

if __name__ == '__main__':
    main() 