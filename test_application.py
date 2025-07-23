#!/usr/bin/env python3
"""
Test script to verify the Market Survey System is working correctly
"""

import subprocess
import time
import requests
import sys
import os

def test_backend():
    """Test backend API endpoints"""
    print("Testing backend API...")
    
    base_url = "http://localhost:8000"
    
    try:
        # Test root endpoint
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            print("✅ Root endpoint working")
        else:
            print("❌ Root endpoint failed")
            return False
        
        # Test projects endpoint
        response = requests.get(f"{base_url}/api/projects")
        if response.status_code == 200:
            print("✅ Projects endpoint working")
        else:
            print("❌ Projects endpoint failed")
            return False
        
        # Test status endpoint
        response = requests.get(f"{base_url}/api/status")
        if response.status_code == 200:
            print("✅ Status endpoint working")
        else:
            print("❌ Status endpoint failed")
            return False
        
        # Test AI insights endpoint
        response = requests.get(f"{base_url}/api/ai-insights")
        if response.status_code == 200:
            print("✅ AI insights endpoint working")
        else:
            print("❌ AI insights endpoint failed")
            return False
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to backend. Make sure it's running on port 8000")
        return False
    except Exception as e:
        print(f"❌ Backend test failed: {e}")
        return False

def test_frontend():
    """Test frontend accessibility"""
    print("Testing frontend...")
    
    try:
        response = requests.get("http://localhost:3000")
        if response.status_code == 200:
            print("✅ Frontend accessible")
            return True
        else:
            print("❌ Frontend not accessible")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to frontend. Make sure it's running on port 3000")
        return False
    except Exception as e:
        print(f"❌ Frontend test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Testing Market Survey System...")
    print("=" * 50)
    
    backend_ok = test_backend()
    frontend_ok = test_frontend()
    
    print("=" * 50)
    
    if backend_ok and frontend_ok:
        print("🎉 All tests passed! The application is working correctly.")
        print("\nAccess the application:")
        print("- Frontend: http://localhost:3000")
        print("- Backend API: http://localhost:8000")
        print("- API Docs: http://localhost:8000/docs")
        print("- Admin Dashboard: http://localhost:3000/admin")
    else:
        print("❌ Some tests failed. Please check the logs above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
