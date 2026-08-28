# src/jatayu/io/database.py
import os
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

class JatayuDatabaseManager:
    def __init__(self):
        """Initializes the secure connection layer to your Supabase PostgreSQL core."""
        url: str = os.getenv("SUPABASE_URL", "")
        key: str = os.getenv("SUPABASE_ANON_KEY", "")
        
        if not url or not key:
            print("[Database Warning] Missing Supabase environment credentials!")
            
        self.client: Client = create_client(url, key)

    def register_user(self, email: str, password: str, role: str) -> dict:
        """
        Creates a new user account. Supabase automatically hashes the password
        internally before writing it to the encrypted auth database.
        """
        try:
            # 1. Sign up the user inside Supabase Auth
            auth_response = self.client.auth.sign_up({
                "email": email,
                "password": password
            })
            
            user = auth_response.user
            if not user:
                raise Exception("Authentication sign-up initialization failed.")

            # 2. Insert metadata profile details into public.profiles table
            profile_data = {
                "id": user.id,
                "email": email,
                "role": role
            }
            self.client.table("profiles").insert(profile_data).execute()
            
            print(f"[DB Core] Successfully registered user: {email} with role: {role}")
            return {"status": "success", "user_id": user.id}
            
        except Exception as e:
            print(f"[DB Error] Registration failure: {e}")
            return {"status": "error", "message": str(e)}

    def authenticate_user(self, email: str, password: str) -> dict:
        """Validates credentials and retrieves a secure runtime state session token."""
        try:
            session_response = self.client.auth.sign_in_with_password({
                "email": email,
                "password": password
            })
            print(f"[DB Core] Successful workspace login event for: {email}")
            return {
                "status": "success", 
                "token": session_response.session.access_token,
                "user_id": session_response.user.id
            }
        except Exception as e:
            print(f"[DB Error] Authentication failed: {e}")
            return {"status": "error", "message": str(e)}
