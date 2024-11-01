class AuthenticationSystem:
    def __init__(self):
        # Dictionary to store username-password pairs
        self.credentials = {
            "user1": "password1",
            "user2": "password2",
            "user3": "password3"
        }

    def sign_in(self, username, password):
        # Check if the username exists in the credentials dictionary
        if username in self.credentials:
            # Check if the entered password matches the stored password
            if self.credentials[username] == password:
                print("Sign-in successful. Welcome, ",username,"!")
            else:
                print("Incorrect password. Please try again.")
        else:
            print("Username ",username," not found. Please check your username or register if you are a new user.")

# Example usage
auth_system = AuthenticationSystem()
username = input("Enter your username: ")
password = input("Enter your password: ")
auth_system.sign_in(username, password)
