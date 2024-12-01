from locust import HttpUser, task, between

class InvestigationAppUser(HttpUser):
    wait_time = between(1, 5)  # Simulate a wait time between tasks to mimic real users

    def on_start(self):
        # Perform login to access authenticated routes
        self.login()

    def login(self):
        # Perform login using valid credentials
        login_payload = {
            "username": "harshbohra916",
            "password": "123"
        }
        response = self.client.post("/login", data=login_payload)
        if response.status_code == 200:
            print("Logged in successfully!")
        else:
            print(f"Failed to login. Status code: {response.status_code}")

    @task(1)
    def load_homepage(self):
        # Simulate loading the homepage
        self.client.get("/")

    @task(2)
    def explore_page(self):
        # Simulate browsing the explore page
        self.client.get("/explore")



    @task(1)
    def download_entries(self):
        # Simulate downloading user entries
        self.client.get("/download_entries")

    @task(1)
    def user_profile(self):
        # Simulate viewing a user profile
        self.client.get("/user/harshbohra916")


    @task(1)
    def follow_user(self):
        # Simulate following a user (assuming 'harshbohra916' is a valid user)
        self.client.post("/follow/harshbohra916")

    @task(1)
    def unfollow_user(self):
        # Simulate unfollowing a user (assuming 'harshbohra916' is a valid user)
        self.client.post("/unfollow/harshbohra916")

    @task(1)
    def edit_profile(self):
        # Simulate editing a user profile
        edit_profile_payload = {
            "username": "harshbohra916",
            "about_me": "Testing load for profile update"
        }
        self.client.post("/edit_profile", data=edit_profile_payload)

    @task(1)
    def logout(self):
        # Simulate logging out
        self.client.get("/logout")
