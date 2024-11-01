class User:
    def __init__(self, username):
        self.username = username
        self.taken_courses = []

    def take_course(self, course):
        self.taken_courses.append(course)

    def view_taken_courses(self):
        if self.taken_courses:
            print("Courses taken by ",self.username,": ")
            for course in self.taken_courses:
                print("- " + course)
        else:
            print(self.username," has not taken any courses yet.")

# Example usage
user1 = User("user1")
user1.take_course("Python Programming")
user1.take_course("Data Science Fundamentals")
user1.view_taken_courses()
