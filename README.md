# 🕒 Attendance Tracking System

A simple and easy-to-use **Attendance Tracking System** that helps you record and manage attendance for students or employees. This project is useful for schools, colleges, offices, and any place where you need to keep track of who was present or absent.

---

## 📘 What is This Project?

This system allows you to:

* ✅ Record when a person (like a student or employee) comes in (login)
* ✅ Record when they leave (logout)
* ✅ See how many hours they worked or attended
* ✅ Save this information in a proper format (like an Excel sheet)
* ✅ Prevent people from logging in again if already logged in
* ✅ Track if someone came late (example: after 9:05 AM)

You can use this to **track daily attendance**, **working hours**, and even **generate reports**.

---

## 🎯 Why I Built This

Most attendance systems are complex or expensive. I wanted to create a simple system that anyone can understand and use — especially useful for:

* 📚 Students working on database or computer vision projects
* 👨‍🏫 Teachers who want a digital system to track attendance
* 🧑‍💼 Office managers who want to monitor staff timings

---

## 🔧 How Does It Work?

This system uses a **webcam** to detect faces and log attendance automatically.

Here’s how it works:

1. **Registration**
   New users register by showing their face in front of the camera. Their face data is saved.

2. **Model Training**
   After registration, the system learns how each person looks by training a model.

3. **Login (Attendance In)**
   When a person shows up again, the system recognizes their face and marks them as "present" with the login time.

4. **Late Login Warning**
   If someone logs in after 9:05 AM, the system marks them as "Late".

5. **Logout (Attendance Out)**
   When a person leaves, they can log out by showing their face again. The system records the time and calculates how many hours they stayed.

6. **Export Data**
   All login/logout data is saved in an Excel file with the person's name, date, time, working hours, and status (on time or late).

---

## 🧰 Tools & Technologies Used

* **Python**
* **OpenCV** – for webcam and face detection
* **MediaPipe** – for face recognition
* **Pandas** – for handling Excel files
* **FaceMesh** – for facial landmark tracking

---

## 📁 What’s Inside the Project

```
├── main.py                # Main code to run the system
├── registration.py        # Face registration script
├── train_model.py         # Trains the face recognition model
├── login.py               # Handles face-based login
├── logout.py              # Handles face-based logout
├── attendance_log.xlsx    # Excel file where data is saved
├── README.md              # This file (project explanation)
```

---

## 🚀 How to Use It (Step-by-Step)

1. **Install Python and Required Libraries**
   You need Python installed on your system. Install required libraries:

   ```bash
   pip install opencv-python mediapipe pandas
   ```

2. **Register Users**
   Run `registration.py` and stand in front of the camera to register.

3. **Train the Model**
   After registration, run `train_model.py` to let the system learn your face.

4. **Login (Mark Attendance)**
   Run `login.py` to detect and mark attendance for the day.

5. **Logout (Mark Exit Time)**
   Run `logout.py` to record the logout time.

6. **Check Attendance**
   Open `attendance_log.xlsx` to view the attendance record.

---

## 📊 Example of Excel Output

| Name  | Date       | Login Time | Logout Time | Working Hours | Status  |
| ----- | ---------- | ---------- | ----------- | ------------- | ------- |
| Avula | 2025-07-22 | 08:57 AM   | 04:30 PM    | 7 hrs 33 mins | On Time |
| Nikhil  | 2025-07-22 | 09:12 AM   | 04:45 PM    | 7 hrs 33 mins | Late    |

---

## ✅ Benefits of This Project

* 👀 No need for ID cards or manual entry — face-based login
* 🧾 Saves time and avoids mistakes in attendance tracking
* 📁 Keeps clean, shareable Excel records of every person
* 💡 Great project to learn **Computer Vision + Python + Real Use Case**

---

## 📌 Use Cases

* 🏫 Classroom attendance
* 🏢 Office login/logout tracking
* 👨‍💻 College mini or major project
* 📋 Digital check-in system for events

---
