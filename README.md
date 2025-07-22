# 🕒 Face Recognition-Based Attendance Tracking System

This project is an **AI-powered Attendance Tracking System** that uses a webcam and facial recognition to **automatically detect and record a person's attendance** — without the need for manual entry or biometric devices.

It combines **Computer Vision** and **Machine Learning** to accurately track who is present, when they arrive, and when they leave.

---

## 📌 What Does This System Do?

* ✅ Detects faces in real-time using your webcam
* ✅ Identifies each person using face recognition
* ✅ Automatically marks **Login Time**
* ✅ Tracks **Logout Time** as soon as the person leaves
* ✅ Records the session duration and saves it to an Excel/CSV file
* ✅ Prevents a person from logging in multiple times on the same day
* ✅ Shows whether the person was **“Late”** (if login is after 9:05 AM)

---

## 🧠 Technologies & Tools Used

| Tool/Library          | Purpose                                   |
| --------------------- | ----------------------------------------- |
| **MediaPipe**         | Detect and extract facial landmarks       |
| **OpenCV**            | Video capture and face preprocessing      |
| **Scikit-learn**      | Train a facial recognition model          |
| **Random Forest**     | Machine Learning algorithm for prediction |
| **pandas**            | Manage and save attendance records        |
| **ExcelWriter / CSV** | Save reports in readable format           |

---

## ⚙️ How the System Works (Step-by-Step)

1. **Registration**

   * New users register by showing their face to the webcam.
   * Face embeddings (unique facial features) are saved.

2. **Model Training**

   * A **Random Forest Classifier** is trained using facial data.
   * The model is automatically retrained every time a new user is added.

3. **Login Detection**

   * When someone stands in front of the camera, the system identifies their face.
   * If the person is not already logged in for the day, it records their login time.
   * If login is after 9:05 AM, it labels it as **“Late Login”**.

4. **Logout Detection**

   * When the same person returns later and confirms identity, their logout time is recorded.
   * Total duration of stay is calculated.

5. **Report Generation**

   * All login/logout data is saved in an **Excel sheet** with:

     * Date
     * User name
     * Login time
     * Logout time
     * Session duration
     * Late login status

---

## 🎯 Key Features

* 🧠 **AI-based Face Recognition**
* 📅 **Daily Attendance Sheet**
* 🕒 **Automatic Time Tracking**
* ✅ **Prevents Duplicate Logins**
* 🚫 **No Manual Entry or Passwords Needed**
* 📄 **Excel Report Generation**

---

## 💡 Why Random Forest?

We used the **Random Forest** algorithm (a popular Machine Learning technique) because:

* It is robust and performs well on complex patterns (like faces).
* It gives high accuracy even with limited training data.
* It handles multiple users well without overfitting.

---

## 📁 Project Structure

```
├── face_registration.py       # Register new faces
├── train_model.py             # Train facial recognition model
├── login_system.py            # Detect and record login
├── logout_system.py           # Detect logout and update records
├── utils/                     # Helper functions (face detection, Excel writing)
├── embeddings/                # Saved face data
├── attendance_log.xlsx        # Daily attendance file
└── README.md                  # Project overview
```

---

## 🧪 How to Run the System

1. **Register Users**

   ```bash
   python face_registration.py
   ```

2. **Train the Model**

   ```bash
   python train_model.py
   ```

3. **Start Login System**

   ```bash
   python login_system.py
   ```

4. **Start Logout System**

   ```bash
   python logout_system.py
   ```

> All logs will be saved in the Excel file named `attendance_log.xlsx`.

---

## 📌 Who Is This For?

* 📚 **Schools & Colleges** — Student or faculty attendance
* 🏢 **Offices** — Employee time tracking
* 👨‍💻 **Developers** — Learning computer vision + ML + automation
* 🎓 **Final Year Projects** — For B.Tech/BCA/MCA students

---
