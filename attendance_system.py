import pandas as pd
import numpy as np
import seaborn as sns
import mediapipe as mp
import cv2
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from mlxtend.plotting import plot_decision_regions
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import cross_val_score, train_test_split,cross_validate
import time
from datetime import datetime
import os
class FaceSystem:
    def __init__(self):
        self.df = pd.read_csv(r"C:\Users\Nikhil\Downloads\face_mesh_dataset.xls") if self.exists(r"C:\Users\Nikhil\Downloads\face_mesh_dataset.xls") else pd.DataFrame(columns=[str(i) for i in range(1404)] + ["name"])
        self.model = None
        self.login_name = None
        self.login_time = None

    def exists(self, path):
        try:
            pd.read_csv(r"C:\Users\Nikhil\Downloads\face_mesh_dataset.xls")
            return True
        except:
            return False

    def register(self):
        vid = cv2.VideoCapture(2)
        fm = mp.solutions.face_mesh
        fm_mdl = fm.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        face_data = [str(i) for i in range(1404)] + ["name"]
        df = self.df.copy()
        count = len(df)

        while True:
            register = input("Do you want to register? (y or n): ")
            if register == "y":
                name = input("Enter your name: ")

                while True:
                    s, f = vid.read()
                    if not s:
                        print("Camera error.")
                        break

                    rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                    op = fm_mdl.process(rgb)

                    if op.multi_face_landmarks:
                        mp.solutions.drawing_utils.draw_landmarks(
                            image=f,
                            landmark_list=op.multi_face_landmarks[0],
                            connections=fm.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
                        )

                    if cv2.waitKey(1) & 255 == ord('s'):
                        print("Starting capture for", name)
                        sample = 0

                        while sample < 300:
                            s, f = vid.read()
                            if not s:
                                break

                            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                            op = fm_mdl.process(rgb)

                            if op.multi_face_landmarks:
                                face = []
                                for i in range(468):
                                    lm = op.multi_face_landmarks[0].landmark[i]
                                    face.extend([lm.x, lm.y, lm.z])
                                face.append(name)
                                df.loc[count] = face
                                count += 1
                                sample += 1
                                print(f"Captured {sample}/300 for {name}")

                            cv2.imshow("w1", f)
                            if cv2.waitKey(1) & 255 == ord('c'):
                                break

                        print("Done collecting for", name)
                        break

                    cv2.imshow("w1", f)
                    if cv2.waitKey(10) & 255 == ord('c'):
                        break
            else:
                break

        df.to_csv(r"C:\Users\Nikhil\Downloads\face_mesh_dataset.xls", index=False)
        self.train()
        print("Data saved to face_mesh_dataset.csv")
        vid.release()
        cv2.destroyAllWindows()

    def train(self):
        df = pd.read_csv(r"C:\Users\Nikhil\Downloads\face_mesh_dataset.xls")
        fv = df.drop("name", axis=1)
        cv = df["name"]
        final_pr_data = []
        for i in fv.values:
            md = i.reshape(468, 3)
            centre = md - md[1]
            dis = np.linalg.norm(md[33] - md[263])
            fpd = centre / dis
            final_pr_data.append(fpd.flatten())

        fv = pd.DataFrame(final_pr_data)
        X_train, X_test, y_train, y_test = train_test_split(fv, cv, test_size=0.2, random_state=42, stratify=cv)
        
        rf = RandomForestClassifier(n_estimators=74,max_depth=13,max_features='log2')
        rf.fit(X_train, y_train)
        self.model = rf
        print("Training done successfully.")

    def login(self):        
        df = pd.read_csv(r"C:\Users\Nikhil\Downloads\face_mesh_dataset.xls")
        if self.model is None:
            if not os.path.exists(r"C:\Users\Nikhil\Downloads\face_mesh_dataset.xls"):
                return
    
            df = pd.read_csv(r"C:\Users\Nikhil\Downloads\face_mesh_dataset.xls")
            fv = df.drop("name", axis=1)
            cv = df["name"]
    
            final_pr_data = []
            for i in fv.values:
                md = i.reshape(468,3)
                centre = md-i.reshape(468,3)[1]
                dis = np.linalg.norm(i.reshape(468,3)[33]-i.reshape(468,3)[263])
                fpd = centre/dis
                final_pr_data.append(fpd.flatten())
    
            fv = pd.DataFrame(final_pr_data)
            X_train,X_test,y_train,y_test = train_test_split(fv,cv,test_size=0.2,random_state=42,stratify=cv)
    
            self.model = RandomForestClassifier(n_estimators=74,max_depth=13,max_features='log2')
            self.model.fit(X_train, y_train)
    
        
       
        vid=cv2.VideoCapture(2)
        fp = mp.solutions.face_mesh
        fm_model=fp.FaceMesh(static_image_mode=False, # for img it is true - video = False
            max_num_faces=1, # number of faces to track
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        
        while True:
                s,f=vid.read()
                if s==False:
                    break
                rgb=cv2.cvtColor(f,cv2.COLOR_BGR2RGB)
            
                result=fm_model.process(rgb)
            
                if result.multi_face_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks (image = f,landmark_list=result.multi_face_landmarks[0],connections=fp.FACEMESH_TESSELATION,
                          landmark_drawing_spec=None,connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style())
                    
                cv2.imshow("face",f)
            
                if cv2.waitKey(1) & 255 == ord("c"):
                    break
                face=[]
                if cv2.waitKey(100) & 255 == ord("s"):
                    if result.multi_face_landmarks:
                        for i in result.multi_face_landmarks[0].landmark:
                            face.append(i.x)
                            face.append(i.y)
                            face.append(i.z)
        
                
                if bool(face)!=False:
                    get_face = []
                    f1=np.array(face).reshape(468,3)
                    center = f1 - f1[1]
                    distance = np.linalg.norm(f1[33]-f1[263])
                    fpd = center/distance
                    get_face.append(fpd.flatten())
        
                    # prediction
                    y_pred = self.model.predict(get_face)
                    predicted_name = y_pred[0]

                    if self.login_name == predicted_name:
                        print(f"Already logged in as {self.login_name}")
                        break
                    else:
                        self.login_name = predicted_name
                        self.login_time = time.ctime()
                        print(f"Login successful for {predicted_name}")
                        print(f"Login Time: {self.login_time}")
                        self.save_attendance(predicted_name, self.login_time, None)

                        
                        # Late login check
                        current_time = time.localtime()
                        if current_time.tm_hour > 9 or (current_time.tm_hour == 9 and current_time.tm_min > 5):
                            print("Late login")
                        break

                    
        
                    print(f"Login successfull for {y_pred}")
                    print(f"Login Time:{time.ctime()}")
        
    def logout(self):
        if self.login_name is None:
            print("No user is currently logged in. Please login first.")
            return

        if self.model is None:
            if not os.path.exists(r"C:\Users\Nikhil\Downloads\face_mesh_dataset.xls"):
                return
    
            df = pd.read_csv(r"C:\Users\Nikhil\Downloads\face_mesh_dataset.xls")
            fv = df.drop("name", axis=1)
            cv = df["name"]
    
            final_pr_data = []
            for i in fv.values:
                md = i.reshape(468,3)
                centre = md-i.reshape(468,3)[1]
                dis = np.linalg.norm(i.reshape(468,3)[33]-i.reshape(468,3)[263])
                fpd = centre/dis
                final_pr_data.append(fpd.flatten())
    
            fv = pd.DataFrame(final_pr_data)
            X_train,X_test,y_train,y_test = train_test_split(fv,cv,test_size=0.2,random_state=42,stratify=cv)
    
            self.model = RandomForestClassifier(n_estimators=74,max_depth=13,max_features='log2')
            self.model.fit(X_train, y_train)
    
        
       
        vid=cv2.VideoCapture(2)
        fp = mp.solutions.face_mesh
        fm_model=fp.FaceMesh(static_image_mode=False, # for img it is true - video = False
            max_num_faces=1, # number of faces to track
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        
        while True:
                s,f=vid.read()
                if s==False:
                    break
                rgb=cv2.cvtColor(f,cv2.COLOR_BGR2RGB)
            
                result=fm_model.process(rgb)
            
                if result.multi_face_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks (image = f,landmark_list=result.multi_face_landmarks[0],connections=fp.FACEMESH_TESSELATION,
                          landmark_drawing_spec=None,connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style())
                    
                cv2.imshow("face",f)
            
                if cv2.waitKey(1) & 255 == ord("c"):
                    break
                face=[]
                if cv2.waitKey(100) & 255 == ord("s"):
                    if result.multi_face_landmarks:
                        for i in result.multi_face_landmarks[0].landmark:
                            face.append(i.x)
                            face.append(i.y)
                            face.append(i.z)
        
                
                if bool(face)!=False:
                    get_face = []
                    f1=np.array(face).reshape(468,3)
                    center = f1 - f1[1]
                    distance = np.linalg.norm(f1[33]-f1[263])
                    fpd = center/distance
                    get_face.append(fpd.flatten())
        
                    # prediction
                    y_pred = self.model.predict(get_face)
                    logout_name = y_pred[0]

                    if logout_name != self.login_name:
                        print(f"Face does not match the logged-in user '{self.login_name}'. Cannot logout.")
                        continue

                    self.logout_name = logout_name
                    self.logout_time = time.ctime()
                    print(f"Logout successful for {logout_name}. Thank you!")
                    self.login_name = None  # Reset login state
                    print(f"Logout Time: {self.logout_time}")
                    self.save_attendance(logout_name, self.login_time, self.logout_time)
                    break
                    
        
                    print(f"Logout successfull for {y_pred} thank you!")
                    print(f"Logout Time:{time.ctime()}")
        
        vid.release()
        cv2.destroyAllWindows()

    def save_attendance(self, name, login_time, logout_time):
        file_path = (r"C:\Users\Nikhil\Downloads\attendance_log.csv")
        today = datetime.now().strftime("%Y-%m-%d")
    
    
        if logout_time:
            t1 = datetime.strptime(login_time, "%a %b %d %H:%M:%S %Y")
            t2 = datetime.strptime(logout_time, "%a %b %d %H:%M:%S %Y")
            hours = round((t2 - t1).total_seconds() / 3600, 2)
        else:
            hours = ""
    
       
        row = {
            "Name": name,
            "Date": today,
            "Login Time": login_time,
            "Logout Time": logout_time,
            "Working Hours": hours
        }
    
        
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
    
            
            if not df[(df["Name"] == name) & (df["Date"] == today)].empty:
                print(f"{name} already marked for today.")
                return
    
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])
    
        df.to_csv(file_path, index=False)
        print(f"Attendance saved for {name}.")



face_sys = FaceSystem()

while True:
    print("\nChoose an option:")
    print("1. Register")
    print("2.Login")
    print("3.Logout")
    print("4. Exit")

    choice = input("Enter your choice: ")

    if choice == "1":
        face_sys.register()
    elif choice == "2":
        face_sys.login()
    elif choice == "3":
        face_sys.logout()
    elif choice == "4":
        break
    else:
        print("Invalid choice. Try again.")