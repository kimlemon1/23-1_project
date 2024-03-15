from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pandas as pd
import time
import traceback

def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image

# Create a PoseLandmarker object.
mp_pose = solutions.pose

def calc_angle(L0, L1, L2):
    p0 = np.array([L0.x, L0.y, L0.z])
    p1 = np.array([L1.x, L1.y, L1.z])
    p2 = np.array([L2.x, L2.y, L2.z])

    v0 = p0 - p1
    v1 = p1 - p2
    unit_v0 = v0 / np.linalg.norm(v0)
    unit_v1 = v1 / np.linalg.norm(v1)

    arccos = np.arccos(np.dot(unit_v0, unit_v1))
    return np.degrees(arccos)

def calc_middle(L0, L1): # return numpy array of middel point
    p0 = np.array([L0.x, L0.y, L0.z])
    p1 = np.array([L1.x, L1.y, L1.z])
    
    return (p0 + p1) / 2

def get_angles(landmark, right_knee_angle, left_knee_angle, back_angle):
    p23 = landmark[0][23]
    p25 = landmark[0][25]
    p27 = landmark[0][27]
    p24 = landmark[0][24]
    p26 = landmark[0][26]
    p28 = landmark[0][28]
    p12 = landmark[0][12]
    p11 = landmark[0][11]

    right_knee_angle.append(calc_angle(p24, p26, p28))
    left_knee_angle.append(calc_angle(p23, p25, p27))

    #back : avg of two points
    p12_11 = calc_middle(p12, p11)
    p24_23 = calc_middle(p24, p23)
    p26_25 = calc_middle(p26, p25)

    v0 = p12_11 - p24_23
    v1 = p24_23 - p26_25

    unit_v0 = v0 / np.linalg.norm(v0)
    unit_v1 = v1 / np.linalg.norm(v1)

    arccos = np.arccos(np.dot(unit_v0, unit_v1))
    back_angle.append(np.degrees(arccos))   

# Initialize webcam capture.
# cap = cv2.VideoCapture(0)  # 0 is the default camera index. You can change it if you have multiple cameras.
left_right = ["left", "right"]
name_list = ["chaeyun", "chanwoo", "inseo", "jaehoon", "jeongwoo", "junho", "kihyun", "suho", "sunghyun", "wonyoung"]
for direction in left_right:
    for name in name_list:
        video_name = '_'.join([direction, name])
        dir = '.'.join([video_name,"MOV"])
        cap = cv2.VideoCapture("./videos/"+dir)
        print(video_name, "Open")
        print("#############################\n")
        current_frame = 0

        # STEP 2: Create an PoseLandmarker object.
        base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=True)
        detector = vision.PoseLandmarker.create_from_options(options)

        for i in range(33):
            globals()["idx_" + str(i) + "_x"] = []
            globals()["idx_" + str(i) + "_y"] = []
            globals()["idx_" + str(i) + "_z"] = []
        frame_idx = []
        end_flag = 0
        landmark_ls = []
        right_knee_angle = []
        left_knee_angle = []
        back_angle = []
        
        ignoring_flag = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                if ignoring_flag == 20:    
                    print("GETOUT")
                    break
                ignoring_flag += 1
                print("Ignoring empty camera frame.")
                continue
            
            # Process the frame to detect pose landmarks.
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

            results = detector.detect(image)

            # Visualize the detected landmarks on the frame.
            annotated_frame = draw_landmarks_on_image(image.numpy_view(), detection_result=results)
            landmark = results.pose_landmarks

            try:
                # append 좌표들 in dataset
                for i in range(33):
                    globals()["idx_" + str(i) + "_x"].append(landmark[0][i].x) # 1 * num_frames
                    globals()["idx_" + str(i) + "_y"].append(landmark[0][i].y)
                    globals()["idx_" + str(i) + "_z"].append(landmark[0][i].z)
                
                landmark_ls.append(landmark)
                get_angles(landmark, right_knee_angle, left_knee_angle, back_angle)

            except:
                if landmark_ls != []: # 마지막 전에 key points가 소실된 경우 이전 point의 값 사용
                    landmark = landmark_ls[-1]
                    for i in range(33):
                        globals()["idx_" + str(i) + "_x"].append(landmark[0][i].x) # 1 * num_frames
                        globals()["idx_" + str(i) + "_y"].append(landmark[0][i].y)
                        globals()["idx_" + str(i) + "_z"].append(landmark[0][i].z)
                    get_angles(landmark, right_knee_angle, left_knee_angle, back_angle)
                elif landmark_ls == []: # 초반에 keypoints 없을 때 : -1값 넣기
                    for i in range(33):
                        globals()["idx_" + str(i) + "_x"].append(-1) # 1 * num_frames
                        globals()["idx_" + str(i) + "_y"].append(-1)
                        globals()["idx_" + str(i) + "_z"].append(-1)
                    right_knee_angle.append(-1)
                    left_knee_angle.append(-1)
                    back_angle.append(-1)
                    continue
                else:
                    print("what?\n")
                    err_msg = traceback.format_exc()
                    print("error :", err_msg)
            
 

            # Display the annotated frame.
            cv2.imshow("Pose Landmarks", cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        # Release the webcam and close all OpenCV windows.
        cap.release()
        cv2.destroyAllWindows()
        
        # save keypoints (keypoints * 3)
        #save(video_name)
        mat_array = [] # (33*3) * num_frames
        for i in range(33):
            mat_array.append(globals()["idx_" + str(i) + "_x"])
            mat_array.append(globals()["idx_" + str(i) + "_y"])
            mat_array.append(globals()["idx_" + str(i) + "_z"])
        mat_array.append(right_knee_angle)
        mat_array.append(left_knee_angle)
        mat_array.append(back_angle)

        mat = np.array(mat_array)
        mat = mat.T # num_frames * 99
        print("size of mat : ", mat.shape)
        
        df = pd.DataFrame(mat)
        df.to_csv("./key_points/"+video_name+".csv", index=True)
        time.sleep(5)
        
        