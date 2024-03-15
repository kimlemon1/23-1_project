from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2

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

# Initialize webcam capture.
# cap = cv2.VideoCapture(0)  # 0 is the default camera index. You can change it if you have multiple cameras.
cap = cv2.VideoCapture('kihyun.MOV')

# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# STEP 2: Create an PoseLandmarker object.
base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)


# array to save angle in each frame
right_knee_angle = []
left_knee_angle = []
back_angle = []

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


while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Process the frame to detect pose landmarks.
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    
    results = detector.detect(image)

    # TODO: results.pose_landmarks를 활용하여 각도를 계산하는 함수를 만들면 됨.
    # 1. results.pose_landmarks에 딕셔너리 형태로 포즈 관련 키포인트가 저장되어 있음. 이를 전처리 하는 함수를 만들면 됨.
    # 예시로 3개의 x,y,z keypoint를 입력 후, 이를 활용하여 각도를 계산하는 함수를 만들면 되는데 input과 output은 너가 고민해서 하면 될 듯.
    # (첨부 파일의 pose_landmark.png 참고하면 landmark의 인덱스가 어떻게 되는지 확인 가능.)

    # results.pose_landmarks : for each frame (1, 33) dim

    # Visualize the detected landmarks on the frame.
    annotated_frame = draw_landmarks_on_image(image.numpy_view(), detection_result=results)
    landmark = results.pose_landmarks
    
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


    # Display the annotated frame.
    cv2.imshow("Pose Landmarks", cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))

    # Exit the loop if 'q' is pressed.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows.
cap.release()
cv2.destroyAllWindows()

print("Total frames : ", len(right_knee_angle))
print("rigth knee angle : ", sum(right_knee_angle)/ len(right_knee_angle))
print("left knee angle : ", sum(left_knee_angle)/ len(left_knee_angle))
print("back angle : ", sum(back_angle)/ len(back_angle))