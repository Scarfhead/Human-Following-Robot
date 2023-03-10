####Mediapipe Posenet beginning

import cv2
import mediapipe as mp
import math as m
import time
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

def findDistance(x1, y1, x2, y2):
    dist = m.sqrt((x2-x1)**2+(y2-y1)**2)
    return dist

# For static images:
IMAGE_FILES = []
BG_COLOR = (192, 192, 192) # gray
with mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.5) as pose:
  for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)
    image_height, image_width, _ = image.shape
    # Convert the BGR image to RGB before processing.
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.pose_landmarks:
      continue
    print(
        f'Nose coordinates: ('
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width}, '
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height})'
    )

    annotated_image = image.copy()
    # Draw segmentation on the image.
    # To improve segmentation around boundaries, consider applying a joint
    # bilateral filter to "results.segmentation_mask" with "image".
    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
    bg_image = np.zeros(image.shape, dtype=np.uint8)
    bg_image[:] = BG_COLOR
    annotated_image = np.where(condition, annotated_image, bg_image)
    # Draw pose landmarks on the image.
    mp_drawing.draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)
    # Plot pose world landmarks.
    mp_drawing.plot_landmarks(
        results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
        

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
    print("[ Image Tracking Active ]")
    lm = results.pose_landmarks
    lmPose  = mp_pose.PoseLandmark
  ####Mediapipe Posenet End

    try:
  ####LearnOPenCV Code beginning
      l_shldr_x = str(lm.landmark[lmPose.LEFT_SHOULDER].x)
      if l_shldr_x is "None":
        print("Camera Frame is Empty")
        break
    ####LearnOpenCV Code end

    ####My Code beginning
      l_shldr_y = str(lm.landmark[lmPose.LEFT_SHOULDER].y)
      l_shldr_z = str(lm.landmark[lmPose.LEFT_SHOULDER].z)
      r_shldr_x = str(lm.landmark[lmPose.RIGHT_SHOULDER].x)
      r_shldr_y = str(lm.landmark[lmPose.RIGHT_SHOULDER].y)
      r_shldr_z = str(lm.landmark[lmPose.RIGHT_SHOULDER].z)
      nose_x = str(lm.landmark[lmPose.NOSE].x)
      distance = findDistance(float(l_shldr_x), float(l_shldr_y), float(r_shldr_x), float(r_shldr_y))
      print("Nose X Co-ordinate: " + nose_x)
      print("Left Shoulder (TOP) and Right Shoulder (BOTTOM) Co-Ordinates")
      print("[" + l_shldr_x + " , " + l_shldr_y + " , " + l_shldr_z + "]")
      print("[" + r_shldr_x + " , " + r_shldr_y + " , " + r_shldr_z + "]")
      print("....")
      print("Distance: " + str(distance))

      ####Detecting whether the subject is facing forward.
      if distance < 0.25:
        print("PLEASE FACE FORWARD")
      else:
        ####Code to centre the subject in the camera (before the robot moves towards them).
        if float(nose_x) < 0.45:
          print("TURN LEFT")
        elif float(nose_x) > 0.55:
          print("TURN RIGHT")
        else:
          print("CENTERED")
          ####Detecting how far the subject is from the robot.
          if distance < 0.62:
            print("MOVE FORWARD")
            print("....")
          ####Stopping the robot to prevent collision with the subject.
          elif distance > 0.62:
            print("HALT")
            print("....")
      
      if cv2.waitKey(5) & 0xFF == 27:
        print("Ending Simulation")
        print(".")
        print("..")
        print("...")
        print("....")
        break
      ####My Code End
    ####Make the robot do a 360 to look for the target
    except AttributeError as e:
      print("....")
      print("No person detected, please re-enter the frame.")
      print("STARTING 360 DEGREE ENVIROMENTAL SCAN....")
      print("Turning Left")
cap.release()
