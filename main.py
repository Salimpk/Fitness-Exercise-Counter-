import cv2
import mediapipe as mp
import numpy as np

class FitnessTracker:
    def __init__(self):
        # Initialize MediaPipe Pose solution
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize pose detection
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        )
        
        # Exercise tracking variables
        self.exercise_type = None
        self.rep_count = 0
        self.stage = None

    def calculate_angle(self, a, b, c):
        """
        Calculate angle between three points
        
        Args:
            a (list): First point coordinates [x, y]
            b (list): Middle point coordinates [x, y]
            c (list): Last point coordinates [x, y]
        
        Returns:
            float: Angle between the points
        """
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
        
        return angle

    def detect_pushup(self, landmarks):
        """
        Detect push-up repetitions based on elbow and shoulder angles
        
        Args:
            landmarks (list): MediaPipe pose landmarks
        
        Returns:
            tuple: Updated rep count and stage
        """
        # Get coordinates
        shoulder = [
            landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
            landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
        ]
        elbow = [
            landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
            landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y
        ]
        wrist = [
            landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,
            landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y
        ]
        
        # Calculate angle
        angle = self.calculate_angle(shoulder, elbow, wrist)
        
        # Push-up logic
        if angle > 160:
            self.stage = "up"
        if angle < 90 and self.stage == "up":
            self.stage = "down"
            self.rep_count += 1
        
        return self.rep_count, self.stage

    def detect_squat(self, landmarks):
        """
        Detect squat repetitions based on hip, knee, and ankle angles
        
        Args:
            landmarks (list): MediaPipe pose landmarks
        
        Returns:
            tuple: Updated rep count and stage
        """
        # Get coordinates
        hip = [
            landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
            landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y
        ]
        knee = [
            landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x,
            landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y
        ]
        ankle = [
            landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
            landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y
        ]
        
        # Calculate angle
        angle = self.calculate_angle(hip, knee, ankle)
        
        # Squat logic
        if angle > 160:
            self.stage = "standing"
        if angle < 90 and self.stage == "standing":
            self.stage = "squatting"
            self.rep_count += 1
        
        return self.rep_count, self.stage

    def track_exercise(self, exercise_type='pushup'):
        """
        Main method to track exercises using webcam
        
        Args:
            exercise_type (str): Type of exercise to track (pushup or squat)
        """
        # Open webcam
        cap = cv2.VideoCapture(0)
        
        # Reset exercise tracking
        self.exercise_type = exercise_type
        self.rep_count = 0
        self.stage = None
        
        while cap.isOpened():
            # Read feed
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Make detection
            results = self.pose.process(image)
            
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Choose exercise detection
                if exercise_type == 'pushup':
                    rep_count, stage = self.detect_pushup(landmarks)
                elif exercise_type == 'squat':
                    rep_count, stage = self.detect_squat(landmarks)
                
                # Visualize detection
                cv2.putText(image, f'Exercise: {exercise_type.upper()}', 
                            (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.putText(image, f'Reps: {rep_count}', 
                            (10,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.putText(image, f'Stage: {stage}', 
                            (10,110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                
            except Exception as e:
                print(f"Error detecting landmarks: {e}")
            
            # Render detections
            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    image, 
                    results.pose_landmarks, 
                    self.mp_pose.POSE_CONNECTIONS
                )
            
            # Display
            cv2.imshow('Fitness Tracker', image)
            
            # Exit condition
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        
        # Clean up
        cap.release()
        cv2.destroyAllWindows()

def main():
    # Create fitness tracker instance
    fitness_tracker = FitnessTracker()
    
    # Start tracking (default push-ups, can be changed to 'squat')
    fitness_tracker.track_exercise('pushup')

if __name__ == "__main__":
    main()

# Requirements:
# pip install opencv-python
# pip install mediapipe
# pip install numpy