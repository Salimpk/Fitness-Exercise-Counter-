class ExerciseCounter {
    constructor(videoId, canvasId) {
        this.videoElement = document.getElementById(videoId);
        this.canvasElement = document.getElementById(canvasId);
        this.ctx = this.canvasElement.getContext('2d');
        this.exerciseCount = 0;
        this.exerciseType = 'pushups';
        this.isCounting = false;
        this.detector = null;
        this.lastPose = null;
        this.currentState = 'up';
        this.usingFrontCamera = true;
        this.uploadedVideo = false;
        this.analysisInProgress = false;
        
        this.thresholds = {
            pushups: { angle: 100, minConfidence: 0.5 },
            squats: { angle: 90, minConfidence: 0.5 }
        };

        // Initialize video upload
        this.initVideoUpload();
    }

    async initialize() {
        try {
            await this.setupCamera();
            await this.loadPoseDetectionModel();
            this.updateUI();
            this.detectPoses();
        } catch (error) {
            console.error('Initialization error:', error);
            this.showStatus(`Error: ${error.message}`, 'error');
        }
    }

    initVideoUpload() {
        const videoUpload = document.getElementById('video-upload');
        const progressContainer = document.getElementById('progress-container');
        const progressBar = document.getElementById('progress-bar');
        
        videoUpload.addEventListener('change', async (event) => {
            const file = event.target.files[0];
            if (!file) return;
            
            if (!file.type.startsWith('video/')) {
                this.showStatus('Please upload a video file', 'error');
                return;
            }
            
            try {
                this.analysisInProgress = true;
                progressContainer.style.display = 'block';
                progressBar.style.width = '0%';
                progressBar.textContent = '0%';
                
                // Stop current camera stream if active
                if (this.videoElement.srcObject) {
                    this.videoElement.srcObject.getTracks().forEach(track => track.stop());
                }
                
                // Create video URL and set as source
                const videoURL = URL.createObjectURL(file);
                this.videoElement.src = videoURL;
                this.uploadedVideo = true;
                
                // Wait for video to load
                await new Promise((resolve) => {
                    this.videoElement.onloadedmetadata = resolve;
                    this.videoElement.onerror = () => {
                        throw new Error('Failed to load video');
                    };
                });
                
                // Show loading progress
                this.videoElement.onprogress = () => {
                    if (this.videoElement.buffered.length > 0) {
                        const percent = (this.videoElement.buffered.end(0) / this.videoElement.duration) * 100;
                        progressBar.style.width = `${percent}%`;
                        progressBar.textContent = `${Math.round(percent)}%`;
                    }
                };
                
                await this.videoElement.play();
                progressContainer.style.display = 'none';
                this.showStatus('Video loaded successfully. Analyzing...', 'loading');
                
                // Analyze the uploaded video
                await this.analyzeUploadedVideo();
                
            } catch (error) {
                console.error('Video upload error:', error);
                this.showStatus(`Error: ${error.message}`, 'error');
                progressContainer.style.display = 'none';
                this.analysisInProgress = false;
            }
        });
    }

    async analyzeUploadedVideo() {
        if (!this.detector) await this.loadPoseDetectionModel();
        
        this.exerciseCount = 0;
        this.currentState = 'up';
        this.updateUI();
        
        // Create a copy of the video element for analysis
        const analysisVideo = this.videoElement.cloneNode();
        analysisVideo.currentTime = 0;
        
        // Process each frame
        const processFrame = async () => {
            if (analysisVideo.paused || analysisVideo.ended || !this.analysisInProgress) {
                this.analysisInProgress = false;
                return;
            }
            
            try {
                const poses = await this.detector.estimatePoses(analysisVideo);
                this.drawPoses(poses);
                this.analyzePoses(poses);
                
                // Move to next frame
                analysisVideo.currentTime += 0.1; // Process 10 frames per second
            } catch (error) {
                console.error('Frame processing error:', error);
            }
            
            setTimeout(processFrame, 0);
        };
        
        analysisVideo.addEventListener('seeked', processFrame);
        processFrame();
    }

    async setupCamera() {
        if (this.uploadedVideo) return;
        
        const constraints = {
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: this.usingFrontCamera ? 'user' : 'environment'
            }
        };

        try {
            const stream = await navigator.mediaDevices.getUserMedia(constraints);
            this.videoElement.srcObject = stream;
            this.videoElement.src = ''; // Clear any uploaded video
            this.uploadedVideo = false;
            
            await new Promise((resolve) => {
                this.videoElement.onloadedmetadata = () => {
                    this.videoElement.play();
                    resolve();
                };
            });
        } catch (error) {
            throw new Error('Could not access the camera. Please ensure you have granted camera permissions.');
        }
    }

    async loadPoseDetectionModel() {
        await tf.ready();
        const model = poseDetection.SupportedModels.BlazePose;
        const detectorConfig = {
            runtime: 'tfjs',
            enableSmoothing: true,
            modelType: 'lite'
        };
        this.detector = await poseDetection.createDetector(model, detectorConfig);
    }

    async detectPoses() {
        if (!this.detector || this.analysisInProgress) {
            requestAnimationFrame(() => this.detectPoses());
            return;
        }

        try {
            const poses = await this.detector.estimatePoses(this.videoElement);
            this.drawPoses(poses);
            this.analyzePoses(poses);
        } catch (error) {
            console.error('Pose detection error:', error);
        }
        
        requestAnimationFrame(() => this.detectPoses());
    }

    drawPoses(poses) {
        this.ctx.clearRect(0, 0, this.canvasElement.width, this.canvasElement.height);
        
        if (poses.length > 0) {
            const pose = poses[0];
            this.drawKeypoints(pose.keypoints);
            this.drawSkeleton(pose.keypoints);
        }
    }

    drawKeypoints(keypoints) {
        this.ctx.fillStyle = 'red';
        this.ctx.strokeStyle = 'white';
        this.ctx.lineWidth = 2;

        for (const keypoint of keypoints) {
            if (keypoint.score > 0.3) {
                this.ctx.beginPath();
                this.ctx.arc(keypoint.x, keypoint.y, 5, 0, 2 * Math.PI);
                this.ctx.fill();
                this.ctx.stroke();
            }
        }
    }

    drawSkeleton(keypoints) {
        const adjacentKeyPoints = [
            [0, 1], [0, 2], [1, 3], [2, 4], [5, 6], [5, 7], [6, 8], [7, 9],
            [8, 10], [5, 11], [6, 12], [11, 12], [11, 13], [12, 14], [13, 15], [14, 16]
        ];

        this.ctx.strokeStyle = 'white';
        this.ctx.lineWidth = 2;

        adjacentKeyPoints.forEach(([i, j]) => {
            const kp1 = keypoints[i];
            const kp2 = keypoints[j];
            
            if (kp1.score > 0.3 && kp2.score > 0.3) {
                this.ctx.beginPath();
                this.ctx.moveTo(kp1.x, kp1.y);
                this.ctx.lineTo(kp2.x, kp2.y);
                this.ctx.stroke();
            }
        });
    }

    analyzePoses(poses) {
        if (poses.length === 0) return;
        
        const pose = poses[0];
        const keypoints = pose.keypoints;
        const threshold = this.thresholds[this.exerciseType];
        
        if (this.exerciseType === 'pushups') {
            this.analyzePushups(keypoints, threshold);
        } else if (this.exerciseType === 'squats') {
            this.analyzeSquats(keypoints, threshold);
        }
    }

    analyzePushups(keypoints, threshold) {
        const leftShoulder = keypoints[11];
        const leftElbow = keypoints[13];
        const leftWrist = keypoints[15];
        
        const rightShoulder = keypoints[12];
        const rightElbow = keypoints[14];
        const rightWrist = keypoints[16];
        
        if (leftShoulder.score > threshold.minConfidence && 
            leftElbow.score > threshold.minConfidence && 
            leftWrist.score > threshold.minConfidence) {
            
            const angle = this.calculateAngle(
                {x: leftShoulder.x, y: leftShoulder.y},
                {x: leftElbow.x, y: leftElbow.y},
                {x: leftWrist.x, y: leftWrist.y}
            );
            
            this.updateCountBasedOnAngle(angle, threshold.angle);
        }
    }

    analyzeSquats(keypoints, threshold) {
        const leftHip = keypoints[23];
        const leftKnee = keypoints[25];
        const leftAnkle = keypoints[27];
        
        const rightHip = keypoints[24];
        const rightKnee = keypoints[26];
        const rightAnkle = keypoints[28];
        
        if (leftHip.score > threshold.minConfidence && 
            leftKnee.score > threshold.minConfidence && 
            leftAnkle.score > threshold.minConfidence) {
            
            const angle = this.calculateAngle(
                {x: leftHip.x, y: leftHip.y},
                {x: leftKnee.x, y: leftKnee.y},
                {x: leftAnkle.x, y: leftAnkle.y}
            );
            
            this.updateCountBasedOnAngle(angle, threshold.angle);
        }
    }

    calculateAngle(a, b, c) {
        const ab = Math.sqrt(Math.pow(b.x - a.x, 2) + Math.pow(b.y - a.y, 2));
        const bc = Math.sqrt(Math.pow(b.x - c.x, 2) + Math.pow(b.y - c.y, 2));
        const ac = Math.sqrt(Math.pow(c.x - a.x, 2) + Math.pow(c.y - a.y, 2));
        
        const angle = Math.acos((bc * bc + ab * ab - ac * ac) / (2 * bc * ab)) * (180 / Math.PI);
        return angle;
    }

    updateCountBasedOnAngle(angle, threshold) {
        if (this.currentState === 'up' && angle < threshold) {
            this.currentState = 'down';
        } else if (this.currentState === 'down' && angle > threshold + 20) {
            this.currentState = 'up';
            this.exerciseCount++;
            this.updateUI();
        }
    }

    setExerciseType(type) {
        this.exerciseType = type;
        this.exerciseCount = 0;
        this.currentState = 'up';
        document.getElementById('exercise-type').textContent = `Current Exercise: ${type.toUpperCase()}`;
        this.updateUI();
    }

    resetCount() {
        this.exerciseCount = 0;
        this.updateUI();
    }

    updateUI() {
        document.getElementById('exercise-count').textContent = `Count: ${this.exerciseCount}`;
    }

    showStatus(message, type) {
        const statusElement = document.getElementById('status');
        statusElement.textContent = message;
        statusElement.className = `status ${type}`;
        statusElement.style.display = 'block';
        
        if (type !== 'error') {
            setTimeout(() => {
                statusElement.style.display = 'none';
            }, 3000);
        }
    }

    async toggleCamera() {
        if (this.uploadedVideo) {
            this.analysisInProgress = false;
            this.uploadedVideo = false;
            this.videoElement.src = '';
            this.videoElement.srcObject = null;
        }
        
        this.usingFrontCamera = !this.usingFrontCamera;
        await this.setupCamera();
    }
}

// Initialize the app when the page loads
window.addEventListener('DOMContentLoaded', () => {
    const exerciseCounter = new ExerciseCounter('input-video', 'output-canvas');
    window.exerciseCounter = exerciseCounter;
    
    // Global functions for button controls
    window.changeExerciseType = (type) => exerciseCounter.setExerciseType(type);
    window.resetCount = () => exerciseCounter.resetCount();
    window.toggleCamera = () => exerciseCounter.toggleCamera();
    
    exerciseCounter.initialize();
});