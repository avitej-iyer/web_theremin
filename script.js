// Import necessary MediaPipe components
import { HandLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";

// --- DOM ELEMENT SELECTION ---
const video = document.getElementById("webcam");
const canvasElement = document.getElementById("overlayCanvas");
const canvasCtx = canvasElement.getContext("2d");
const loadingElement = document.getElementById("loading");
const startButton = document.getElementById('start-button');
const startScreen = document.getElementById('start-screen');
const appContainer = document.getElementById('app-container');
const pitchDisplay = document.getElementById('pitch-display');
const volDisplay = document.getElementById('vol-display');
const wahDisplay = document.getElementById('wah-display');

// --- GLOBAL STATE & CONSTANTS ---
let handLandmarker;
const VIDEO_WIDTH = 640;
const VIDEO_HEIGHT = 480;
let audioContext, oscillator, gainNode, filterNode;

// --- MUSICAL SCALES ---
const SCALES = {
    pentatonic: [130.81, 146.83, 164.81, 196.00, 220.00, 261.63, 293.66, 329.63, 392.00, 440.00, 523.25, 659.25, 783.99, 880.00],
    major: [130.81, 146.83, 164.81, 174.61, 196.00, 220.00, 246.94, 261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88],
    minor: [130.81, 146.83, 155.56, 174.61, 196.00, 207.65, 233.08, 261.63, 293.66, 311.13, 349.23, 392.00, 415.30, 466.16],
    blues: [130.81, 155.56, 174.61, 185.00, 196.00, 233.08, 261.63, 311.13, 349.23, 369.99, 392.00, 466.16]
};

// --- MAIN START FUNCTION ---
async function start() {
    startScreen.classList.add('hidden');
    appContainer.classList.remove('hidden');
    setupWebAudio();
    if (audioContext.state === 'suspended') {
        audioContext.resume();
    }
    await setupMediaPipe();
    await setupWebcam();
    setupControls();
    loadingElement.style.display = 'none';
    predictWebcam();
}

// --- INITIALIZATION ---
function setupWebAudio() {
    if (audioContext) return;
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    oscillator = audioContext.createOscillator();
    gainNode = audioContext.createGain();
    filterNode = audioContext.createBiquadFilter();
    oscillator.type = 'sine';
    filterNode.type = 'lowpass';
    oscillator.connect(filterNode);
    filterNode.connect(gainNode);
    gainNode.connect(audioContext.destination);
    gainNode.gain.setValueAtTime(0, audioContext.currentTime);
    oscillator.start();
}

async function setupMediaPipe() {
    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm");
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
            delegate: "GPU"
        },
        runningMode: "VIDEO",
        numHands: 2
    });
}

async function setupWebcam() {
    return new Promise((resolve) => {
        navigator.mediaDevices.getUserMedia({ video: { width: VIDEO_WIDTH, height: VIDEO_HEIGHT } })
            .then((stream) => {
                video.srcObject = stream;
                video.addEventListener("loadeddata", () => {
                    canvasElement.width = video.videoWidth;
                    canvasElement.height = video.videoHeight;
                    resolve();
                });
            });
    });
}

function setupControls() {
    document.getElementById('waveform-select').addEventListener('change', (e) => {
        if (oscillator) oscillator.type = e.target.value;
    });
}

// --- REAL-TIME PREDICTION & DRAWING ---
function predictWebcam() {
    if (!handLandmarker || video.readyState < 2) {
        requestAnimationFrame(predictWebcam);
        return;
    }
    const results = handLandmarker.detectForVideo(video, performance.now());
    
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    canvasCtx.scale(-1, 1);
    canvasCtx.translate(-canvasElement.width, 0);

    if (results.landmarks && results.landmarks.length > 0) {
        for (const landmarks of results.landmarks) {
            drawConnectors(canvasCtx, landmarks, HandLandmarker.HAND_CONNECTIONS, { color: '#03dac6', lineWidth: 3 });
            drawLandmarks(canvasCtx, landmarks, { color: '#e0e0e0', radius: 4 });
        }
    }

    handleMusicalControl(results);
    
    canvasCtx.restore();
    requestAnimationFrame(predictWebcam);
}


// --- THE FINAL MUSICAL CONTROL LOGIC ---
function handleMusicalControl(results) {
    let pitchHand = null;
    let volumeHand = null;
    let frequency = oscillator.frequency.value;
    let volume = 0.0;
    let isWahOn = false;
    let pitchOpenness = 0.0;

    // Use screen position for robustness
    if (results.landmarks && results.landmarks.length > 0) {
        for (const landmarks of results.landmarks) {
            const wristX = landmarks[0].x;
            if (wristX < 0.5) { // User's RIGHT hand
                pitchHand = landmarks;
            } else { // User's LEFT hand
                volumeHand = landmarks;
            }
        }
    }

    // --- PITCH (Right Hand Openness) ---
    if (pitchHand) {
        const wrist = pitchHand[0];
        const fingertips = [pitchHand[4], pitchHand[8], pitchHand[12], pitchHand[16], pitchHand[20]];
        let totalDistance = 0;
        let validFingers = 0;

        for (const tip of fingertips) {
            if (wrist && tip) {
                totalDistance += Math.hypot(wrist.x - tip.x, wrist.y - tip.y);
                validFingers++;
            }
        }
        
        if (validFingers > 0) {
            const avgDistance = totalDistance / validFingers;
            // FIX #1: Adjusted calibration values for pitch openness
            const minOpen = 0.08; // A tighter fist
            const maxOpen = 0.38; // A more reachable open hand
            pitchOpenness = Math.max(0, Math.min(1, (avgDistance - minOpen) / (maxOpen - minOpen)));

            const selectedScaleName = document.getElementById('scale-select').value;
            const scale = SCALES[selectedScaleName];
            if (scale) {
                const noteIndex = Math.floor(pitchOpenness * (scale.length - 1));
                frequency = scale[noteIndex];
            }
        }
    }

    // --- VOLUME & WAH (Left Hand) ---
    if (volumeHand) {
        const volumeY = volumeHand[0].y;
        volume = Math.max(0, (1 - volumeY) * 0.7); // Can go up to 0.7

        const thumbTip = volumeHand[4];
        const indexTip = volumeHand[8];
        if (thumbTip && indexTip) {
            const pinchDistance = Math.hypot(thumbTip.x - indexTip.x, thumbTip.y - indexTip.y);
            if (pinchDistance < 0.05) {
                isWahOn = true;
            }
        }
    } 
    // FIX #2: Cleaned up one-handed play logic
    else if (pitchHand) {
        // If ONLY the pitch hand is visible, set a default volume.
        volume = 0.5;
    }

    // --- APPLY AUDIO EFFECTS & VALUES ---
    // FIX #3: Corrected "Wah" effect implementation
    if (isWahOn) {
        // Lower the cutoff and crank the resonance for a vocal "wah" sound
        filterNode.frequency.setTargetAtTime(800, audioContext.currentTime, 0.01);
        filterNode.Q.setTargetAtTime(12, audioContext.currentTime, 0.01);
    } else {
        // Set filter to be transparent (high cutoff, low resonance)
        filterNode.frequency.setTargetAtTime(20000, audioContext.currentTime, 0.01);
        filterNode.Q.setTargetAtTime(1, audioContext.currentTime, 0.01);
    }

    if (isFinite(frequency)) oscillator.frequency.setTargetAtTime(frequency, audioContext.currentTime, 0.01);
    if (isFinite(volume)) gainNode.gain.setTargetAtTime(volume, audioContext.currentTime, 0.02);
    
    // --- UPDATE VISUAL DISPLAY ---
    pitchDisplay.innerText = pitchHand ? (pitchOpenness * 100).toFixed(0) + '%' : "---";
    volDisplay.innerText = volume.toFixed(2);
    wahDisplay.innerText = isWahOn ? "ON" : "OFF";
    wahDisplay.classList.toggle('active', isWahOn);
}

// --- DRAWING UTILITIES ---
function drawConnectors(ctx, landmarks, connections, style) {
  ctx.strokeStyle = style.color || 'white';
  ctx.lineWidth = style.lineWidth || 2;
  for (const connection of connections) {
    const start = landmarks[connection.start];
    const end = landmarks[connection.end];
    if (start && end) {
      ctx.beginPath();
      ctx.moveTo(start.x * ctx.canvas.width, start.y * ctx.canvas.height);
      ctx.lineTo(end.x * ctx.canvas.width, end.y * ctx.canvas.height);
      ctx.stroke();
    }
  }
}

function drawLandmarks(ctx, landmarks, style) {
  ctx.fillStyle = style.color || 'red';
  for (const landmark of landmarks) {
    if (landmark) {
      ctx.beginPath();
      ctx.arc(landmark.x * ctx.canvas.width, landmark.y * ctx.canvas.height, style.radius || 3, 0, 2 * Math.PI);
      ctx.fill();
    }
  }
}

// --- START THE APPLICATION ---
startButton.addEventListener('click', start);