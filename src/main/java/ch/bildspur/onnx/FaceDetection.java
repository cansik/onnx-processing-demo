package ch.bildspur.onnx;

import processing.core.PVector;

public class FaceDetection {
    // Bounding box
    public PVector center;
    public PVector extent;

    // Key points
    public PVector leftEye;
    public PVector rightEye;
    public PVector nose;
    public PVector mouth;
    public PVector leftEar;
    public PVector rightEar;

    // Confidence score [0, 1]
    public float score;
}
