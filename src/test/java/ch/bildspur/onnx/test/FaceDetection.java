package ch.bildspur.onnx.test;

import ch.bildspur.onnx.MediaPipeFaceDetection;
import ch.bildspur.onnx.Midas;
import processing.core.PApplet;
import processing.core.PImage;

public class FaceDetection extends PApplet {

    public static void main(String... args) {
        FaceDetection sketch = new FaceDetection();
        sketch.runSketch();
    }

    public void settings() {
        size(640, 480);
    }

    PImage input;

    MediaPipeFaceDetection faceDetection;

    public void setup() {
        colorMode(HSB, 360, 100, 100);
        input = loadImage("assets/elevate-nYgy58eb9aw-unsplash.jpg");

        faceDetection = new MediaPipeFaceDetection("assets/face_detection_back_256x256.onnx");
        faceDetection.predict(input);
    }

    public void draw() {
        background(55);
        image(input, 0, 0);

        surface.setTitle("Face Detection - FPS: " + Math.round(frameRate));
    }
}