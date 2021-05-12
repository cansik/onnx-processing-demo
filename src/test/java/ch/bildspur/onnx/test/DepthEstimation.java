package ch.bildspur.onnx.test;

import processing.core.PApplet;
import processing.core.PImage;

public class DepthEstimation extends PApplet {

    public static void main(String... args) {
        DepthEstimation sketch = new DepthEstimation();
        sketch.runSketch();
    }

    public void settings() {
        size(1280, 480);
    }

    PImage input;
    PImage result;

    public void setup() {
        colorMode(HSB, 360, 100, 100);
        input = loadImage("assets/carl-heyerdahl-KE0nC8-58MQ-unsplash.jpg");
    }

    public void draw() {
        background(55);
        image(input, 0, 0);
        // image(result, 640, 0);
        surface.setTitle("Depth Estimation - FPS: " + Math.round(frameRate));
    }
}