package ch.bildspur.onnx;

import ai.onnxruntime.*;
import processing.core.PApplet;
import processing.core.PConstants;
import processing.core.PImage;

import java.util.Collections;

public class Midas {
    private OrtEnvironment env = OrtEnvironment.getEnvironment();
    private OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
    private OrtSession session;

    PImage result = new PImage(256, 256, PConstants.RGB);

    public Midas(String modelPath) {
        try {
            opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT);
            session = env.createSession(modelPath, opts);

            System.out.println("Inputs:");
            for (NodeInfo i : session.getInputInfo().values()) {
                System.out.println(i.toString());
            }

            System.out.println("Outputs:");
            for (NodeInfo i : session.getOutputInfo().values()) {
                System.out.println(i.toString());
            }

        } catch (OrtException e) {
            e.printStackTrace();
        }
    }

    public PImage predict(PImage image) {
        float[][][][] inputData = new float[1][3][256][256];
        imageToTensor(image, inputData);

        try {
            OnnxTensor test = OnnxTensor.createTensor(env, inputData);
            OrtSession.Result output = session.run(Collections.singletonMap("0", test));

            // get first output
            float[][][] data = (float[][][]) output.get(0).getValue();
            tensorToImage(data, result);
        } catch (OrtException e) {
            e.printStackTrace();
        }

        return result;
    }

    private void tensorToImage(float[][][] tensor, PImage image) {
        for(int y = 0; y < image.height; y++) {
            for (int x = 0; x < image.width; x++) {
                float value = tensor[0][y][x];
                int loc = x + y * image.width;
                image.pixels[loc] = (int)Math.round(value / 1000.0 * 255);
            }
        }
    }

    private void imageToTensor(PImage image, float[][][][] tensor) {
        // resize image
        PImage input = image.copy();
        input.resize(256, 256);

        int[] pixels = input.pixels;

        for(int y = 0; y < input.height; y++) {
            for(int x = 0; x < input.width; x++) {
                int loc = x + y * input.width;
                int pixel = pixels[loc];

                // extract R G B
                // todo: check if it should be BGR
                tensor[0][0][y][x] = pixel & 0xFF;
                tensor[0][1][y][x] = pixel >> 8 & 0xFF;
                tensor[0][2][y][x] = pixel >> 16 & 0xFF;
            }
        }
    }
}
