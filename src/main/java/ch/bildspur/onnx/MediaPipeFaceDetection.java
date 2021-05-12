package ch.bildspur.onnx;

import ai.onnxruntime.*;
import processing.core.PImage;

import java.util.Collections;

public class MediaPipeFaceDetection {

    private OrtEnvironment env = OrtEnvironment.getEnvironment();
    private OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
    private OrtSession session;

    public MediaPipeFaceDetection(String modelPath) {
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
                System.out.println(i.getName());
            }

        } catch (OrtException e) {
            e.printStackTrace();
        }
    }

    public void predict(PImage image) {
        float[][][][] inputData = new float[1][256][256][3];
        imageToTensor(image, inputData);

        try {
            OnnxTensor test = OnnxTensor.createTensor(env, inputData);
            OrtSession.Result output = session.run(Collections.singletonMap("Identity:0", test));

            // get first output
            float[][][] identity = (float[][][]) output.get(0).getValue();
            System.out.println("data");
        } catch (OrtException e) {
            e.printStackTrace();
        }
    }

    private void imageToTensor(PImage image, float[][][][] tensor) {
        // resize image
        PImage input = image.copy();
        input.resize(256, 256);

        int[] pixels = input.pixels;

        for (int y = 0; y < input.height; y++) {
            for (int x = 0; x < input.width; x++) {
                int loc = x + y * input.width;
                int pixel = pixels[loc];

                // extract R G B
                // todo: check if it should be BGR
                tensor[0][y][x][0] = pixel & 0xFF;
                tensor[0][y][x][1] = pixel >> 8 & 0xFF;
                tensor[0][y][x][2] = pixel >> 16 & 0xFF;
            }
        }
    }

}
