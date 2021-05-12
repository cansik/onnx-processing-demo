package ch.bildspur.onnx;

import ai.onnxruntime.*;
import processing.core.PImage;
import processing.core.PVector;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class MediaPipeFaceDetection {

    private OrtEnvironment env = OrtEnvironment.getEnvironment();
    private OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
    private OrtSession session;

    private float threshold = 0.5f;

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
            }

        } catch (OrtException e) {
            e.printStackTrace();
        }
    }

    public List<FaceDetection> predict(PImage image) {
        List<FaceDetection> results = new ArrayList<>();
        float[][][][] inputData = new float[1][256][256][3];
        imageToTensor(image, inputData);

        try {
            OnnxTensor test = OnnxTensor.createTensor(env, inputData);
            OrtSession.Result output = session.run(Collections.singletonMap("input:0", test));

            // get outputs
            float[][][] scores1 = (float[][][]) output.get(0).getValue(); // 512
            float[][][] scores2 = (float[][][]) output.get(1).getValue(); // 384
            float[][][] boxes1 = (float[][][]) output.get(2).getValue(); // 512
            float[][][] boxes2 = (float[][][]) output.get(3).getValue(); // 384

            // analyse score
            for (int i = 0; i < scores1[0].length; i++) {
                float score = scores1[0][i][0];
                if (score >= threshold) {
                    int ptr = 0;

                    // post process anchors

                    // create new result
                    FaceDetection result = new FaceDetection();
                    result.score = score;
                    result.center = new PVector(boxes1[0][i][ptr++], boxes1[0][i][ptr++]);

                    results.add(result);
                }
            }

        } catch (OrtException e) {
            e.printStackTrace();
        }

        return results;
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
                tensor[0][y][x][0] = ((pixel >> 16) & 0xff) / 255.0f;
                tensor[0][y][x][1] = ((pixel >> 8) & 0xff) / 255.0f;
                tensor[0][y][x][2] = ((pixel) & 0xff) / 255.0f;
            }
        }
    }

    public float getThreshold() {
        return threshold;
    }

    public void setThreshold(float threshold) {
        this.threshold = threshold;
    }
}
