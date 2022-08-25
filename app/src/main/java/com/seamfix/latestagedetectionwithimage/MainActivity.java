package com.seamfix.latestagedetectionwithimage;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.widget.Toast;
import androidx.appcompat.app.AppCompatActivity;
import com.seamfix.latestagedetectionwithimage.databinding.ActivityMainBinding;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import java.io.BufferedInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

public class MainActivity extends AppCompatActivity {

    private static String TAG = "mainActivity";

    //Age Recognition Areas
    private Net mAgeNet;
    private static final String[] AGES = new String[]{"0-2", "4-6", "8-13", "15-20", "25-32", "38-43", "48-53", "60+"};

    //Setting up detection area
    private Size m65Size = new Size(65, 65);
    private Size mDefault = new Size();

    private static final String[] GENDERS = new String[]{"male", "female"};
    private Rect[] mFrontalFacesArray;

    private Mat mat;
    private Rect rect;
    private ActivityMainBinding activityMainBinding;
    private CascadeClassifier mFrontalFaceClassifier = null; //Front Face Cascade Classifier

    //check opencv loaded or not
    static {
        if (OpenCVLoader.initDebug()) {
            Log.d(TAG, "static initializer: OpenCV loaded");
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        activityMainBinding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(activityMainBinding.getRoot());
        //loading models
        loadModels();

        activityMainBinding.btnPickImage.setOnClickListener(v -> {
            activityMainBinding.txvResult.setText("0");
            pickimage();
        });

        activityMainBinding.btnEstimateAge.setOnClickListener(v -> {
            String ageResult = analyseAge(mat, rect);
            activityMainBinding.txvResult.setText(ageResult);
            Log.d(TAG, "onCreate: " + ageResult);
        });

        activityMainBinding.btnDetectFace.setOnClickListener(v->{
            if(mat != null && rect !=null){
                drawRectangle(mat,rect);
            }else{
                Toast.makeText(this, "Please pick an image.", Toast.LENGTH_SHORT).show();
            }

        });
    }

    private void pickimage() {
        Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
        intent.setType("image/*");
        startActivityForResult(intent, 10);
    }

    private void loadModels() {
        //Loading age module
        String proto = getPath("deploy_age.prototxt");
        String weights = getPath("age_net.caffemodel");
        Log.i(TAG, "onCameraViewStarted| ageProto : " + proto + ",ageWeights : " + weights);
        mAgeNet = Dnn.readNetFromCaffe(proto, weights);

        if (mAgeNet.empty()) {
            Log.i(TAG, "Network loading failed");
        } else {
            Log.i(TAG, "Network loading success");
        }

        initFrontalFace();
    }

    private String analyseAge(Mat mRgba, Rect face) {
        try {
            Mat capturedFace = new Mat(mRgba, face);
            //Resizing pictures to resolution of Caffe model
            Imgproc.resize(capturedFace, capturedFace, new Size(227, 227));
            //Converting RGBA to BGR
            Imgproc.cvtColor(capturedFace, capturedFace, Imgproc.COLOR_RGBA2BGR);

            //Forwarding picture through Dnn
            Mat inputBlob = Dnn.blobFromImage(capturedFace, 1.0f, new Size(227, 227),
                    new Scalar(78.4263377603, 87.7689143744, 114.895847746), false, false);
            mAgeNet.setInput(inputBlob, "data");
            Mat probs = mAgeNet.forward("prob").reshape(1, 1);
            Core.MinMaxLocResult mm = Core.minMaxLoc(probs); //Getting largest softmax output

            double result = mm.maxLoc.x; //Result of age recognition prediction
            Log.i(TAG, "Result is: " + result);
            return AGES[(int) result];
        } catch (Exception e) {
            Log.e(TAG, "Error processing age", e);
        }
        return null;
    }

    private String getPath(String file) {
        AssetManager assetManager = getApplicationContext().getAssets();
        BufferedInputStream inputStream;

        try {
            //Reading data from app/src/main/assets
            inputStream = new BufferedInputStream(assetManager.open(file));
            byte[] data = new byte[inputStream.available()];
            inputStream.read(data);
            inputStream.close();

            File outputFile = new File(getApplicationContext().getFilesDir(), file);
            FileOutputStream fileOutputStream = new FileOutputStream(outputFile);
            fileOutputStream.write(data);
            fileOutputStream.close();
            return outputFile.getAbsolutePath();
        } catch (IOException ex) {
            Log.e(TAG, ex.toString());
        }
        return "";
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == 10 && resultCode == Activity.RESULT_OK) {
            Uri imageURI = data.getData();
            Bitmap bitmap = null;
            try {
                bitmap = MediaStore.Images.Media.getBitmap(getContentResolver(), imageURI);
            } catch (IOException e) {
                e.printStackTrace();
            }
            activityMainBinding.imgSample.setImageBitmap(bitmap);

            ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
            bitmap.compress(Bitmap.CompressFormat.JPEG, 100, byteArrayOutputStream);
            byte[] byteArray = byteArrayOutputStream.toByteArray();
           // imagePath = Base64.encodeToString(byteArray, Base64.NO_WRAP);

            //convert bitmap to mat
            Mat mat1 = new Mat();
            Utils.bitmapToMat(bitmap, mat1);
            mat = mat1;

            //finding face area on image
            //rect = new Rect(20, 0, 50, 70);
            rect = detectFaceArea(mat);

        }
    }

    private Rect detectFaceArea(Mat matt) {

        //Detection and display
        MatOfRect frontalFaces = new MatOfRect();

        if (mFrontalFaceClassifier != null) {//Here, two Size are used to detect faces. The smaller the size, the farther the detection distance is. Four parameters, 1.1, 5, 2, m65Size and mDefault, can improve the accuracy of detection. Five are confirmed five times. Baidu MultiScale is a specific method.
            mFrontalFaceClassifier.detectMultiScale(matt, frontalFaces, 1.1, 5, 2, m65Size, mDefault);
            mFrontalFacesArray = frontalFaces.toArray();
            if (mFrontalFacesArray.length > 0) {
                Log.i(TAG, "The number of faces is : " + mFrontalFacesArray.length);
            }
        }

        if(mFrontalFacesArray.length>0){
            return mFrontalFacesArray[0];
        }else{
            Toast.makeText(this, "Sorry, No face found. Please pick an image with face", Toast.LENGTH_LONG).show();
        }
       return null;
    }

    private void drawRectangle(Mat mat, Rect rect) {
        Imgproc.rectangle(mat, new Point( rect.x, rect.y),
               new Point((rect.x + rect.width), (rect.y + rect.height)), new Scalar(0,255.0,0), 5);

    }

    private void initFrontalFace() {
        try {
            //This model is relatively good for me.
            InputStream is = getResources().openRawResource(R.raw.haarcascade_frontalface_alt); //OpenCV Face Model File: lbpcascade_frontalface_improved
            File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
            File mCascadeFile = new File(cascadeDir, "haarcascade_frontalface_alt.xml");
            FileOutputStream os = new FileOutputStream(mCascadeFile);
            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
            is.close();
            os.close();
            // Loading Face Classifier
            mFrontalFaceClassifier = new CascadeClassifier(mCascadeFile.getAbsolutePath());
        } catch (Exception e) {
            Log.e(TAG, e.toString());
        }
    }
}