package com.example.camrecorder

import android.graphics.Matrix
import android.util.Log
import android.util.Size
import androidx.camera.core.*
import java.io.File
import java.nio.ByteBuffer
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.ExecutorService
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import kotlinx.android.synthetic.main.activity_main.*
import androidx.core.content.ContextCompat
import androidx.core.graphics.scaleMatrix
import androidx.lifecycle.LifecycleOwner


data class CameraConfiguration(val save_image:Boolean, val show_image:Boolean, val process_image:Boolean,val contextCamera:android.content.Context,val activity: LifecycleOwner)  {

     val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA //Only need this shit


    //Need to give if I want to capture
      lateinit var outputDirectory: File
    //ned to give if I want  to view
      lateinit var cameraPreviewSurface:Preview.SurfaceProvider

    // need to give if I want to process
      lateinit var cameraExecutor: ExecutorService
      lateinit var analyzer:ImageAnalysis.Analyzer
      var rotation : Int = 0

    public fun ConfigView(surface:Preview.SurfaceProvider){
        cameraPreviewSurface  =surface
    }

    public fun ConfigCapture(f:File){
        outputDirectory = f
    }

    public fun ConfigProcess(analyzerObj: ImageAnalysis.Analyzer, cameraExecutorObj: ExecutorService)
    {
        analyzer = analyzerObj
        cameraExecutor = cameraExecutorObj
    }
}
class MyCamera(val config:CameraConfiguration) {

    var imageCapture: ImageCapture? = null
    var preview:Preview?=null
    var imageAnalyzer : ImageAnalysis?=null


    companion object {
        private const val TAG = "CameraXBasic"
        private const val FILENAME_FORMAT = "yyyy-MM-dd-HH-mm-ss-SSS"
    }

    public fun initCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(config.contextCamera)

        cameraProviderFuture.addListener(Runnable {
            // Used to bind the lifecycle of cameras to the lifecycle owner
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            if(config.show_image) { // I need this just to view
                preview = Preview.Builder()
                    .build()
                    .also {
                        it.setSurfaceProvider(config.cameraPreviewSurface)
                    }
            }

            if(config.save_image) { // I need to also save some images
                imageCapture = ImageCapture.Builder()
                    .setFlashMode(ImageCapture.FLASH_MODE_OFF)
                    .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                    .setTargetResolution(Size(640,480))
                    .build()
            }

            if(config.process_image) { // I need to process the images as well
                imageAnalyzer = ImageAnalysis.Builder()
                        .setTargetResolution(Size(480,640))
                        //.setTargetRotation(rot)
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        //.setImageQueueDepth(2)
                        .build()
                        .also {
                            it.setAnalyzer(config.cameraExecutor, config.analyzer)
                        }
            }

            try {
                // Unbind use cases before rebinding
                cameraProvider.unbindAll()
                // Bind use cases to camera
                cameraProvider.bindToLifecycle(
                    config.activity, config.cameraSelector, preview, imageCapture, imageAnalyzer)

            } catch(exc: Exception) {
                null
            }

        }, ContextCompat.getMainExecutor(config.contextCamera))


    }

    fun runCamera() {
        takePhoto()
    }

    private fun takePhoto() {
        // Get a stable reference of the modifiable image capture use case
        val imageCapture = imageCapture ?: return

        // Create time-stamped output file to hold the image
        val photoFile = File(
            config.outputDirectory,
            SimpleDateFormat(MyCamera.FILENAME_FORMAT, Locale.US
            ).format(System.currentTimeMillis()) + ".jpg")

        // Create output options object which contains file + metadata
        val outputOptions = ImageCapture.OutputFileOptions.Builder(photoFile).build()

        // Set up image capture listener, which is triggered after photo has
        // been taken
        /*imageCapture.takePicture(
            outputOptions, ContextCompat.getMainExecutor(config.contextCamera), object : ImageCapture.OnImageSavedCallback {
                override fun onError(exc: ImageCaptureException) {
                    Log.e(MyCamera.TAG, "Photo capture failed: ${exc.message}", exc)
                }

                override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                    val savedUri = Uri.fromFile(photoFile)
                    val msg = "Photo capture succeeded: $savedUri"
                    Toast.makeText(config.contextCamera, msg, Toast.LENGTH_SHORT).show()
                    Log.d(MyCamera.TAG, msg)
                }
            })*/

        imageCapture.takePicture(ContextCompat.getMainExecutor(config.contextCamera), object :
            ImageCapture.OnImageCapturedCallback() {
                override fun onError(exc: ImageCaptureException) {
                    Log.e(MyCamera.TAG, "Photo capture failed: ${exc.message}", exc)
                }

                override fun onCaptureSuccess(imageProxy: ImageProxy) {
                    println("Width = ${imageProxy.width}")
                    println("Height = ${imageProxy.height}")
                    //var buff =  imageProxy.planes[0].buffer


                    imageProxy.close()

                }
            })
    }

    fun getCorrectionMatrix(imageProxy: ImageProxy, previewView: PreviewView) : Matrix {
        val cropRect = imageProxy.cropRect
        val rotationDegrees = imageProxy.imageInfo.rotationDegrees
        val matrix = Matrix()

        // A float array of the source vertices (crop rect) in clockwise order.
        val source = floatArrayOf(
            cropRect.left.toFloat(),
            cropRect.top.toFloat(),
            cropRect.right.toFloat(),
            cropRect.top.toFloat(),
            cropRect.right.toFloat(),
            cropRect.bottom.toFloat(),
            cropRect.left.toFloat(),
            cropRect.bottom.toFloat()
        )

        // A float array of the destination vertices in clockwise order.
        val destination = floatArrayOf(
            0f,
            0f,
            previewView.width.toFloat(),
            0f,
            previewView.width.toFloat(),
            previewView.height.toFloat(),
            0f,
            previewView.height.toFloat()
        )

        // The destination vertexes need to be shifted based on rotation degrees. The
        // rotation degree represents the clockwise rotation needed to correct the image.

        // Each vertex is represented by 2 float numbers in the vertices array.
        val vertexSize = 2
        // The destination needs to be shifted 1 vertex for every 90° rotation.
        val shiftOffset = rotationDegrees / 90 * vertexSize;
        val tempArray = destination.clone()
        for (toIndex in source.indices) {
            val fromIndex = (toIndex + shiftOffset) % source.size
            destination[toIndex] = tempArray[fromIndex]
        }
        matrix.setPolyToPoly(source, 0, destination, 0, 4)
        return matrix
    }

}