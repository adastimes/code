package com.example.camrecorder
//https://codelabs.developers.google.com/codelabs/camerax-getting-started#3

//YUV formats
//https://user-images.githubusercontent.com/9286092/89119601-4f6f8100-d4b8-11ea-9a51-2765f7e513c2.jpg

//Example
//https://codelabs.developers.google.com/codelabs/recognize-flowers-with-tensorflow-on-android/#1
//https://heartbeat.fritz.ai/image-classification-on-android-with-tensorflow-lite-and-camerax-4f72e8fdca79
//https://www.bignerdranch.com/blog/using-firebasemlkit-with-camerax/

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.*
import android.util.Log
import android.view.OrientationEventListener
import android.view.Surface
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.Toast
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import java.util.concurrent.Executors
import androidx.camera.core.*
import androidx.camera.view.PreviewView
import androidx.core.graphics.set
import androidx.core.view.doOnPreDraw
import androidx.lifecycle.LifecycleOwner
import com.example.camrecorder.ml.PixelLabeling
import kotlinx.android.synthetic.main.activity_main.*
import kotlinx.android.synthetic.main.activity_main.view.*
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.File
import java.nio.ByteBuffer
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.ExecutorService



typealias LumaListener = (luma: Double) -> Unit
typealias MaskListener = (m: IntArray) -> Unit


typealias FrameCounter = (fn: Int) -> Unit


class MainActivity : AppCompatActivity() {

    private lateinit var outputDirectory: File
    private lateinit var cameraExecutor: ExecutorService

    private lateinit var config: CameraConfiguration
    private lateinit var cam: MyCamera


    var lastFpsTimestamp:Long=0 // used for fps measurement, holds the time for the previous frame
    var fps_previous:Double=0.0 // we run an averaging filter over instant fps to not make it jump so much
    var file_chain:Int = 0 // when we click we start again a data collection process
    private  lateinit var capture_analyzer : CaptureFrames // this is the analyzer, what runs in analyzer
    private lateinit var nn_analyzer: ProcessFrames

    private var frame_time_old:Long = 0 // the time for the previous frame saved
    private var delta:Int=10 //delta between frames
    private var start_frame:Int=100
    private var stop_frame:Int = 200

    private var capture_or_run:Boolean = true
    lateinit var overlay_img: Bitmap

    // looks like they define here in kotlin things the other way arround, as the device would be tilted
    private var width:Int=640
    private var height:Int=480

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        overlay_img = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)

        outputDirectory = getOutputDirectory()
        cameraExecutor = Executors.newSingleThreadExecutor()


        config = CameraConfiguration(true,true,true,contextCamera = baseContext,this as LifecycleOwner)
        config.ConfigCapture(outputDirectory)
        config.ConfigView(viewFinder.surfaceProvider)

        capture_analyzer = CaptureFrames (d = delta,stopf=stop_frame, startf = start_frame,file_list = file_chain,imgdir = outputDirectory,fn =0) { fn->

            var now = System.currentTimeMillis()
            var fps = 0.9*fps_previous + (1000 / (now - lastFpsTimestamp)) *0.1
            lastFpsTimestamp = now
            fps_previous = fps

            if ((fn+1 >= start_frame) and (fn+1<=stop_frame) and ((fn+1)%delta==0))
            {// this is just previously to the frame comming in, hopefully it makes no difference
                capture_analyzer.delta_time = now - frame_time_old
                if(frame_time_old==0L){
                    capture_analyzer.delta_time = 0
                }
                frame_time_old = now
            }

            runOnUiThread(Runnable {
                text_prediction.text = fps.toInt().toString()
                text_prediction.visibility = View.VISIBLE })
        }

        nn_analyzer = ProcessFrames (ctx = baseContext, d = delta,stopf=stop_frame, startf = start_frame,fn =0) { fn, mask->

            var now = System.currentTimeMillis()
            var fps = 0.9*fps_previous + (1000 / (now - lastFpsTimestamp)) *0.1
            lastFpsTimestamp = now
            fps_previous = fps

            runOnUiThread(Runnable {
                text_prediction.text = fps.toInt().toString()
                text_prediction.visibility = View.VISIBLE })


            imageView.rotation = 90F
            imageView.scaleX = viewFinder.measuredHeight.toFloat()/width.toFloat()
            imageView.scaleY = viewFinder.measuredHeight.toFloat()/width.toFloat()

            //imageView.scaleY = -(viewFinder.measuredHeight/width).toFloat()

            for(i in 0..width-1) {
                for(j in 0..height-1){
                    if(mask[j*width + i].toInt()==1) {
                        overlay_img.set(i, j, Color.RED)
                    }
                    else {
                        overlay_img.set(i, j, Color.TRANSPARENT)
                    }
                }
            }
            runOnUiThread {
                imageView.setImageBitmap(overlay_img)
            }

        }

        config.ConfigProcess( capture_analyzer, cameraExecutor)

        cam = MyCamera(config)


        // Request camera permissions
        if (allPermissionsGranted()) {
            //startCamera()
            cam.initCamera()
        } else {
            ActivityCompat.requestPermissions(
                    this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }


        camera_nn_button.setOnClickListener{
            nn_analyzer.fn = 0

            if (capture_or_run) {
                config.analyzer = nn_analyzer
                cam.initCamera()
                capture_or_run = false
            }
            //ApplyNet(baseContext) // This is a test
        }


        //plotRect()
        camera_capture_button.setOnClickListener {
            file_chain +=1
            capture_analyzer.fn = 0
            capture_analyzer.file_list = file_chain

            capture_analyzer.delta_time = 0 //reset when we click
            frame_time_old = 0 //reset when we click as we start new things

            if(!capture_or_run)
            {
                config.analyzer = capture_analyzer
                cam.initCamera()
                capture_or_run = true
            }
            //cam.runCamera()
        }

         //Set orientation, in case needed. There is an event when device changes orientation.
         val orientationEventListener = object : OrientationEventListener(this as Context) {
             override fun onOrientationChanged(orientation : Int) {
                 // Monitors orientation values to determine the target rotation value
                 val rotation : Int = when (orientation) {
                     in 45..134 -> Surface.ROTATION_90
                     in 135..224 -> Surface.ROTATION_180
                     in 225..314 -> Surface.ROTATION_90
                     else -> Surface.ROTATION_0
                 }

                 //cam.imageCapture?.targetRotation ?:  = rotation
                 //cam.imageCapture?.targetRotation = rotation
                 cam.imageAnalyzer?.targetRotation = rotation

             }
         }
         orientationEventListener.enable()

    }

    private fun ApplyNet(ctx: Context) {

        val plModel = PixelLabeling.newInstance(ctx)
        var fname = File("/storage/emulated/0/Android/media/com.example.camrecorder/CamRecorder/1_100_0_y.bin")
        var datay = fname.readBytes().toUByteArray()
        var luma = FloatArray(480*640)
        var i:Int

        for(i in datay.indices)
        {
            luma[i] = datay[i].toFloat()
        }

        val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 1, 480, 640), DataType.FLOAT32)
        inputFeature0.loadArray(luma)

        // Run model and get output
        val outputs = plModel.process(inputFeature0)
        val out = outputs.outputFeature0AsTensorBuffer.floatArray

        plModel.close()

        var mask = ByteArray(480*640)


        for(i in 0 until out.size/2) {
            mask[i] = 1
            if (out[i] > out[out.size/2 + i]) {
                mask[i] = 0
            }
        }


        var fout = File("/storage/emulated/0/Android/media/com.example.camrecorder/CamRecorder/mask.bin")
        fout.writeBytes(mask)
    }
    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(
                baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    fun getOutputDirectory(): File {
        val mediaDir = externalMediaDirs.firstOrNull()?.let {
            File(it, resources.getString(R.string.app_name)).apply { mkdirs() } }
        return if (mediaDir != null && mediaDir.exists())
            mediaDir else filesDir
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }

    override fun onRequestPermissionsResult(
            requestCode: Int, permissions: Array<String>, grantResults:
            IntArray) {
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                cam.initCamera()
            } else {
                Toast.makeText(this,
                        "Permissions not granted by the user.",
                        Toast.LENGTH_SHORT).show()
                finish()
            }
        }
    }


    companion object {
        private const val TAG = "CameraXBasic"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }

    fun plotRect() {
        // Set up the listener for take photo button
        (box_prediction.layoutParams as ViewGroup.MarginLayoutParams).apply {
            topMargin = 10
            leftMargin = 10
            width = 100
            height = 100
        }

        // Make sure all UI elements are visible
        box_prediction.visibility = View.VISIBLE
    }
    private class CaptureFrames(var d:Int,var stopf:Int,var startf:Int,var file_list:Int, var imgdir:File , var fn:Int, private val listener: (fn: Int) -> Unit) : ImageAnalysis.Analyzer {

        private val FILENAME_FORMAT = "yyyy-MM-dd-HH-mm-ss-SSS"
        var delta_time:Long=0

        private fun ByteBuffer.toByteArray(): ByteArray {
            rewind()    // Rewind the buffer to zero
            val data = ByteArray(remaining())
            get(data)   // Copy the buffer into a byte array
            return data // Return the byte array
        }


        override fun analyze(image: ImageProxy) {

            // note that this format is dubios as one can have differet the raw and pixel strides
            // other devices might really have thi thing different, it might need some adaptatio
            // layer to get the format right or just not use color, dumb beyond words
            val datay = image.planes[0].buffer.toByteArray()
            val datau = image.planes[1].buffer.toByteArray()
            val datav = image.planes[2].buffer.toByteArray()

            //println(image.planes[1].rowStride)
            //println(image.planes[1].pixelStride)

            //val pixels = data.map { it.toInt() and 0xFF }
            //val luma = pixels.average()
            fn +=1

            //println("${image.height} ${image.width}")
            //println("Rotation = ${image.imageInfo.rotationDegrees}")

            val date_time = SimpleDateFormat(FILENAME_FORMAT, Locale.US
            ).format(System.currentTimeMillis())

            if ((file_list>0) and (fn >= startf) and (fn<=stopf) and (fn%d==0)) {
                var fname = File(imgdir, file_list.toString() + "_"+fn.toString()+ "_" + delta_time.toString()+'_'+ date_time+"_y.bin")
                fname.writeBytes(datay)

                fname = File(imgdir, file_list.toString() + "_"+fn.toString()+ "_" + delta_time.toString()+'_'+date_time+"_u.bin")
                fname.writeBytes(datau)

                fname = File(imgdir, file_list.toString() + "_"+fn.toString()+ "_" + delta_time.toString()+'_'+date_time+"_v.bin")
                fname.writeBytes(datav)

            }

            listener(fn)

            image.close()
        }
    }

    private class ProcessFrames(ctx:Context,var d:Int,var stopf:Int,var startf:Int, var fn:Int, private val listener: (fn:Int, m: ByteArray) -> Unit) : ImageAnalysis.Analyzer {

        private val plModel = PixelLabeling.newInstance(ctx)

        private fun ByteBuffer.toByteArray(): ByteArray {
            rewind()    // Rewind the buffer to zero
            val data = ByteArray(remaining())
            get(data)   // Copy the buffer into a byte array
            return data // Return the byte array
        }


        override fun analyze(image: ImageProxy) {

            //prepare Input

            var datay = image.planes[0].buffer
            datay.rewind()
            var luma = FloatArray(480*640)


            var i:Int=0

            while(datay.hasRemaining())
            {
                luma.set(i,datay.get().toUByte().toFloat())
                i +=1
            }

            val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 1, 480, 640), DataType.FLOAT32)
            inputFeature0.loadArray(luma)

            // Run model and get output
            val outputs = plModel.process(inputFeature0)
            val out = outputs.outputFeature0AsTensorBuffer.floatArray

            var mask = ByteArray(480*640)

            for(i in 0 until out.size/2) {
                mask[i] = 1
                if (out[i] > out[out.size/2 + i]) {
                    mask[i] = 0
                }
            }

            fn +=1
            listener(fn,mask)

            image.close()

            //just to test if the NN does the right thing, looks like it does
            //var fout = File("/storage/emulated/0/Android/media/com.example.camrecorder/CamRecorder/mask.bin")
            //fout.writeBytes(mask)

        }
    }
}