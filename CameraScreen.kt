package com.ariateam.ariacamtest.ui

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Size
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import com.ariateam.ariacamtest.ml.LiteRtSegmenter
import com.ariateam.ariacamtest.util.ImageUtils
import java.util.concurrent.Executors

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CameraScreen() {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current

    // Permission State
    var hasCameraPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
        )
    }

    val permissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        hasCameraPermission = isGranted
    }

    LaunchedEffect(Unit) {
        if (!hasCameraPermission) {
            permissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    if (!hasCameraPermission) {
        PermissionDeniedScreen(onRequestPermission = {
            permissionLauncher.launch(Manifest.permission.CAMERA)
        })
        return
    }

    // Segmenter & Model State
    val segmenter = remember { LiteRtSegmenter() }
    var isModelLoaded by remember { mutableStateOf(segmenter.isLoaded) }
    var modelName by remember { mutableStateOf(segmenter.modelName) }
    var statusMessage by remember { mutableStateOf(segmenter.statusMessage) }

    // Camera State
    var lensFacing by remember { mutableIntStateOf(CameraSelector.LENS_FACING_BACK) }
    var availableCameraCount by remember { mutableIntStateOf(1) }

    // Inference & UI Settings
    var targetFps by remember { mutableIntStateOf(30) }
    var showSettingsSheet by remember { mutableStateOf(false) }

    // Realtime Render Frame State
    var rawCameraBitmap by remember { mutableStateOf<Bitmap?>(null) }
    var resultMaskBitmap by remember { mutableStateOf<Bitmap?>(null) }
    var lastInferenceTimeMs by remember { mutableLongStateOf(0L) }
    var actualFps by remember { mutableIntStateOf(0) }

    // FPS calculation state
    var frameCount by remember { mutableIntStateOf(0) }
    var lastFpsCalcTime by remember { mutableLongStateOf(SystemClock.uptimeMillis()) }

    val cameraExecutor = remember { Executors.newSingleThreadExecutor() }

    DisposableEffect(Unit) {
        onDispose {
            cameraExecutor.shutdown()
            segmenter.close()
        }
    }

    // Camera Binding
    LaunchedEffect(lensFacing, targetFps) {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        val cameraProvider = cameraProviderFuture.get()

        availableCameraCount = cameraProvider.availableCameraInfos.size

        val cameraSelector = CameraSelector.Builder()
            .requireLensFacing(lensFacing)
            .build()

        val resolutionSelector = ResolutionSelector.Builder()
            .setResolutionStrategy(
                ResolutionStrategy(
                    Size(640, 480),
                    ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER
                )
            )
            .build()

        val imageAnalysis = ImageAnalysis.Builder()
            .setResolutionSelector(resolutionSelector)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
            .build()

        var isProcessing = false
        var lastExecTime = 0L

        imageAnalysis.setAnalyzer(cameraExecutor) { imageProxy ->
            val now = SystemClock.uptimeMillis()
            val minInterval = 1000L / targetFps

            if (isProcessing || (now - lastExecTime) < minInterval) {
                imageProxy.close()
                return@setAnalyzer
            }

            isProcessing = true
            lastExecTime = now

            try {
                val isFront = (lensFacing == CameraSelector.LENS_FACING_FRONT)
                val bitmap = ImageUtils.imageProxyToBitmap(
                    imageProxy = imageProxy,
                    targetWidth = 512,
                    targetHeight = 288,
                    isFrontCamera = isFront
                )

                if (bitmap != null) {
                    val inferenceResult = segmenter.segment(bitmap)

                    // Calculate FPS
                    frameCount++
                    if (now - lastFpsCalcTime >= 1000L) {
                        actualFps = frameCount
                        frameCount = 0
                        lastFpsCalcTime = now
                    }

                    rawCameraBitmap = bitmap
                    resultMaskBitmap = inferenceResult.maskBitmap
                    lastInferenceTimeMs = inferenceResult.inferenceTimeMs
                    isModelLoaded = segmenter.isLoaded
                    modelName = segmenter.modelName
                    statusMessage = inferenceResult.statusMessage
                }
            } catch (e: Exception) {
                e.printStackTrace()
            } finally {
                isProcessing = false
                imageProxy.close()
            }
        }

        try {
            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                lifecycleOwner,
                cameraSelector,
                imageAnalysis
            )
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Column {
                        Text(
                            text = "LiteRT Segmentation Test",
                            fontSize = 18.sp,
                            fontWeight = FontWeight.Bold
                        )
                        Text(
                            text = "Model: $modelName | Input: 512x288",
                            fontSize = 12.sp,
                            color = Color.LightGray
                        )
                    }
                },
                actions = {
                    IconButton(onClick = { showSettingsSheet = true }) {
                        Text(text = "⚙️", fontSize = 20.sp)
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer,
                    titleContentColor = MaterialTheme.colorScheme.onPrimaryContainer
                )
            )
        },
        floatingActionButton = {
            if (availableCameraCount >= 2) {
                FloatingActionButton(
                    onClick = {
                        lensFacing = if (lensFacing == CameraSelector.LENS_FACING_BACK) {
                            CameraSelector.LENS_FACING_FRONT
                        } else {
                            CameraSelector.LENS_FACING_BACK
                        }
                    },
                    containerColor = MaterialTheme.colorScheme.primary
                ) {
                    Text(text = "🔄", fontSize = 20.sp)
                }
            }
        }
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .background(Color(0xFF121212))
                .padding(12.dp)
        ) {
            // Metrics Banner
            Surface(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(bottom = 12.dp),
                shape = RoundedCornerShape(8.dp),
                color = Color(0xFF222222)
            ) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 12.dp, vertical = 8.dp),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = "Inference: ${lastInferenceTimeMs}ms",
                        color = Color.Green,
                        fontSize = 13.sp,
                        fontWeight = FontWeight.SemiBold
                    )
                    Text(
                        text = "FPS: $actualFps / $targetFps",
                        color = Color.Cyan,
                        fontSize = 13.sp,
                        fontWeight = FontWeight.SemiBold
                    )
                    Text(
                        text = if (isModelLoaded) "Status: OK" else "Status: No Model",
                        color = if (isModelLoaded) Color.LightGray else Color.Yellow,
                        fontSize = 13.sp
                    )
                }
            }

            // Dual View Section (Raw Camera Input vs Segmentation Result)
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .weight(1f),
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                // View 1: Scaled Raw Camera Image (512x288)
                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .weight(1f),
                    colors = CardDefaults.cardColors(containerColor = Color.Black),
                    shape = RoundedCornerShape(12.dp)
                ) {
                    Box(modifier = Modifier.fillMaxSize()) {
                        val currentRaw = rawCameraBitmap
                        if (currentRaw != null) {
                            Image(
                                bitmap = currentRaw.asImageBitmap(),
                                contentDescription = "Camera Raw Input (512x288)",
                                modifier = Modifier.fillMaxSize(),
                                contentScale = ContentScale.Fit
                            )
                        } else {
                            Text(
                                text = "Initializing Camera Stream...",
                                color = Color.Gray,
                                modifier = Modifier.align(Alignment.Center)
                            )
                        }

                        // Badge Label
                        Box(
                            modifier = Modifier
                                .align(Alignment.TopStart)
                                .padding(8.dp)
                                .background(Color.Black.copy(alpha = 0.7f), RoundedCornerShape(4.dp))
                                .padding(horizontal = 8.dp, vertical = 4.dp)
                        ) {
                            Text(
                                text = "Camera Input (512x288)",
                                color = Color.White,
                                fontSize = 11.sp,
                                fontWeight = FontWeight.Bold
                            )
                        }
                    }
                }

                // View 2: Segmentation Result or Model Not Loaded
                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .weight(1f),
                    colors = CardDefaults.cardColors(containerColor = Color.Black),
                    shape = RoundedCornerShape(12.dp)
                ) {
                    Box(modifier = Modifier.fillMaxSize()) {
                        val currentMask = resultMaskBitmap
                        if (isModelLoaded && currentMask != null) {
                            Image(
                                bitmap = currentMask.asImageBitmap(),
                                contentDescription = "Semantic Segmentation Output",
                                modifier = Modifier.fillMaxSize(),
                                contentScale = ContentScale.Fit
                            )
                        } else {
                            Column(
                                modifier = Modifier.align(Alignment.Center),
                                horizontalAlignment = Alignment.CenterHorizontally
                            ) {
                                Text(
                                    text = "Model Not Loaded",
                                    color = Color.Red,
                                    fontSize = 18.sp,
                                    fontWeight = FontWeight.Bold
                                )
                                Spacer(modifier = Modifier.height(6.dp))
                                Text(
                                    text = "Please tap ⚙️ settings and load a .tflite model",
                                    color = Color.LightGray,
                                    fontSize = 12.sp,
                                    textAlign = TextAlign.Center
                                )
                            }
                        }

                        // Badge Label
                        Box(
                            modifier = Modifier
                                .align(Alignment.TopStart)
                                .padding(8.dp)
                                .background(
                                    if (isModelLoaded) Color.Blue.copy(alpha = 0.7f) else Color.Red.copy(alpha = 0.7f),
                                    RoundedCornerShape(4.dp)
                                )
                                .padding(horizontal = 8.dp, vertical = 4.dp)
                        ) {
                            Text(
                                text = if (isModelLoaded) "Segmentation Output" else "No Model",
                                color = Color.White,
                                fontSize = 11.sp,
                                fontWeight = FontWeight.Bold
                            )
                        }
                    }
                }
            }
        }

        if (showSettingsSheet) {
            SettingsSheet(
                modelName = modelName,
                isModelLoaded = isModelLoaded,
                statusMessage = statusMessage,
                targetFps = targetFps,
                onModelSelected = { uri, name ->
                    val success = segmenter.loadModelFromUri(context, uri, name)
                    isModelLoaded = segmenter.isLoaded
                    modelName = segmenter.modelName
                    statusMessage = segmenter.statusMessage
                },
                onTargetFpsChanged = { fps ->
                    targetFps = fps
                },
                onDismiss = { showSettingsSheet = false }
            )
        }
    }
}

@Composable
fun PermissionDeniedScreen(onRequestPermission: () -> Unit) {
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.Black),
        contentAlignment = Alignment.Center
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            modifier = Modifier.padding(24.dp)
        ) {
            Text(
                text = "📷 Camera Permission Required",
                color = Color.White,
                fontSize = 20.sp,
                fontWeight = FontWeight.Bold
            )
            Spacer(modifier = Modifier.height(12.dp))
            Text(
                text = "To test real-time semantic segmentation, camera access is required.",
                color = Color.LightGray,
                textAlign = TextAlign.Center
            )
            Spacer(modifier = Modifier.height(24.dp))
            Button(onClick = onRequestPermission) {
                Text(text = "Grant Permission")
            }
        }
    }
}
