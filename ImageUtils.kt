package com.ariateam.ariacamtest.util

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.PixelFormat
import android.graphics.Rect
import android.graphics.YuvImage
import androidx.camera.core.ImageProxy
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer

object ImageUtils {

    /**
     * Color palette used for visualizing different segmentation class IDs.
     */
    val CLASS_COLORS = intArrayOf(
        Color.TRANSPARENT,           // 0: Background / Unlabeled (Transparent)
        Color.argb(180, 255, 57, 57),  // 1: Red
        Color.argb(180, 57, 255, 57),  // 2: Green
        Color.argb(180, 57, 57, 255),  // 3: Blue
        Color.argb(180, 255, 255, 57), // 4: Yellow
        Color.argb(180, 255, 57, 255), // 5: Magenta
        Color.argb(180, 57, 255, 255), // 6: Cyan
        Color.argb(180, 255, 153, 51), // 7: Orange
        Color.argb(180, 153, 51, 255), // 8: Purple
        Color.argb(180, 255, 102, 178),// 9: Pink
        Color.argb(180, 102, 255, 178) // 10: Teal
    )

    /**
     * Converts CameraX ImageProxy to a Bitmap cropped to the target aspect ratio (bottom-aligned) and scaled to target width and height (512x288).
     */
    fun imageProxyToBitmap(
        imageProxy: ImageProxy,
        targetWidth: Int = 512,
        targetHeight: Int = 288,
        isFrontCamera: Boolean = false
    ): Bitmap? {
        val bitmap = when (imageProxy.format) {
            ImageFormat.YUV_420_888 -> yuv420ToBitmap(imageProxy)
            PixelFormat.RGBA_8888, ImageFormat.FLEX_RGBA_8888 -> rgbaToBitmap(imageProxy)
            else -> yuv420ToBitmap(imageProxy) // Fallback
        } ?: return null

        val rotation = imageProxy.imageInfo.rotationDegrees
        val matrix = Matrix().apply {
            postRotate(rotation.toFloat())
            if (isFrontCamera) {
                postScale(-1f, 1f, bitmap.width / 2f, bitmap.height / 2f)
            }
        }

        val rotatedBitmap = Bitmap.createBitmap(
            bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true
        )

        // 목표 비율(targetWidth:targetHeight)로 잘라서 찌그러짐 없이 맞춥니다.
        // 세로가 남으면 위쪽을 버리고 아래쪽(길이 있는 쪽)을 남깁니다.
        val srcWidth = rotatedBitmap.width
        val srcHeight = rotatedBitmap.height
        val targetAspect = targetWidth.toFloat() / targetHeight.toFloat()
        var cropW = srcWidth
        var cropH = (srcWidth / targetAspect).toInt()
        if (cropH > srcHeight) {
            cropH = srcHeight
            cropW = (srcHeight * targetAspect).toInt()
        }
        val startX = (srcWidth - cropW) / 2
        val startY = srcHeight - cropH

        val croppedBitmap = if (cropW == srcWidth && cropH == srcHeight) {
            rotatedBitmap
        } else {
            Bitmap.createBitmap(rotatedBitmap, startX, startY, cropW, cropH)
        }

        val scaledBitmap = Bitmap.createScaledBitmap(croppedBitmap, targetWidth, targetHeight, true)

        if (croppedBitmap != rotatedBitmap && !croppedBitmap.isRecycled) {
            croppedBitmap.recycle()
        }
        if (rotatedBitmap != bitmap && !rotatedBitmap.isRecycled) {
            rotatedBitmap.recycle()
        }
        if (!bitmap.isRecycled) {
            bitmap.recycle()
        }

        return scaledBitmap
    }

    private fun yuv420ToBitmap(image: ImageProxy): Bitmap? {
        val yBuffer = image.planes[0].buffer
        val uBuffer = image.planes[1].buffer
        val vBuffer = image.planes[2].buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)

        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = YuvImage(
            nv21,
            ImageFormat.NV21,
            image.width,
            image.height,
            null
        )

        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, image.width, image.height), 100, out)
        val imageBytes = out.toByteArray()
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
    }

    private fun rgbaToBitmap(image: ImageProxy): Bitmap? {
        val buffer = image.planes[0].buffer
        val bitmap = Bitmap.createBitmap(image.width, image.height, Bitmap.Config.ARGB_8888)
        buffer.rewind()
        bitmap.copyPixelsFromBuffer(buffer)
        return bitmap
    }

    /**
     * Converts a 2D class index array (height x width) to an ARGB mask Bitmap.
     */
    fun createMaskBitmap(
        classIndices: Array<IntArray>,
        width: Int,
        height: Int
    ): Bitmap {
        val pixels = IntArray(width * height)
        for (y in 0 until height) {
            for (x in 0 until width) {
                val classIdx = classIndices[y][x]
                val color = if (classIdx in CLASS_COLORS.indices) {
                    CLASS_COLORS[classIdx]
                } else {
                    // Fallback color for unexpected class index
                    Color.argb(180, (classIdx * 45) % 256, (classIdx * 85) % 256, (classIdx * 125) % 256)
                }
                pixels[y * width + x] = color
            }
        }
        return Bitmap.createBitmap(pixels, width, height, Bitmap.Config.ARGB_8888)
    }
}
