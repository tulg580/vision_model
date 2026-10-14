import android.content.Context
import android.graphics.Bitmap
import com.google.ai.edge.litert.Interpreter   // 예전 TFLite 의존성이면 org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class SegResult(
    /** [왼쪽, 가운데, 오른쪽] 보행 가능(인도+횡단보도) 비율, 0~1 */
    val free: FloatArray,
    /** 가운데 영역에서 막힌 픽셀이 가장 많은 클래스 번호 (없으면 -1) */
    val blockedBy: Int
) {
    val blockedByName: String
        get() = WalkableSegmenter.NAMES.getOrElse(blockedBy) { "-" }
}

class WalkableSegmenter(context: Context) {

    companion object {
        // 학습 때 쓴 클래스 순서와 같아야 합니다
        val NAMES = arrayOf("other", "road", "sidewalk", "crosswalk", "person", "vehicle", "pole_sign")
        private const val IN_W = 512
        private const val IN_H = 224
        private const val NC = 7
    }

    private val interpreter: Interpreter
    private val inBuf = ByteBuffer.allocateDirect(4 * IN_W * IN_H * 3).order(ByteOrder.nativeOrder())
    private val outBuf = ByteBuffer.allocateDirect(4 * IN_H * IN_W * NC).order(ByteOrder.nativeOrder())
    private val pixels = IntArray(IN_W * IN_H)

    init {
        interpreter = Interpreter(
            loadModel(context, "walkable_seg_multi.tflite"),
            Interpreter.Options().setNumThreads(4)
        )
    }

    private fun loadModel(context: Context, name: String): MappedByteBuffer =
        context.assets.openFd(name).use { fd ->
            FileInputStream(fd.fileDescriptor).channel.map(
                FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength
            )
        }

    fun analyze(frame: Bitmap): SegResult {
        // 입력: 512x224, RGB, 0~255 float
        val resized = Bitmap.createScaledBitmap(frame, IN_W, IN_H, true)
        resized.getPixels(pixels, 0, IN_W, 0, 0, IN_W, IN_H)
        inBuf.rewind()
        for (p in pixels) {
            inBuf.putFloat(((p shr 16) and 0xFF).toFloat())
            inBuf.putFloat(((p shr 8) and 0xFF).toFloat())
            inBuf.putFloat((p and 0xFF).toFloat())
        }
        inBuf.rewind()
        outBuf.rewind()

        interpreter.run(inBuf, outBuf)      // 출력: [1, 224, 512, 7] 클래스별 확률

        outBuf.rewind()
        val probs = outBuf.asFloatBuffer()

        val free = FloatArray(3)
        val total = IntArray(3)
        val blocked = IntArray(NC)

        for (y in (IN_H * 0.55f).toInt() until IN_H) {          // 화면 아래쪽 = 내 앞길
            for (x in 0 until IN_W) {
                val base = (y * IN_W + x) * NC
                var best = 0
                var bestV = probs.get(base)
                for (c in 1 until NC) {
                    val v = probs.get(base + c)
                    if (v > bestV) { bestV = v; best = c }
                }
                val zone = x * 3 / IN_W                          // 0=왼쪽, 1=가운데, 2=오른쪽
                total[zone]++
                if (best == 2 || best == 3) free[zone] += 1f     // 인도, 횡단보도만 보행 가능
                else if (zone == 1) blocked[best]++
            }
        }

        val blockedBy = blocked.indices.maxByOrNull { blocked[it] }
            ?.takeIf { blocked[it] > 0 } ?: -1
        return SegResult(FloatArray(3) { free[it] / total[it] }, blockedBy)
    }
}

// 사용 예:
//   val r = segmenter.analyze(bitmap)
//   val obstacle = r.free[1] < 0.6f          // 가운데 보행 가능 비율이 낮으면 장애물
//   Log.d("seg", "L=${r.free[0]} C=${r.free[1]} R=${r.free[2]} blockedBy=${r.blockedByName}")
