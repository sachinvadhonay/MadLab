import androidx.appcompat.app.AppCompatActivity;
import android.app.WallpaperManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.drawable.AnimationDrawable;
import android.graphics.drawable.BitmapDrawable;
import android.graphics.drawable.Drawable;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.Toast;
import java.io.IOException;
import java.util.Timer;
import java.util.TimerTask;
public class MainActivity extends AppCompatActivity {
 Button changewallpaper;
 Timer mytimer;
 Drawable drawable;
 WallpaperManager wpm;
 int prev=1;
 @Override
 protected void onCreate(Bundle savedInstanceState) {
 super.onCreate(savedInstanceState);
 setContentView(R.layout.activity_main);
 mytimer = new Timer();
 wpm = WallpaperManager.getInstance(this);
 changewallpaper = findViewById(R.id.button);
 changewallpaper.setOnClickListener(new
View.OnClickListener() {
 @Override public void onClick(View view) {
Mobile Application Development Laboratory – 18CSMP68
Department of Computer Science & Engineering Page 32
 setWallpaper();
 }
 });
 }
 private void setWallpaper() {
 mytimer.schedule(new TimerTask() {
 @Override
 public void run() {
 if(prev==1) {
 drawable =
getResources().getDrawable(R.drawable.one);
 prev = 2;
 }
 else if(prev==2) {
 drawable =
getResources().getDrawable(R.drawable.two);
 prev=3;
 }
 else if(prev==3) {
 drawable =
getResources().getDrawable(R.drawable.three);
 prev=4;
 }
 else if(prev==4) {
 drawable =
getResources().getDrawable(R.drawable.four);
 prev=5;
 }
 else if(prev==5) {
 drawable =
getResources().getDrawable(R.drawable.five);
 prev=1;
 }
 Bitmap wallpaper =
((BitmapDrawable)drawable).getBitmap();
 try {
 wpm.setBitmap(wallpaper);
 } catch (IOException e) {
 e.printStackTrace();
 }
 }
 },0,30000); } }