import yt_dlp

URL = "https://youtu.be/8fkHFzUSuag"

ydl_opts = {
    # 🔴 ชี้ ffmpeg
    "ffmpeg_location": r"C:\Users\usEr\AppData\Local\Overwolf\Extensions\ncfplpkmiejjaklknfnkgcpapnhkggmlcppckhcb\270.0.25\obs\bin\64bit",

    # โหลดวิดีโอ + เสียง
    "format": "bestvideo+bestaudio/best",

    # บังคับ merge เป็น mp4
    "merge_output_format": "mp4",

    # ชื่อไฟล์
    "outtmpl": "video.%(ext)s",

    # ❗ สำคัญ: บอก yt-dlp ว่าอย่าลบไฟล์ต้นฉบับ
    "keepvideo": True,

    # แยกเสียงเป็น wav
    "postprocessors": [
        {
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "192",
        }
    ],
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([URL])
