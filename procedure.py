from pathlib import Path
from zipfile import ZipFile

archive_path = Path()/".tmp"/"predict-student-performance-from-game-play.zip"

with ZipFile(file=archive_path, mode="r") as f:
    f.extractall(".tmp")
