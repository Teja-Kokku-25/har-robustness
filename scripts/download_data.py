"""Download and unzip the UCI HAR dataset into data/UCI HAR Dataset/."""
import io, os, urllib.request, zipfile

URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
ROOT = os.path.join(os.path.dirname(__file__), "..", "data")

def main():
    os.makedirs(ROOT, exist_ok=True)
    if os.path.isdir(os.path.join(ROOT, "UCI HAR Dataset")):
        print("Already downloaded."); return
    print("Downloading...")
    outer = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(URL).read()))
    inner = next(n for n in outer.namelist() if n.endswith(".zip"))  # zip inside zip
    zipfile.ZipFile(io.BytesIO(outer.read(inner))).extractall(ROOT)
    print("Done ->", os.path.abspath(ROOT))

if __name__ == "__main__":
    main()
