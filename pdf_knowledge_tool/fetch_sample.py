import urllib.request
import os

url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
output = "tests/sample.pdf"

os.makedirs("tests", exist_ok=True)
print(f"Downloading {url} to {output}...")
urllib.request.urlretrieve(url, output)
print("Done.")
