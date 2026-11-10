import zipfile
import xml.etree.ElementTree as ET

path = r"docs/SNDA Project Template (1).docx"
NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

with zipfile.ZipFile(path) as z:
    with z.open("word/document.xml") as f:
        tree = ET.parse(f)

root = tree.getroot()
body = root.find(f".//{NS}body")

paras = []
for para in body.iter(f"{NS}p"):
    text = "".join(r.text or "" for r in para.iter(f"{NS}t")).strip()
    if text:
        paras.append(text)

print("\n".join(paras))
