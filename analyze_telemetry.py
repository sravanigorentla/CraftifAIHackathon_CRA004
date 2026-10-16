import json

with open(r"c:\Users\kowsh\OneDrive\Desktop\hackathon\evidence\output\metadata.json", "r", encoding="utf-8") as f:
    data = json.load(f)

frames = data.get("frames", [])
print(f"Pipeline Name: {data.get('pipeline')}")
print(f"Total frames processed: {len(frames)}")
total_objs = sum(len(x.get("objects", [])) for x in frames)
print(f"Total objects detected: {total_objs}")

classes = {}
confidence_by_class = {}
for x in frames:
    for obj in x.get("objects", []):
        lbl = obj["label"]
        classes[lbl] = classes.get(lbl, 0) + 1
        confidence_by_class.setdefault(lbl, []).append(obj.get("confidence", 0))

print("\n--- Detection Summary by Class ---")
for k, v in sorted(classes.items(), key=lambda item: item[1], reverse=True):
    avg_conf = sum(confidence_by_class[k]) / len(confidence_by_class[k])
    print(f"  - {k:12s}: {v:5d} detections (Avg Confidence: {avg_conf:.1%})")

print("\nSample detection (Frame 10):")
print(json.dumps(frames[0] if frames else {}, indent=2))
